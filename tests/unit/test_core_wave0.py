from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cta import (
    AspectResult,
    CompareConfig,
    TraceArtifact,
    TrialGroup,
    compare,
    inspect_trace,
    load_manifest,
    register_adapter,
    register_aspect,
    run_comparison,
)
from cta.core.alignment import enrich_ops_df, normalize_signature
from cta.core.cache import CacheStore
from cta.core.diagnosis import parse_gate_spec
from cta.core.stats import compare_entity_tables, compare_scalar_metrics


@register_adapter("mock_trace", version="1.0")
class _MockAdapter:
    format_name = "mock_trace"
    version = "1.0"
    ingest_calls = 0

    def detect(self, path: Path) -> bool:
        return path.suffix == ".mocktrace"

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        _MockAdapter.ingest_calls += 1
        rows = [json.loads(p.read_text()) for p in artifact.paths]
        df = pl.DataFrame(rows)
        dst = out_dir / "metrics.parquet"
        df.write_parquet(dst)
        return {"metrics": dst}


@register_aspect("mock_perf", title="Mock Performance", required_formats=("mock_trace",))
class _MockAspect:
    name = "mock_perf"
    title = "Mock Performance"
    required_formats = ("mock_trace",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[TrialGroup, ...],
        tables: dict[str, list[dict[str, Path]]],
        config: CompareConfig,
    ) -> AspectResult:
        labels = [g.label for g in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}
        for g in groups:
            lat_vals, comm_vals, comp_vals = [], [], []
            for tmap in tables[g.group_id]:
                df = pl.read_parquet(tmap["mock_trace.metrics"])
                lat_vals.append(float(df["latency_ms"].cast(pl.Float64).sum()))
                comm_vals.append(float(df["exposed_comm"].to_numpy().mean()))
                comp_vals.append(float(df["compute"].to_numpy().mean()))
            trials.append({"latency_ms": lat_vals})
            tot = (sum(comp_vals) + sum(comm_vals)) or 1.0
            scalars[g.label] = {
                "share_Compute": sum(comp_vals) / tot,
                "share_Exposed-Comm": sum(comm_vals) / tot,
            }
        summary = compare_scalar_metrics(
            [("latency_ms", "ms", True)],
            labels,
            trials,
            alpha=config.alpha,
        )
        return AspectResult(name=self.name, title=self.title, summary=summary, scalars=scalars)


def test_cache_and_pair_comparison(tmp_path: Path) -> None:
    _MockAdapter.ingest_calls = 0
    cache_dir = tmp_path / ".cache"
    a = tmp_path / "base.mocktrace"
    b = tmp_path / "cand.mocktrace"
    a.write_text(json.dumps({"latency_ms": 100.0, "compute": 80.0, "exposed_comm": 20.0}))
    b.write_text(json.dumps({"latency_ms": 125.0, "compute": 35.0, "exposed_comm": 65.0}))

    rep1 = compare(a, b, gates={"latency_ms": 10.0}, cache_dir=cache_dir)
    assert _MockAdapter.ingest_calls == 2
    df = rep1.to_polars("mock_perf")
    assert df["delta_pct"][0] == pytest.approx(25.0)
    assert df["speedup"][0] == pytest.approx(0.8)
    assert len(rep1.violations) == 1
    assert rep1.diagnoses[0].baseline_regime == "Compute-Dominated"
    assert rep1.diagnoses[0].candidate_regime == "Exposed-Comm-Dominated"

    # Second call hits content-addressed Parquet cache without re-ingesting
    rep2 = compare(a, b, cache_dir=cache_dir)
    assert _MockAdapter.ingest_calls == 2
    assert rep2.to_pandas("mock_perf")["candidate"].iloc[0] == pytest.approx(125.0)

    info = inspect_trace(a, cache_dir=cache_dir)
    assert info["formats"]["mock_trace"]["tables"]["metrics"]["rows"] == 1
    assert CacheStore(cache_dir).status()["entries"] == 2
    assert CacheStore(cache_dir).clear() == 2


def test_manifest_multitrial_and_exporters(tmp_path: Path) -> None:
    for grp, vals in [("g0", [10.0, 10.2, 9.8]), ("g1", [5.0, 5.1, 4.9]), ("g2", [7.5, 7.6, 7.4])]:
        for idx, v in enumerate(vals):
            d = tmp_path / f"{grp}_t{idx}"
            d.mkdir()
            (d / "run.mocktrace").write_text(json.dumps({"latency_ms": v, "compute": 70.0, "exposed_comm": 30.0}))

    manifest = tmp_path / "sweep.yaml"
    manifest.write_text(
        """
groups:
  - id: base
    label: Baseline
    trials: [g0_t0, g0_t1, g0_t2]
  - id: fast
    label: Optimized
    trials: [g1_t0, g1_t1, g1_t2]
config:
  alpha: 0.05
  gates:
    latency_ms: 5.0
"""
    )
    groups, cfg = load_manifest(manifest)
    rep = run_comparison(groups, cfg)
    row = rep.to_polars("mock_perf").row(0, named=True)
    assert bool(row["significant"]) is True
    assert row["speedup"] == pytest.approx(2.0, rel=1e-2)
    assert len(rep.violations) == 0

    out = rep.export(tmp_path / "out", formats=("markdown", "latex", "csv", "json"))
    assert (tmp_path / "out" / "report.md").exists()
    assert "\\toprule" in (tmp_path / "out" / "report.tex").read_text()
    assert len(out) == 4


def test_alignment_and_entity_comparison() -> None:
    assert normalize_signature("void cutlass::Kernel<float, 128>(float const*, int)") == "cutlass::Kernel"
    df = pl.DataFrame({"name": ["ncclDevKernel_AllReduce_Sum_f16", "ampere_fp16_s16816gemm", "aten::layer_norm"]})
    enriched = enrich_ops_df(df)
    assert enriched["op_family"].to_list() == ["NCCL_AllReduce", "GEMM", "Normalization"]

    d0 = pl.DataFrame({"op_family": ["GEMM", "NCCL_AllReduce"], "dur_ms": [50.0, 20.0], "calls": [10, 5]})
    d1 = pl.DataFrame({"op_family": ["GEMM", "FlashAttention"], "dur_ms": [25.0, 15.0], "calls": [10, 4]})
    diff = compare_entity_tables(["Base", "Cand"], [d0, d1], key_col="op_family", val_col="dur_ms", count_col="calls")
    assert set(diff["op_family"].to_list()) == {"GEMM", "NCCL_AllReduce", "FlashAttention"}
    assert parse_gate_spec("p99:+5%, throughput:-10%") == {"p99": 5.0, "throughput": -10.0}
