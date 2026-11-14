"""Tests for the core comparison engine and its shared helpers.

A mock adapter and aspect exercise the Parquet cache, pairwise and
manifest-driven comparisons, gates, diagnoses and exporters. Further tests
cover kernel-name alignment, op-family classification, entity-table diffs,
lazy imports and the Chakra node-type constants.
"""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pytest

import cta
from cta.core import alignment
from cta.core import cache
from cta.core import diagnosis
from cta.core import stats


@cta.register_adapter("mock_trace", version="1.0")
class _MockAdapter:
    """Adapter for `.mocktrace` files, each holding one JSON row of metrics.

    Attributes:
        format_name: The format name the adapter is registered under.
        version: The adapter version.
        ingest_calls: How many times `ingest` ran; tests read it to detect
            cache hits.
    """

    format_name = "mock_trace"
    version = "1.0"
    ingest_calls = 0

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether `path` is a `.mocktrace` file.

        Args:
            path: The candidate trace file.

        Returns:
            True when the file has the `.mocktrace` suffix.
        """
        return path.suffix == ".mocktrace"

    def ingest(
        self, artifact: cta.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Writes the JSON rows of all files of `artifact` as one table.

        Args:
            artifact: The mock trace files to read.
            out_dir: Directory for the Parquet output.

        Returns:
            The path of the `metrics` table.
        """
        _MockAdapter.ingest_calls += 1
        rows = [json.loads(path.read_text()) for path in artifact.paths]
        metrics = pl.DataFrame(rows)
        metrics_path = out_dir / "metrics.parquet"
        metrics.write_parquet(metrics_path)
        return {"metrics": metrics_path}


@cta.register_aspect(
    "mock_perf", title="Mock Performance", required_formats=("mock_trace",)
)
class _MockAspect:
    """Aspect that compares the latency and time shares of mock traces.

    Attributes:
        name: The aspect name.
        title: The human-readable aspect title.
        required_formats: The trace formats the aspect needs.
        any_of_formats: Whether the aspect needs only one of
            `required_formats` rather than all of them.
    """

    name = "mock_perf"
    title = "Mock Performance"
    required_formats = ("mock_trace",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[cta.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: cta.CompareConfig,
    ) -> cta.AspectResult:
        """Compares summed latency and reports compute and comm shares.

        Args:
            groups: The trial groups being compared.
            tables: For each group id, one mapping of table name to Parquet
                path per trial.
            config: The comparison settings; only `alpha` is used.

        Returns:
            A `latency_ms` summary row and the time shares of each group.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}
        for group in groups:
            latencies_ms: list[float] = []
            exposed_comm: list[float] = []
            compute: list[float] = []
            for table_paths in tables[group.group_id]:
                metrics = pl.read_parquet(table_paths["mock_trace.metrics"])
                latency_ms = metrics["latency_ms"].cast(pl.Float64).sum()
                latencies_ms.append(float(latency_ms))
                exposed_comm.append(
                    float(metrics["exposed_comm"].to_numpy().mean())
                )
                compute.append(float(metrics["compute"].to_numpy().mean()))
            trials.append({"latency_ms": latencies_ms})
            total = (sum(compute) + sum(exposed_comm)) or 1.0
            scalars[group.label] = {
                "share_Compute": sum(compute) / total,
                "share_Exposed-Comm": sum(exposed_comm) / total,
            }
        summary = stats.compare_scalar_metrics(
            [("latency_ms", "ms", True)],
            labels,
            trials,
            alpha=config.alpha,
        )
        return cta.AspectResult(
            name=self.name, title=self.title, summary=summary, scalars=scalars
        )


def test_cache_and_pair_comparison(tmp_path: pathlib.Path) -> None:
    _MockAdapter.ingest_calls = 0
    cache_dir = tmp_path / ".cache"
    baseline_trace_path = tmp_path / "base.mocktrace"
    candidate_trace_path = tmp_path / "cand.mocktrace"
    baseline_trace_path.write_text(
        json.dumps({"latency_ms": 100.0, "compute": 80.0, "exposed_comm": 20.0})
    )
    candidate_trace_path.write_text(
        json.dumps({"latency_ms": 125.0, "compute": 35.0, "exposed_comm": 65.0})
    )

    first_report = cta.compare(
        baseline_trace_path,
        candidate_trace_path,
        gates={"latency_ms": 10.0},
        cache_dir=cache_dir,
    )
    assert _MockAdapter.ingest_calls == 2
    summary = first_report.to_polars("mock_perf")
    assert summary["delta_pct"][0] == pytest.approx(25.0)
    assert summary["speedup"][0] == pytest.approx(0.8)
    assert len(first_report.violations) == 1
    pair_diagnosis = first_report.diagnoses[0]
    assert pair_diagnosis.baseline_regime == "Compute-Dominated"
    assert pair_diagnosis.candidate_regime == "Exposed-Comm-Dominated"

    # The second call hits the content-addressed Parquet cache without
    # ingesting again.
    cached_report = cta.compare(
        baseline_trace_path, candidate_trace_path, cache_dir=cache_dir
    )
    assert _MockAdapter.ingest_calls == 2
    cached_summary = cached_report.to_pandas("mock_perf")
    assert cached_summary["candidate"].iloc[0] == pytest.approx(125.0)

    info = cta.inspect_trace(baseline_trace_path, cache_dir=cache_dir)
    assert info["formats"]["mock_trace"]["tables"]["metrics"]["rows"] == 1
    assert cache.CacheStore(cache_dir).status()["entries"] == 2
    assert cache.CacheStore(cache_dir).clear() == 2


def test_manifest_multitrial_and_exporters(tmp_path: pathlib.Path) -> None:
    for group_prefix, latencies_ms in [
        ("g0", [10.0, 10.2, 9.8]),
        ("g1", [5.0, 5.1, 4.9]),
        ("g2", [7.5, 7.6, 7.4]),
    ]:
        for trial, latency_ms in enumerate(latencies_ms):
            trial_dir = tmp_path / f"{group_prefix}_t{trial}"
            trial_dir.mkdir()
            trial_metrics = {
                "latency_ms": latency_ms,
                "compute": 70.0,
                "exposed_comm": 30.0,
            }
            (trial_dir / "run.mocktrace").write_text(json.dumps(trial_metrics))

    manifest_path = tmp_path / "sweep.yaml"
    manifest_path.write_text(
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
    groups, config = cta.load_manifest(manifest_path)
    report = cta.run_comparison(groups, config)
    row = report.to_polars("mock_perf").row(0, named=True)
    assert bool(row["significant"]) is True
    assert row["speedup"] == pytest.approx(2.0, rel=1e-2)
    assert len(report.violations) == 0

    out_dir = tmp_path / "out"
    written = report.export(
        out_dir, formats=("markdown", "latex", "csv", "json")
    )
    assert (out_dir / "report.md").exists()
    assert "\\toprule" in (out_dir / "report.tex").read_text()
    assert len(written) == 4


def test_alignment_and_entity_comparison() -> None:
    signature = alignment.normalize_signature(
        "void cutlass::Kernel<float, 128>(float const*, int)"
    )
    assert signature == "cutlass::Kernel"
    kernels = pl.DataFrame(
        {
            "name": [
                "ncclDevKernel_AllReduce_Sum_f16",
                "ampere_fp16_s16816gemm",
                "aten::layer_norm",
            ]
        }
    )
    enriched = alignment.enrich_ops_df(kernels)
    assert enriched["op_family"].to_list() == [
        "NCCL_AllReduce",
        "GEMM",
        "Normalization",
    ]

    baseline_families = pl.DataFrame(
        {
            "op_family": ["GEMM", "NCCL_AllReduce"],
            "dur_ms": [50.0, 20.0],
            "calls": [10, 5],
        }
    )
    candidate_families = pl.DataFrame(
        {
            "op_family": ["GEMM", "FlashAttention"],
            "dur_ms": [25.0, 15.0],
            "calls": [10, 4],
        }
    )
    diff = stats.compare_entity_tables(
        ["Base", "Cand"],
        [baseline_families, candidate_families],
        key_column="op_family",
        value_column="dur_ms",
        count_column="calls",
    )
    assert set(diff["op_family"].to_list()) == {
        "GEMM",
        "NCCL_AllReduce",
        "FlashAttention",
    }
    gates = diagnosis.parse_gate_spec("p99:+5%, throughput:-10%")
    assert gates == {"p99": 5.0, "throughput": -10.0}


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("ncclDevKernel_AllReduce_Sum_f32_RING_LL", "NCCL_AllReduce"),
        ("ncclDevKernel_ReduceScatter_Sum_f16_RING", "NCCL_ReduceScatter"),
        ("ncclDevKernel_SendRecv", "Comm_P2P"),
        ("sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32", "GEMM"),
        ("ampere_h884gemm_256x128_ldg8_nn", "GEMM"),
        ("void gemv2T_kernel_val<int, int, float>", "GEMM"),
        ("cublasLt::splitKreduce_kernel", "GEMM"),
        (
            "void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128>>",
            "FlashAttention",
        ),
        (
            "sm80_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_"
            "nhwckrsc_nhwc_tilesize64x64x32",
            "Convolution",
        ),
        ("aten::_convolution", "Convolution"),
        ("void cudnn::bn_fw_inf_1C11_kernel_NCHW", "Normalization"),
        (
            "void at::native::(anonymous namespace)::"
            "indexSelectLargeIndex<float>",
            "Embedding",
        ),
        (
            "void at::native::reduce_kernel<512, 1, "
            "at::native::ReduceOp<float, MeanOps>>",
            "Reduction",
        ),
        ("void cub::DeviceReduceKernel<float>", "Reduction"),
        ("triton_red_fused_sum_1", "Reduction"),
        (
            "void at::native::vectorized_elementwise_kernel<4, "
            "at::native::AddFunctor<float>>",
            "Elementwise",
        ),
        ("Memcpy HtoD (Pageable -> Device)", "Memcpy_H2D"),
        ("void comma_separated_kernel", "Other"),
        ("dilemma_kernel", "Other"),
    ],
)
def test_op_family_classification(name: str, family: str) -> None:
    kernels = pl.DataFrame({"name": [name]})
    op_family = alignment.op_family_expr("name").alias("op_family")
    assert kernels.select(op_family)["op_family"][0] == family


def test_import_cta_is_lazy() -> None:
    import subprocess
    import sys

    code = (
        "import sys, cta; "
        "heavy = [m for m in ('polars', 'scipy', 'pandas', 'hta', "
        "'matplotlib', 'google.protobuf', 'chakra') "
        "if m in sys.modules]; "
        "assert not heavy, heavy; "
        "cta.compare; assert 'polars' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_chakra_node_type_numbers_match_generated_enum() -> None:
    from cta.adapters.chakra import schema

    pb2 = schema.et_def_pb2()
    for node_type in (
        "MEM_LOAD_NODE",
        "MEM_STORE_NODE",
        "COMP_NODE",
        "COMM_SEND_NODE",
        "COMM_RECV_NODE",
        "COMM_COLL_NODE",
        "STORAGE_NODE",
        "DATA_OP_NODE",
    ):
        assert getattr(schema, node_type) == getattr(pb2, node_type), node_type


def test_compare_entity_tables_survives_an_empty_untyped_group() -> None:
    # For example, a capture without OS-runtime rows.
    empty = pl.DataFrame({"name": [], "duration_ms": [], "calls": []})
    full = pl.DataFrame({"name": ["read"], "duration_ms": [2.0], "calls": [3]})
    diff = stats.compare_entity_tables(
        ["a", "b"],
        [empty, full],
        key_column="name",
        value_column="duration_ms",
        count_column="calls",
    )
    assert diff["name"].to_list() == ["read"]
    assert diff["delta"].to_list() == [2.0]
