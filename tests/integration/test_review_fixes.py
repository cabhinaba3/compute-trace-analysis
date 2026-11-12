from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cta
from cta.cli.app import app
from cta.core.diagnosis import parse_gate_spec
from cta.synth.generators import write_chakra_et_run

runner = CliRunner()


@pytest.fixture(scope="module")
def chakra_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("review")
    a, b = root / "a", root / "b"
    write_chakra_et_run(a, world_size=4, compute_us=800, allreduce_bytes=400_000)
    write_chakra_et_run(b, world_size=4, compute_us=800, allreduce_bytes=800_000, straggler_rank=2, straggler_factor=1.8)
    return a, b, root / "cache"


def test_parse_gate_spec_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="not a number"):
        parse_gate_spec("p99:abc%")


def test_unknown_gate_metric_fails_unless_allowed(chakra_pair: tuple[Path, Path, Path]) -> None:
    a, b, cache = chakra_pair
    with pytest.raises(ValueError, match="no_such_metric"):
        cta.compare(a, b, gates={"no_such_metric": 5.0}, cache_dir=cache)
    rep = cta.compare(a, b, gates={"no_such_metric": 5.0}, allow_missing_gates=True, cache_dir=cache)
    assert rep.violations == []

    base = ["compare", str(a), str(b), "--cache-dir", str(cache), "--gate", "no_such_metric:+5%"]
    assert runner.invoke(app, base).exit_code == 2
    assert runner.invoke(app, [*base, "--allow-missing-gates"]).exit_code == 0
    bad = runner.invoke(app, ["compare", str(a), str(b), "--gate", "oops"])
    assert bad.exit_code == 2 and "Traceback" not in bad.output


def test_version_and_module_entry() -> None:
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0 and res.output.strip() == f"cta {cta.__version__}"
    assert (Path(cta.__file__).parent / "__main__.py").exists()


def test_version_single_sourced() -> None:
    root = Path(cta.__file__).resolve().parent.parent
    assert 'dynamic = ["version"]' in (root / "pyproject.toml").read_text()
    assert f'version: "{cta.__version__}"' in (root / "CITATION.cff").read_text()
    assert json.loads((root / "codemeta.json").read_text())["version"] == cta.__version__


def test_list_shows_formats_and_aspects() -> None:
    res = runner.invoke(app, ["list"])
    assert res.exit_code == 0
    for token in ("chakra", "goal", "latency_throughput", "kernel_diff"):
        assert token in re.sub(r"\s+", "", res.output) or token in res.output


def test_format_json_stdout_is_pure_json(chakra_pair: tuple[Path, Path, Path], tmp_path: Path) -> None:
    a, b, cache = chakra_pair
    runner_split = CliRunner()
    res = runner_split.invoke(
        app,
        ["compare", str(a), str(b), "--cache-dir", str(cache), "--format", "json", "-o", str(tmp_path), "--export", "json"],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert {"mode", "groups", "aspects", "details", "scalars"} <= set(payload)


def test_export_writes_every_applicable_plot_kind(chakra_pair: tuple[Path, Path, Path], tmp_path: Path) -> None:
    a, b, cache = chakra_pair
    rep = cta.compare(a, b, cache_dir=cache)
    written = rep.export(tmp_path, formats=("pdf", "json"))
    pdfs = {k for k in written if k.startswith("pdf:")}
    assert len(pdfs) >= 2
    assert all(written[k].exists() for k in pdfs)
    data = json.loads(written["json"].read_text())
    assert data["details"] and data["scalars"]


def test_manifest_mode_honors_cli_overrides(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_chakra_et_run(a, world_size=2, compute_us=500, allreduce_bytes=100_000)
    write_chakra_et_run(b, world_size=2, compute_us=500, allreduce_bytes=200_000)
    man = tmp_path / "m.yaml"
    man.write_text("groups:\n  - {id: a, label: A, path: a}\n  - {id: b, label: B, path: b}\nconfig:\n  collective_algo: ring\n")
    res = runner.invoke(
        app, ["compare", "--manifest", str(man), "--collective-algo", "tree", "--cache-dir", str(tmp_path / "c"), str(a)]
    )
    assert res.exit_code == 0, res.output
    assert "ignored when --manifest" in res.output


def test_cache_hits_for_copied_trace(chakra_pair: tuple[Path, Path, Path], tmp_path: Path) -> None:
    import os
    import shutil

    from cta.core.bundle import resolve_bundle
    from cta.core.cache import compute_artifact_digest

    a, _, _ = chakra_pair
    copy = tmp_path / "copy"
    shutil.copytree(a, copy)
    for f in copy.rglob("*"):
        if f.is_file():
            os.utime(f, ns=(1, 1))  # different mtime, identical content
    art_a = resolve_bundle(a).artifacts["chakra_et"]
    art_c = resolve_bundle(copy).artifacts["chakra_et"]
    assert compute_artifact_digest(art_a, "1.0") == compute_artifact_digest(art_c, "1.0")


def test_compare_does_not_mutate_caller_groups(chakra_pair: tuple[Path, Path, Path]) -> None:
    a, _, cache = chakra_pair
    g1 = cta.resolve_trial_group(a, label="same")
    g2 = cta.resolve_trial_group(a, label="same")
    rep = cta.compare(g1, g2, cache_dir=cache)
    assert (g1.label, g2.label) == ("same", "same")
    assert [g.label for g in rep.groups] == ["same_0", "same_1"]


def test_dag_cycle_warns() -> None:
    import polars as pl

    from cta.core.dag import solve_distributed_dag

    nodes = pl.DataFrame(
        {
            "rank": [0, 0],
            "node_id": [1, 2],
            "op_class": ["compute", "compute"],
            "src_rank": [0, 0],
            "dst_rank": [0, 0],
            "tag": [0, 0],
            "duration_ns": [10.0, 10.0],
            "size_bytes": [0, 0],
        }
    )
    edges = pl.DataFrame({"rank": [0, 0], "src_id": [1, 2], "dst_id": [2, 1]})
    with pytest.warns(RuntimeWarning, match="cycle"):
        solve_distributed_dag(nodes, edges)


def test_hta_aspect_is_opt_in_and_wraps_hta(tmp_path: Path) -> None:
    pytest.importorskip("hta")
    from cta.synth.generators import write_kineto_trace

    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    write_kineto_trace(a, attn_us=450.0, gemm_us=600.0)
    write_kineto_trace(b, attn_us=210.0, gemm_us=280.0, kernel_variant="fp8")
    cache = tmp_path / "cache"

    assert "hta_kineto" not in cta.compare(a, b, cache_dir=cache).aspects  # not run by default

    rep = cta.compare(a, b, aspects=["hta_kineto"], cache_dir=cache)
    summary = rep.to_polars("hta_kineto")
    metrics = set(summary["metric"].to_list())
    assert {"idle_host_wait_us", "launch_delay_mean_us", "queue_length_max"} <= metrics
    assert any(k.startswith("launch_stats[") for k in rep.aspect("hta_kineto").details)

    res = runner.invoke(app, ["compare", str(a), str(b), "-a", "hta_kineto", "--cache-dir", str(cache), "--format", "json"])
    assert res.exit_code == 0, res.output
    json.loads(res.stdout)  # HTA's progress logging must not leak into machine-readable stdout
