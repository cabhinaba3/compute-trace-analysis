"""Regression tests for the CLI and the SDK entry points behind it.

Covers gate parsing and unknown-gate handling, version reporting, the `list`
command, machine-readable JSON output, exports, manifest overrides, cache
hits for copied traces, group immutability, DAG cycle warnings and the
opt-in HTA aspect.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from typer import testing

import cta
from cta.cli import app as cli_app
from cta.core import diagnosis
from cta.synth import chakra

cli_runner = testing.CliRunner()

# Baseline run directory, candidate run directory and cache directory.
ChakraPair = tuple[pathlib.Path, pathlib.Path, pathlib.Path]


@pytest.fixture(scope="module")
def chakra_pair(tmp_path_factory: pytest.TempPathFactory) -> ChakraPair:
    """Writes two 4-rank Chakra-ET runs shared by the tests of this module.

    Args:
        tmp_path_factory: Pytest's factory for module-scoped temp dirs.

    Returns:
        The baseline run directory, the candidate run directory (twice the
        all-reduce traffic and a straggler on rank 2) and a cache directory.
    """
    root = tmp_path_factory.mktemp("review")
    baseline_dir, candidate_dir = root / "a", root / "b"
    chakra.write_chakra_et_run(
        baseline_dir, world_size=4, compute_us=800, allreduce_bytes=400_000
    )
    chakra.write_chakra_et_run(
        candidate_dir,
        world_size=4,
        compute_us=800,
        allreduce_bytes=800_000,
        straggler_rank=2,
        straggler_factor=1.8,
    )
    return baseline_dir, candidate_dir, root / "cache"


def test_parse_gate_spec_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="not a number"):
        diagnosis.parse_gate_spec("p99:abc%")


def test_unknown_gate_metric_fails_unless_allowed(
    chakra_pair: ChakraPair,
) -> None:
    baseline_dir, candidate_dir, cache_dir = chakra_pair
    with pytest.raises(ValueError, match="no_such_metric"):
        cta.compare(
            baseline_dir,
            candidate_dir,
            gates={"no_such_metric": 5.0},
            cache_dir=cache_dir,
        )
    report = cta.compare(
        baseline_dir,
        candidate_dir,
        gates={"no_such_metric": 5.0},
        allow_missing_gates=True,
        cache_dir=cache_dir,
    )
    assert report.violations == []

    compare_args = [
        "compare",
        str(baseline_dir),
        str(candidate_dir),
        "--cache-dir",
        str(cache_dir),
        "--gate",
        "no_such_metric:+5%",
    ]
    assert cli_runner.invoke(cli_app.app, compare_args).exit_code == 2
    allowed = cli_runner.invoke(
        cli_app.app, [*compare_args, "--allow-missing-gates"]
    )
    assert allowed.exit_code == 0
    bad_gate = cli_runner.invoke(
        cli_app.app,
        ["compare", str(baseline_dir), str(candidate_dir), "--gate", "oops"],
    )
    assert bad_gate.exit_code == 2
    assert "Traceback" not in bad_gate.output


def test_version_and_module_entry() -> None:
    result = cli_runner.invoke(cli_app.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"cta {cta.__version__}"
    assert (pathlib.Path(cta.__file__).parent / "__main__.py").exists()


def test_version_single_sourced() -> None:
    repo_root = pathlib.Path(cta.__file__).resolve().parent.parent
    pyproject = (repo_root / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    citation = (repo_root / "CITATION.cff").read_text()
    assert f'version: "{cta.__version__}"' in citation
    codemeta = json.loads((repo_root / "codemeta.json").read_text())
    assert codemeta["version"] == cta.__version__


def test_list_shows_formats_and_aspects() -> None:
    result = cli_runner.invoke(cli_app.app, ["list"])
    assert result.exit_code == 0
    # Rich tables may wrap long names, so also search with whitespace removed.
    compact_output = re.sub(r"\s+", "", result.output)
    for token in ("chakra", "goal", "latency_throughput", "kernel_diff"):
        assert token in compact_output or token in result.output


def test_format_json_stdout_is_pure_json(
    chakra_pair: ChakraPair,
    tmp_path: pathlib.Path,
) -> None:
    baseline_dir, candidate_dir, cache_dir = chakra_pair
    runner_split = testing.CliRunner()
    result = runner_split.invoke(
        cli_app.app,
        [
            "compare",
            str(baseline_dir),
            str(candidate_dir),
            "--cache-dir",
            str(cache_dir),
            "--format",
            "json",
            "-o",
            str(tmp_path),
            "--export",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {"mode", "groups", "aspects", "details", "scalars"} <= set(payload)


def test_export_writes_every_applicable_plot_kind(
    chakra_pair: ChakraPair,
    tmp_path: pathlib.Path,
) -> None:
    baseline_dir, candidate_dir, cache_dir = chakra_pair
    report = cta.compare(baseline_dir, candidate_dir, cache_dir=cache_dir)
    written = report.export(tmp_path, formats=("pdf", "json"))
    pdf_keys = {key for key in written if key.startswith("pdf:")}
    assert len(pdf_keys) >= 2
    assert all(written[key].exists() for key in pdf_keys)
    exported = json.loads(written["json"].read_text())
    assert exported["details"] and exported["scalars"]


def test_manifest_mode_honors_cli_overrides(tmp_path: pathlib.Path) -> None:
    run_a_dir, run_b_dir = tmp_path / "a", tmp_path / "b"
    chakra.write_chakra_et_run(
        run_a_dir, world_size=2, compute_us=500, allreduce_bytes=100_000
    )
    chakra.write_chakra_et_run(
        run_b_dir, world_size=2, compute_us=500, allreduce_bytes=200_000
    )
    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(
        "groups:\n"
        "  - {id: a, label: A, path: a}\n"
        "  - {id: b, label: B, path: b}\n"
        "config:\n"
        "  collective_algorithm: ring\n"
    )
    result = cli_runner.invoke(
        cli_app.app,
        [
            "compare",
            "--manifest",
            str(manifest_path),
            "--collective-algorithm",
            "tree",
            "--cache-dir",
            str(tmp_path / "c"),
            str(run_a_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ignored when --manifest" in result.output


def test_cache_hits_for_copied_trace(
    chakra_pair: ChakraPair,
    tmp_path: pathlib.Path,
) -> None:
    import os
    import shutil

    from cta.core import bundle
    from cta.core import cache as core_cache

    original_dir, _, _ = chakra_pair
    copy_dir = tmp_path / "copy"
    shutil.copytree(original_dir, copy_dir)
    for path in copy_dir.rglob("*"):
        if path.is_file():
            # A different mtime, but identical content.
            os.utime(path, ns=(1, 1))
    original = bundle.resolve_bundle(original_dir).artifacts["chakra_et"]
    copied = bundle.resolve_bundle(copy_dir).artifacts["chakra_et"]
    original_digest = core_cache.compute_artifact_digest(original, "1.0")
    copied_digest = core_cache.compute_artifact_digest(copied, "1.0")
    assert original_digest == copied_digest


def test_compare_does_not_mutate_caller_groups(
    chakra_pair: ChakraPair,
) -> None:
    baseline_dir, _, cache_dir = chakra_pair
    first_group = cta.resolve_trial_group(baseline_dir, label="same")
    second_group = cta.resolve_trial_group(baseline_dir, label="same")
    report = cta.compare(first_group, second_group, cache_dir=cache_dir)
    assert (first_group.label, second_group.label) == ("same", "same")
    assert [group.label for group in report.groups] == ["same_0", "same_1"]


def test_dag_cycle_warns() -> None:
    import polars as pl

    from cta.core import dag

    nodes = pl.DataFrame(
        {
            "rank": [0, 0],
            "node_id": [1, 2],
            "op_class": ["compute", "compute"],
            "src_rank": [0, 0],
            "dst_rank": [0, 0],
            "message_tag": [0, 0],
            "duration_ns": [10.0, 10.0],
            "size_bytes": [0, 0],
        }
    )
    edges = pl.DataFrame({"rank": [0, 0], "src_id": [1, 2], "dst_id": [2, 1]})
    with pytest.warns(RuntimeWarning, match="cycle"):
        dag.solve_distributed_dag(nodes, edges)


def test_hta_aspect_is_opt_in_and_wraps_hta(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("hta")
    from cta.synth import kineto

    baseline_trace_path = tmp_path / "a.json.gz"
    candidate_trace_path = tmp_path / "b.json.gz"
    kineto.write_kineto_trace(baseline_trace_path, attn_us=450.0, gemm_us=600.0)
    kineto.write_kineto_trace(
        candidate_trace_path, attn_us=210.0, gemm_us=280.0, kernel_variant="fp8"
    )
    cache_dir = tmp_path / "cache"

    # HTA is not run by default.
    default_report = cta.compare(
        baseline_trace_path, candidate_trace_path, cache_dir=cache_dir
    )
    assert "hta_kineto" not in default_report.aspects

    report = cta.compare(
        baseline_trace_path,
        candidate_trace_path,
        aspects=["hta_kineto"],
        cache_dir=cache_dir,
    )
    summary = report.to_polars("hta_kineto")
    metrics = set(summary["metric"].to_list())
    assert {
        "idle_host_wait_us",
        "launch_delay_mean_us",
        "queue_length_max",
    } <= metrics
    details = report.aspect("hta_kineto").details
    assert any(name.startswith("launch_stats[") for name in details)

    result = cli_runner.invoke(
        cli_app.app,
        [
            "compare",
            str(baseline_trace_path),
            str(candidate_trace_path),
            "-a",
            "hta_kineto",
            "--cache-dir",
            str(cache_dir),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    # HTA's progress logging must not leak into machine-readable stdout.
    json.loads(result.stdout)
