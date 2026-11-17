"""Tests of ``cta simulate``."""

from __future__ import annotations

import json
import pathlib

from typer import testing

from cta.cli import app as cli_app

runner = testing.CliRunner()


def invoke(*args: str) -> testing.Result:
    # Wide enough that Rich does not shorten the table cells.
    return runner.invoke(
        cli_app.app, ["simulate", *args], env={"COLUMNS": "250"}
    )


def test_run_prints_and_writes(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    cache = str(training_traces["cache"])
    result = invoke(
        "run",
        str(training_traces["chakra"]),
        "--cache-dir",
        cache,
        "--model",
        "flow",
        "--set",
        "topology.kind=fat_tree",
        "--out-dir",
        str(tmp_path / "out"),
        "--trace",
        str(tmp_path / "trace.json"),
    )
    assert result.exit_code == 0, result.output
    assert "Critical path" in result.output
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "links.parquet").exists()
    assert (tmp_path / "trace.json").exists()

    as_json = invoke(
        "run",
        str(training_traces["goal"]),
        "--cache-dir",
        cache,
        "--format",
        "json",
    )
    assert as_json.exit_code == 0
    assert json.loads(as_json.output)["summary"]["makespan_ns"] > 0


def test_compare_across_formats(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    result = invoke(
        "compare",
        str(training_traces["goal"]),
        str(training_traces["chakra"]),
        "--labels",
        "goal,chakra",
        "--cache-dir",
        str(training_traces["cache"]),
        "--out-dir",
        str(tmp_path),
        "--export",
        "markdown",
        "--plot",
        "sim_breakdown",
    )
    assert result.exit_code == 0, result.output
    assert "chakra [analytical]" in result.output
    assert "Makespan gap by critical-path segment" in result.output
    assert (tmp_path / "sim_breakdown.png").exists()
    assert (tmp_path / "overview.parquet").exists()


def test_collective_presets_settings_and_lower(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    collective = invoke(
        "collective",
        "ALL_REDUCE",
        "-n",
        "8",
        "-b",
        "1MiB",
        "--schedule",
        "ring",
    )
    assert collective.exit_code == 0, collective.output
    assert "halving_doubling" in collective.output
    assert invoke("presets").exit_code == 0
    assert "loggops.L" in invoke("settings").output
    out = tmp_path / "lowered.goal"
    lowered = invoke(
        "lower",
        str(training_traces["chakra"]),
        str(out),
        "--cache-dir",
        str(training_traces["cache"]),
    )
    assert lowered.exit_code == 0, lowered.output
    assert out.read_text().startswith("num_ranks 8")


def test_sweep(training_traces: dict[str, pathlib.Path]) -> None:
    result = invoke(
        "sweep",
        str(training_traces["goal"]),
        "--cache-dir",
        str(training_traces["cache"]),
        "--param",
        "loggops.G=6,3",
        "--trials",
        "2",
        "--set",
        "noise.frequency=1kHz",
        "--set",
        "noise.duration=10us",
    )
    assert result.exit_code == 0, result.output
    assert "Monte Carlo over 2 seeds" in result.output


def test_bad_settings_exit_with_an_error(
    training_traces: dict[str, pathlib.Path],
) -> None:
    result = invoke("run", str(training_traces["goal"]), "--set", "loggops.Q=1")
    assert result.exit_code == 2
