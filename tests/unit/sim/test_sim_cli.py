"""Tests of ``cta simulate``."""

from __future__ import annotations

import json
import pathlib

from typer import testing

from cta.cli import app as cli_app


def invoke(*args: str) -> testing.Result:
    # Wide enough that Rich does not shorten cells.
    return testing.CliRunner().invoke(
        cli_app.app, ["simulate", *args], env={"COLUMNS": "250"}
    )


def test_run(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    cache = str(training_traces["cache"])
    result = invoke(
        "run",
        str(training_traces["chakra"]),
        "--cache-dir",
        cache,
        "-m",
        "flow",
        "-s",
        "topology.kind=fat_tree",
        "-o",
        str(tmp_path),
        "--trace",
        str(tmp_path / "trace.json"),
    )
    assert result.exit_code == 0, result.output
    assert "Critical path" in result.output and "Busiest links" in result.output
    assert {"summary.json", "links.parquet", "trace.json"} <= {
        p.name for p in tmp_path.iterdir()
    }
    as_json = invoke(
        "run", str(training_traces["goal"]), "--cache-dir", cache, "-f", "json"
    )
    assert json.loads(as_json.output)["summary"]["makespan_ns"] > 0
    assert (
        invoke(
            "run", str(training_traces["goal"]), "-s", "loggops.Q=1"
        ).exit_code
        == 2
    )


def test_compare(
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
        "-o",
        str(tmp_path),
        "--export",
        "markdown",
        "--plot",
        "sim_breakdown",
    )
    assert result.exit_code == 0, result.output
    assert "chakra [analytical]" in result.output
    assert "Makespan gap by critical-path segment" in result.output
    assert (tmp_path / "sim_breakdown.png").exists() and (
        tmp_path / "overview.parquet"
    ).exists()


def test_other_commands(
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
    settings = invoke("settings").output
    assert "loggops.L" in settings and "dgx-h100" in settings
    lowered = invoke(
        "lower",
        str(training_traces["chakra"]),
        str(tmp_path / "x.goal"),
        "--cache-dir",
        str(training_traces["cache"]),
    )
    assert lowered.exit_code == 0, lowered.output
    assert (tmp_path / "x.goal").read_text().startswith("num_ranks 8")
    swept = invoke(
        "sweep",
        str(training_traces["goal"]),
        "--cache-dir",
        str(training_traces["cache"]),
        "-p",
        "loggops.G=6,3",
        "--trials",
        "2",
        "-s",
        "noise.frequency=1kHz",
        "-s",
        "noise.duration=10us",
    )
    assert swept.exit_code == 0, swept.output
    assert "Monte Carlo over 2 seeds" in swept.output
