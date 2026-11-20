"""Tests of what-if repricing on an interconnect.

The default interconnect must leave every weight as it was, a slower network
must lengthen a graph whose communication has no recorded time, and recorded
times are scaled by new cost over baseline cost. The aspect and the command
take the same interconnect.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import polars as pl
import pytest
from typer import testing

import cta
from cta.analysis import interconnect as ic
from cta.analysis import whatif
from cta.cli import app as cli_app
from cta.synth import goal
from cta.synth import training

MIB = 1 << 20
cli_runner = testing.CliRunner()


# 1 us and 100 GB/s inside a node, 5 us and 25 GB/s between nodes.
NET = ic.Interconnect(
    gpus_per_node=8,
    intra=ic.Link(1000.0, 0.01),
    inter=ic.Link(5000.0, 0.04),
    algorithm="ring",
)


def _cost(
    kind: str, size: int, ranks: int, net: ic.Interconnect = NET
) -> float:
    return float(
        ic.collective_cost_ns(
            np.array([kind]), np.array([size]), np.array([ranks]), net
        )[0]
    )


def _nodes(rows: list[tuple]) -> pl.DataFrame:
    """Builds nodes from (rank, op_class, comm_type, ns, bytes, src, dst)."""
    return pl.DataFrame(
        {
            "rank": [r[0] for r in rows],
            "node_id": list(range(len(rows))),
            "name": [""] * len(rows),
            "op_class": [r[1] for r in rows],
            "comm_type": [r[2] for r in rows],
            "duration_ns": [r[3] for r in rows],
            "size_bytes": [r[4] for r in rows],
            "src_rank": [r[5] for r in rows],
            "dst_rank": [r[6] for r in rows],
            "message_tag": [0] * len(rows),
        },
        schema={
            "rank": pl.UInt32,
            "node_id": pl.UInt64,
            "name": pl.String,
            "op_class": pl.String,
            "comm_type": pl.String,
            "duration_ns": pl.Float64,
            "size_bytes": pl.UInt64,
            "src_rank": pl.Int32,
            "dst_rank": pl.Int32,
            "message_tag": pl.UInt32,
        },
    )


def _goal_model(tmp_path: pathlib.Path) -> whatif.WhatIfModel:
    path = tmp_path / "t.goal"
    training.write_goal(path, training.TrainingSpec(dp=4, tp=2, pp=2))
    return whatif.load_model(path, cache_dir=tmp_path / "cache")


def test_the_default_interconnect_leaves_every_weight_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    model = _goal_model(tmp_path)

    repriced = whatif.reprice(model, ic.NOMINAL)

    assert np.array_equal(repriced.graph.weights_ns, model.graph.weights_ns)
    assert (
        whatif.run_scenarios(repriced, []).table["makespan_ns"][0]
        == whatif.run_scenarios(model, []).table["makespan_ns"][0]
    )


def test_a_slower_network_lengthens_a_run_whose_comm_has_no_recorded_time(
    tmp_path: pathlib.Path,
) -> None:
    model = _goal_model(tmp_path)
    fast = ic.parse("gpus_per_node=4,intra=1us:300GB/s,inter=2us:100GB/s")
    slow = ic.parse("gpus_per_node=4,intra=1us:300GB/s,inter=2us:10GB/s")
    same_node = ic.parse("gpus_per_node=64,intra=1us:300GB/s,inter=2us:10GB/s")

    def makespan(net: ic.Interconnect) -> float:
        return float(
            whatif.run_scenarios(whatif.reprice(model, net), []).table[
                "makespan_ns"
            ][0]
        )

    # 16 ranks: with 4 per node most transfers between pipeline stages or
    # data-parallel peers cross nodes; with 64 per node none do.
    assert makespan(slow) > makespan(fast)
    assert makespan(same_node) == pytest.approx(
        makespan(ic.parse("gpus_per_node=64,intra=1us:300GB/s"))
    )


def test_recorded_comm_times_are_scaled_by_new_over_base() -> None:
    nodes = _nodes(
        [
            (0, "comm_coll", "ALL_REDUCE", 1000.0, MIB, -1, -1),
            (1, "comm_coll", "ALL_REDUCE", 1000.0, MIB, -1, -1),
            (0, "compute", "", 700.0, 0, -1, -1),
        ]
    )
    edges = pl.DataFrame(
        schema={
            "rank": pl.UInt32,
            "src_id": pl.UInt64,
            "dst_id": pl.UInt64,
            "dep_kind": pl.String,
        }
    )
    model = whatif.build_model(nodes, edges)
    base = ic.parse("algo=ring,group.ALL_REDUCE=8", base=NET)
    new = ic.parse("inter=5us:25GB/s,group.ALL_REDUCE=16", base=base)

    repriced = whatif.reprice(model, new, base)

    ratio = _cost("ALL_REDUCE", MIB, 16, new) / _cost(
        "ALL_REDUCE", MIB, 8, base
    )
    assert repriced.graph.weights_ns[0] == pytest.approx(1000.0 * ratio)
    assert repriced.graph.weights_ns[1] == pytest.approx(1000.0 * ratio)
    assert repriced.graph.weights_ns[2] == 700.0


def test_recorded_comm_times_are_kept_without_a_baseline_interconnect() -> None:
    nodes = _nodes([(0, "comm_coll", "ALL_REDUCE", 1000.0, MIB, -1, -1)])
    edges = pl.DataFrame(
        schema={
            "rank": pl.UInt32,
            "src_id": pl.UInt64,
            "dst_id": pl.UInt64,
            "dep_kind": pl.String,
        }
    )
    model = whatif.build_model(nodes, edges)

    with pytest.warns(RuntimeWarning, match="no baseline interconnect"):
        repriced = whatif.reprice(model, NET)

    assert repriced.graph.weights_ns.tolist() == [1000.0]


def test_the_aspect_prices_the_baseline_on_the_interconnect_asked_for(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "t.goal"
    training.write_goal(path, training.TrainingSpec(dp=4, tp=2, pp=2))

    plain = cta.compare(
        path,
        path,
        aspects=("whatif",),
        cache_dir=tmp_path / "cache",
        predict="comm:2x",
    )
    slow = cta.compare(
        path,
        path,
        aspects=("whatif",),
        cache_dir=tmp_path / "cache",
        predict="comm:2x",
        interconnect="gpus_per_node=2,intra=1us:100GB/s,inter=5us:2GB/s",
    )

    def metric(report: cta.Report, name: str) -> dict[str, float]:
        return (
            report.aspects["whatif"]
            .summary.filter(pl.col("metric") == name)
            .row(0, named=True)
        )

    # The baseline's own makespan is the measured one either way.
    assert metric(slow, "whatif_makespan_us")["baseline"] == pytest.approx(
        metric(plain, "whatif_makespan_us")["baseline"]
    )
    # Predicting it on a much slower network gives a longer predicted run.
    assert (
        metric(slow, "whatif_residual")["baseline"]
        > metric(plain, "whatif_residual")["baseline"]
    )


def test_the_whatif_command_reprices_and_rejects_a_bad_spec(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "t.goal"
    goal.write_allreduce(path, world_size=4)
    base = [
        "whatif",
        str(path),
        "--cache-dir",
        str(tmp_path / "c"),
        "--format",
        "json",
        "--speed",
        "comm:2x",
    ]

    plain = cli_runner.invoke(cli_app.app, base)
    slow = cli_runner.invoke(
        cli_app.app,
        [*base, "--interconnect", "gpus_per_node=1,inter=50us:0.1GB/s"],
    )
    bad = cli_runner.invoke(cli_app.app, [*base, "--interconnect", "speed=3"])

    assert plain.exit_code == 0 and slow.exit_code == 0
    plain_ms = json.loads(plain.stdout)["scenarios"][0]["makespan_ns"]
    slow_ms = json.loads(slow.stdout)["scenarios"][0]["makespan_ns"]
    assert slow_ms > plain_ms
    assert bad.exit_code == 2


def test_no_warning_without_an_interconnect(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "t.goal"
    goal.write_allreduce(path, world_size=4)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cta.compare(
            path, path, aspects=("whatif",), cache_dir=tmp_path / "cache"
        )
