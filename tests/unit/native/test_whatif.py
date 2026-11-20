"""Tests of the what-if analysis (`cta.analysis.whatif`, `cta._native.DagPlan`).

The native solver must give the unchanged graph the same answers as
`cta._native.solve_dag`, and a scenario the same answers as solving the graph
again with the scaled weights.
"""

from __future__ import annotations

import json
import math
import pathlib
import random

import numpy as np
import polars as pl
import pytest
from typer import testing

import cta
from cta import _native as native
from cta.analysis import dag
from cta.analysis import whatif
from cta.cli import app as cli_app
from cta.synth import goal

cli_runner = testing.CliRunner()


def _tables(
    ops: list[tuple[int, int, str, str, float]],
    edges: list[tuple[int, int, int]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds node and edge tables from (rank, id, op_class, name, ns)."""
    nodes = pl.DataFrame(
        {
            "rank": [op[0] for op in ops],
            "node_id": [op[1] for op in ops],
            "name": [op[3] for op in ops],
            "op_class": [op[2] for op in ops],
            "comm_type": [
                "ALL_REDUCE" if op[2] == "comm_coll" else "" for op in ops
            ],
            "duration_ns": [op[4] for op in ops],
            "size_bytes": [1000 * (i % 7) for i in range(len(ops))],
            "src_rank": [-1] * len(ops),
            "dst_rank": [-1] * len(ops),
            "message_tag": [0] * len(ops),
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
    edge_table = pl.DataFrame(
        edges,
        schema={"rank": pl.UInt32, "src_id": pl.UInt64, "dst_id": pl.UInt64},
        orient="row",
    )
    return nodes, edge_table


def _random_tables(
    seed: int, n: int, ranks: int, integer: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns a random graph; integer durations make many ties."""
    rng = random.Random(seed)
    ids = [0] * ranks
    ops: list[tuple[int, int, str, str, float]] = []
    names = ("gemm_a", "gemm_b", "softmax", "nccl_allreduce", "add")
    for _ in range(n):
        rank = rng.randrange(ranks)
        ids[rank] += 1
        kind = rng.choice(["compute", "compute", "comm_coll"])
        duration = (
            float(rng.randrange(1, 6) * 10)
            if integer
            else rng.uniform(1.0, 100.0)
        )
        name = rng.choice(names) if kind == "compute" else "nccl_allreduce"
        ops.append((rank, ids[rank], kind, name, duration))
    edges = []
    for rank in range(ranks):
        for node_id in range(2, ids[rank] + 1):
            for _ in range(rng.choice([1, 1, 2])):
                edges.append((rank, rng.randrange(1, node_id), node_id))
    return _tables(ops, edges)


@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_unchanged_graph_matches_the_solver(seed: int, integer: bool) -> None:
    nodes, edges = _random_tables(seed, 400, 4, integer)
    expected = dag.solve_distributed_dag(nodes, edges)
    run = whatif.run_scenarios(whatif.build_model(nodes, edges), [])
    row = run.table.row(0, named=True)
    assert row["scenario"] == "baseline"
    assert row["makespan_ns"] == expected.critical_path_ns
    assert row["cp_compute_ns"] == expected.cp_compute_ns
    assert row["cp_exposed_comm_ns"] == expected.cp_exposed_comm_ns
    assert row["cp_comm_bytes"] == expected.cp_comm_bytes
    assert row["dag_depth"] == expected.dag_depth


@pytest.mark.parametrize("integer", [False, True])
def test_scenario_equals_solving_the_scaled_graph(integer: bool) -> None:
    nodes, edges = _random_tables(11, 600, 3, integer)
    model = whatif.build_model(nodes, edges)
    graph = model.graph
    run = whatif.run_scenarios(
        model,
        [
            whatif.parse_scenario("comm:2x"),
            whatif.parse_scenario("name~gemm:1.5x;class=comm_coll:remove"),
            whatif.parse_scenario("rank=1:0.5x"),
        ],
    )
    multipliers = [
        np.where(graph.is_comm, 0.5, 1.0),
        np.where(
            nodes["name"].str.contains("(?i)gemm").to_numpy(), 1 / 1.5, 1.0
        )
        * np.where(nodes["op_class"].to_numpy() == "comm_coll", 0.0, 1.0),
        np.where(nodes["rank"].to_numpy() == 1, 2.0, 1.0),
    ]
    for index, multiplier in enumerate(multipliers, start=1):
        solved = native.solve_dag(
            graph.weights_ns * multiplier,
            graph.is_comm,
            graph.is_compute,
            graph.size_bytes,
            graph.edge_src,
            graph.edge_dst,
        )
        row = run.table.row(index, named=True)
        assert row["makespan_ns"] == solved["makespan"]
        assert row["cp_compute_ns"] == solved["critical_compute"]
        assert row["cp_exposed_comm_ns"] == solved["critical_comm"]
        assert row["dag_depth"] == solved["depth"]


def test_critical_time_by_group_adds_up_to_the_makespan() -> None:
    nodes, edges = _random_tables(5, 500, 4, False)
    run = whatif.run_scenarios(
        whatif.build_model(nodes, edges),
        [whatif.parse_scenario("comm:3x"), whatif.parse_scenario("rank=2:2x")],
    )
    totals = run.critical_by_group.sum(axis=1)
    assert totals == pytest.approx(run.table["makespan_ns"].to_numpy())


def test_results_do_not_depend_on_threads_or_lanes() -> None:
    nodes, edges = _random_tables(3, 800, 4, False)
    model = whatif.build_model(nodes, edges)
    scenarios = [
        whatif.parse_scenario(f"rank={rank}:{factor}x", f"r{rank}x{factor}")
        for rank in range(4)
        for factor in (1.5, 2, 3)
    ]
    reference = whatif.run_scenarios(model, scenarios, threads=1)
    for threads in (2, 7):
        other = whatif.run_scenarios(model, scenarios, threads=threads)
        assert other.table.equals(reference.table)
        assert np.array_equal(
            other.critical_by_group, reference.critical_by_group
        )
    # A budget for one worker and one lane forces the narrowest solve.
    narrow = whatif.run_scenarios(
        model, scenarios, threads=2, memory_budget_bytes=12 * 800 + 1
    )
    assert narrow.lanes == 1 and narrow.workers == 1
    assert narrow.table.equals(reference.table)


def test_a_graph_that_does_not_fit_the_budget_raises() -> None:
    nodes, edges = _random_tables(1, 200, 2, False)
    model = whatif.build_model(nodes, edges)
    with pytest.raises(RuntimeError, match="budget"):
        whatif.run_scenarios(model, [], memory_budget_bytes=100)


def test_slowest_rank_at_median_speed() -> None:
    nodes, edges = _tables(
        [
            (0, 1, "compute", "k", 100.0),
            (1, 1, "compute", "k", 100.0),
            (2, 1, "compute", "k", 100.0),
            (3, 1, "compute", "k", 400.0),
        ],
        [],
    )
    model = whatif.build_model(nodes, edges)
    run = whatif.run_scenarios(
        model,
        [
            whatif.parse_scenario("rank=3:median"),
            whatif.parse_scenario("rank=*:median"),
            whatif.parse_scenario("rank=0:median"),
        ],
    )
    assert run.table["makespan_ns"].to_list() == [400.0, 100.0, 100.0, 400.0]
    with pytest.raises(ValueError, match="rank"):
        whatif.run_scenarios(model, [whatif.parse_scenario("comm:median")])


def test_sweep_separates_critical_time_from_total_time() -> None:
    # Rank 0 computes for 100 ns; rank 1 communicates for 60 ns. Nothing
    # waits for the communication, so making it faster cannot help.
    nodes, edges = _tables(
        [
            (0, 1, "compute", "gemm", 100.0),
            (1, 1, "comm_coll", "nccl_allreduce", 60.0),
        ],
        [],
    )
    result = whatif.sweep(whatif.build_model(nodes, edges), by="class")
    ceiling = {
        row["group"]: row for row in result.ceiling.iter_rows(named=True)
    }
    assert result.baseline_makespan_ns == 100.0
    assert ceiling["compute"]["critical_share_pct"] == pytest.approx(100.0)
    assert ceiling["comm_coll"]["critical_share_pct"] == 0.0
    assert ceiling["comm_coll"]["time_share_pct"] == pytest.approx(37.5)
    assert ceiling["comm_coll"]["ceiling_speedup"] == 1.0
    assert ceiling["compute"]["ceiling_speedup"] == pytest.approx(100 / 60)
    assert result.ceiling["group"][0] == "compute"
    # With compute removed, the communication is the bottleneck.
    removed = result.table.filter(
        (pl.col("group") == "compute") & pl.col("factor").is_infinite()
    ).row(0, named=True)
    assert removed["bottleneck"] == "comm_coll"
    assert removed["makespan_ns"] == 60.0


def test_sweep_keeps_the_heaviest_groups_and_pools_the_rest() -> None:
    nodes, edges = _random_tables(2, 300, 3, False)
    result = whatif.sweep(
        whatif.build_model(nodes, edges), by="name", max_groups=2
    )
    assert set(result.ceiling["group"]) <= {"(other)"} | set(
        nodes["name"].to_list()
    )
    assert result.ceiling.height == 3


@pytest.mark.parametrize(
    ("text", "multiplier"),
    [("2x", 0.5), ("0.5x", 2.0), ("-30%", 0.7), ("+20%", 1.2), ("remove", 0.0)],
)
def test_factors(text: str, multiplier: float) -> None:
    assert whatif.parse_factor(text) == pytest.approx(multiplier)
    assert whatif.parse_factor("median") is None


@pytest.mark.parametrize(
    "text",
    [
        "comm",
        "comm:",
        ":2x",
        "comm:fast",
        "comm:0x",
        "gpu:2x",
        "name~(:2x",
        "rank=a:2x",
        "comm:-150%",
    ],
)
def test_malformed_rules_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        whatif.parse_scenario(text)


def test_sweep_factors() -> None:
    assert whatif.parse_factors("1.5, 2x,inf") == (1.5, 2.0, math.inf)
    with pytest.raises(ValueError):
        whatif.parse_factors("0")


def test_predict_scores_the_prediction_against_a_measured_run(
    tmp_path: pathlib.Path,
) -> None:
    base = goal.write_allreduce(
        tmp_path / "base.goal", compute_ns=1_000_000, steps=3
    )
    fast = goal.write_allreduce(
        tmp_path / "fast.goal", compute_ns=500_000, steps=3
    )
    report = cta.compare(
        base,
        fast,
        aspects=["whatif"],
        predict="compute:2x",
        cache_dir=tmp_path / "cache",
    )
    result = report.aspects["whatif"]
    residual = result.summary.filter(pl.col("metric") == "whatif_residual")
    assert residual.height == 1
    assert abs(residual["delta_pct"][0]) < 1.0
    assert {"sweep", "ceiling", "predicted"} <= set(result.details)

    # Gates read "measured against predicted": positive thresholds fail a run
    # slower than predicted, negative ones a run faster than predicted.
    for predict, gate, fails in (
        ("compute:1.1x", 5.0, False),  # measured is far faster: fine
        ("compute:1.1x", -5.0, True),
        ("compute:4x", 5.0, True),  # measured is slower than a 4x prediction
        ("compute:4x", -5.0, False),
    ):
        gated = cta.compare(
            base,
            fast,
            aspects=["whatif"],
            predict=predict,
            gates={"whatif_residual": gate},
            cache_dir=tmp_path / "cache",
        )
        assert bool(gated.violations) is fails, (predict, gate)


def test_cli_prints_scenarios_and_the_sweep_as_json(
    tmp_path: pathlib.Path,
) -> None:
    schedule = goal.write_allreduce(tmp_path / "run.goal", steps=2)
    result = cli_runner.invoke(
        cli_app.app,
        [
            "whatif",
            str(schedule),
            "--speed",
            "compute:2x",
            "--sweep",
            "--by",
            "class",
            "--format",
            "json",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert [row["scenario"] for row in document["scenarios"]] == [
        "baseline",
        "compute:2x",
    ]
    assert document["scenarios"][1]["speedup"] > 1.0
    assert {row["group"] for row in document["ceiling"]} >= {"compute"}


def test_cli_rejects_a_bad_rule(tmp_path: pathlib.Path) -> None:
    schedule = goal.write_allreduce(tmp_path / "run.goal")
    result = cli_runner.invoke(
        cli_app.app, ["whatif", str(schedule), "--speed", "comm"]
    )
    assert result.exit_code == 2
