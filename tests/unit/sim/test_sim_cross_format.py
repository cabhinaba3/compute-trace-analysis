"""GOAL and Chakra of one job, simulated and compared with each other.

The synthetic training job is written as GOAL (collectives spelled out as
ring all-reduces) and as Chakra (one COMM_COLL node per collective, with a
pg_name). Lowering the Chakra collectives with the ring algorithm must give
back the GOAL schedule's timing exactly, under every network model.
"""

from __future__ import annotations

import pathlib
from typing import Any

import polars as pl
import pytest

import cta
from cta.export import writers
from cta.sim import compare
from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import goal_export
from cta.sim import graph as graph_lib
from cta.sim import trace_export


def load(traces: dict[str, pathlib.Path], name: str) -> graph_lib.SimGraph:
    return graph_lib.load_graph(traces[name], cache_dir=traces["cache"])


@pytest.mark.parametrize("model", config_lib.NETWORK_MODELS)
def test_lowered_chakra_matches_goal_exactly(
    model: str, training_traces: dict[str, pathlib.Path]
) -> None:
    settings = config_lib.load_config(model=model)
    goal = engine.run_graph(load(training_traces, "goal"), settings)
    chakra = engine.run_graph(
        load(training_traces, "chakra"), settings, level="lowered"
    )
    assert goal.level == "p2p" and chakra.level == "lowered"
    assert goal.completed and chakra.completed
    assert chakra.makespan_ns == goal.makespan_ns
    assert (
        chakra.ranks["finish_ns"].to_list() == goal.ranks["finish_ns"].to_list()
    )
    assert (
        chakra.breakdown()["ns"].to_list() == goal.breakdown()["ns"].to_list()
    )


def test_completion_semantics_under_contention(
    training_traces: dict[str, pathlib.Path],
) -> None:
    # A thin fat tree makes flows contend, and the ring chunks are above the
    # eager limit. A collective that is done with its last recv (outputs, as
    # the GOAL schedule chains) still matches GOAL exactly; the default also
    # waits for the last rendezvous sends, as MPI and NCCL do.
    contended = config_lib.load_config(
        model="flow",
        overrides={
            "topology.kind": "fat_tree",
            "topology.hosts_per_leaf": "2",
            "topology.spines": "1",
        },
    )
    goal = engine.run_graph(load(training_traces, "goal"), contended)
    chakra = load(training_traces, "chakra")
    like_goal = engine.run_graph(
        chakra,
        contended.with_overrides({"collectives.completion": "outputs"}),
        level="lowered",
    )
    assert like_goal.makespan_ns == goal.makespan_ns
    assert like_goal.breakdown()["ns"][5] > 0  # contention on the path
    waiting = engine.run_graph(chakra, contended, level="lowered")
    assert waiting.completed


def test_goal_text_and_binary_simulate_alike(
    training_traces: dict[str, pathlib.Path],
) -> None:
    text = engine.run_graph(load(training_traces, "goal"))
    binary = engine.run_graph(load(training_traces, "goal_bin"))
    assert text.makespan_ns == binary.makespan_ns


def test_groups_come_from_pg_names(
    training_traces: dict[str, pathlib.Path],
) -> None:
    chakra = load(training_traces, "chakra")
    # Four tensor-parallel and four data-parallel groups of two ranks.
    assert len(chakra.group_names) == 8
    assert chakra.group("pg_0_1_2") == [0, 1]
    assert chakra.group("pg_0_4_2") == [0, 4]
    assert (
        chakra.issues.filter(pl.col("check") == "groups_inferred").height == 1
    )


def test_explicit_groups_override_inference(
    training_traces: dict[str, pathlib.Path],
) -> None:
    graph = graph_lib.load_graph(
        training_traces["chakra"],
        cache_dir=training_traces["cache"],
        groups={"pg_0_1_2": [1, 0]},
    )
    assert graph.group("pg_0_1_2") == [1, 0]
    with pytest.raises(ValueError, match="does not give them"):
        graph_lib.load_graph(
            training_traces["chakra"],
            cache_dir=training_traces["cache"],
            groups={"pg_0_1_2": [0]},
        )


def test_lowered_goal_export_round_trips(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    chakra = load(training_traces, "chakra")
    settings = config_lib.SimConfig()
    lowered, _ = engine.lower(chakra, settings)
    path = tmp_path / "lowered.goal"
    goal_export.write_goal(lowered, chakra.world_size, path)
    replayed = engine.run_graph(
        graph_lib.load_graph(path, cache_dir=tmp_path / "c"), settings
    )
    assert (
        replayed.makespan_ns
        == engine.run_graph(chakra, settings, level="lowered").makespan_ns
    )


def test_analytical_level_keeps_collectives_whole(
    training_traces: dict[str, pathlib.Path],
) -> None:
    result = engine.run_graph(
        load(training_traces, "chakra"), level="analytical"
    )
    assert result.completed
    assert (
        result.breakdown()
        .filter(pl.col("segment") == "collective")["ns"]
        .item()
        > 0
    )
    assert (
        result.collectives.height
        == load(training_traces, "chakra")
        .nodes.filter(pl.col("kind") == graph_lib.KIND_COLLECTIVE)
        .height
    )


def test_cross_comparison_tables(
    training_traces: dict[str, pathlib.Path],
) -> None:
    comparison = compare.compare(
        [training_traces["goal"], training_traces["chakra"]],
        labels=["goal", "chakra"],
        cache_dir=training_traces["cache"],
    )
    runs = comparison.overview["run"].to_list()
    assert runs == ["goal", "chakra [lowered]", "chakra [analytical]"]
    lowered = comparison.overview.filter(
        pl.col("run") == "chakra [lowered]"
    ).row(0, named=True)
    assert lowered["delta_ms"] == 0.0
    # The deltas of the critical-path segments add up to the makespan gap.
    for run in runs[1:]:
        gap = (
            comparison.overview.filter(pl.col("run") == run)["delta_ms"].item()
            * 1e6
        )
        deltas = comparison.attribution.filter(pl.col("run") == run)[
            "delta_ns"
        ].sum()
        assert deltas == pytest.approx(gap, abs=1)
    # Same messages between the same ranks: no traffic difference at all.
    assert compare.traffic_mismatch(comparison)["chakra [lowered]"] == 0.0
    phases = comparison.phases.filter(pl.col("run") == "chakra [lowered]")
    assert (
        phases.height
        == load(training_traces, "goal")
        .nodes.filter(pl.col("kind") == graph_lib.KIND_CALC)
        .height
    )
    assert (phases["span_delta_ns"] == 0).all()
    assert (phases["bytes_delta"] == 0).all()
    assert set(comparison.structure.columns) >= {
        "run",
        "send_ops",
        "wire_bytes",
        "wire_bytes_delta",
    }


def test_cross_comparison_finds_the_all_to_all_size_convention(
    training_traces: dict[str, pathlib.Path],
) -> None:
    # The GOAL generator sends the whole buffer to every peer of an
    # all-to-all; Chakra's comm_size convention (and the lowering's default)
    # sends B/P. Reading the sizes as per-peer makes the traffic agree.
    comparison = compare.compare(
        [training_traces["moe_goal"], training_traces["moe_chakra"]],
        labels=["goal", "chakra"],
        levels="lowered",
        cache_dir=training_traces["cache"],
    )
    assert compare.traffic_mismatch(comparison)["chakra [lowered]"] > 0
    moe_phases = comparison.phases.filter(
        pl.col("run_collectives").str.contains("ALL_TO_ALL")
    )
    assert moe_phases.height > 0
    assert (moe_phases["bytes_delta"] < 0).all()

    per_peer = compare.compare(
        [training_traces["moe_goal"], training_traces["moe_chakra"]],
        config_lib.SimConfig().with_overrides(
            {"collectives.all_to_all_size": "per_peer"}
        ),
        labels=["goal", "chakra"],
        levels="lowered",
        cache_dir=training_traces["cache"],
    )
    assert compare.traffic_mismatch(per_peer)["chakra [lowered]"] == 0.0
    assert (per_peer.phases["bytes_delta"] == 0).all()


def test_comparison_exports_and_plots(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    comparison = compare.compare(
        [training_traces["goal"], training_traces["chakra"]],
        cache_dir=training_traces["cache"],
        settings=config_lib.load_config(model="flow"),
    )
    report = comparison.to_report()
    written = writers.write_report(report, tmp_path, ["markdown", "json"])
    assert written
    for kind in ("sim_breakdown", "sim_attribution", "sim_ranks", "sim_phases"):
        report.plot(kind=kind, path=tmp_path / f"{kind}.png")
        assert (tmp_path / f"{kind}.png").stat().st_size > 0
    files = comparison.write(tmp_path / "tables")
    assert any(f.name == "phases.parquet" for f in files)


def test_simulation_aspect_in_cta_compare(
    training_traces: dict[str, pathlib.Path],
) -> None:
    report = cta.compare(
        training_traces["goal"],
        training_traces["goal_bin"],
        aspects=["simulation"],
        cache_dir=training_traces["cache"],
    )
    summary = report.to_polars("simulation")
    makespan = summary.filter(pl.col("metric") == "sim_makespan_ms").row(
        0, named=True
    )
    assert makespan["baseline"] == pytest.approx(makespan["candidate"])
    assert not report.to_polars("simulation", table="breakdown").is_empty()


def test_chrome_trace_export(
    training_traces: dict[str, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    import json

    result = engine.run_graph(load(training_traces, "chakra"))
    count = trace_export.write_chrome_trace(
        result, tmp_path / "t.json", ranks=[0, 1]
    )
    events: list[dict[str, Any]] = json.loads(
        (tmp_path / "t.json").read_text()
    )["traceEvents"]
    assert count == len(events)
    assert {e["ph"] for e in events} >= {"M", "X", "s", "f"}
    assert {
        e["pid"]
        for e in events
        if e["ph"] == "X" and e["cat"] != "critical_path"
    } == {0, 1}


def test_public_api(training_traces: dict[str, pathlib.Path]) -> None:
    settings = cta.load_sim_config(
        preset="ib-ndr400", overrides=["collectives.level=analytical"]
    )
    result = cta.simulate(
        training_traces["chakra"], settings, cache_dir=training_traces["cache"]
    )
    assert isinstance(result, cta.SimResult)
    assert result.level == "analytical"
    comparison = cta.compare_simulations(
        [training_traces["goal"], training_traces["goal_bin"]],
        cache_dir=training_traces["cache"],
    )
    assert isinstance(comparison, cta.SimComparison)
