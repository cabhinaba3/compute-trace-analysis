"""One job as GOAL and as Chakra, simulated and compared.

The synthetic training job's GOAL schedule spells its collectives out as
ring all-reduces; its Chakra trace has one COMM_COLL node per collective and
pg_name. Lowering the Chakra collectives with rings must reproduce the GOAL
schedule's timing exactly.
"""

from __future__ import annotations

import json
import pathlib

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

Traces = dict[str, pathlib.Path]


def load(traces: Traces, name: str) -> graph_lib.SimGraph:
    return graph_lib.load_graph(traces[name], cache_dir=traces["cache"])


def contended(**extra: str) -> config_lib.SimConfig:
    return config_lib.load_config(
        model="flow",
        overrides={
            "topology.kind": "fat_tree",
            "topology.hosts_per_leaf": "2",
            "topology.spines": "1",
            **extra,
        },
    )


@pytest.mark.parametrize("model", config_lib.NETWORK_MODELS)
def test_lowered_chakra_matches_goal(
    model: str, training_traces: Traces
) -> None:
    settings = config_lib.load_config(model=model)
    goal = engine.run_graph(load(training_traces, "goal"), settings)
    chakra = engine.run_graph(
        load(training_traces, "chakra"), settings, level="lowered"
    )
    assert goal.completed and chakra.completed
    assert (
        chakra.ranks["finish_ns"].to_list() == goal.ranks["finish_ns"].to_list()
    )
    assert chakra.breakdown().equals(goal.breakdown())


def test_completion_under_contention(training_traces: Traces) -> None:
    # Ending collectives with the result, as the GOAL schedule chains them,
    # keeps the match under contention; the default waits for the last sends.
    goal = engine.run_graph(load(training_traces, "goal"), contended())
    like_goal = engine.run_graph(
        load(training_traces, "chakra"),
        contended(**{"collectives.completion": "outputs"}),
        level="lowered",
    )
    assert like_goal.makespan_ns == goal.makespan_ns
    assert like_goal.summary()["cp_contention_ns"] > 0
    assert engine.run_graph(
        load(training_traces, "chakra"), contended(), level="lowered"
    ).completed


def test_goal_text_and_binary_agree(training_traces: Traces) -> None:
    text, binary = (
        engine.run_graph(load(training_traces, n)) for n in ("goal", "goal_bin")
    )
    assert text.makespan_ns == binary.makespan_ns


def test_groups(training_traces: Traces) -> None:
    chakra = load(training_traces, "chakra")
    assert (
        len(chakra.group_names) == 8
    )  # four tensor- and four data-parallel pairs
    assert (chakra.group("pg_0_1_2"), chakra.group("pg_0_4_2")) == (
        [0, 1],
        [0, 4],
    )
    swapped = graph_lib.load_graph(
        training_traces["chakra"],
        cache_dir=training_traces["cache"],
        groups={"pg_0_1_2": [1, 0]},
    )
    assert swapped.group("pg_0_1_2") == [1, 0]
    with pytest.raises(ValueError, match="does not give them"):
        graph_lib.load_graph(
            training_traces["chakra"],
            cache_dir=training_traces["cache"],
            groups={"pg_0_1_2": [0]},
        )


def test_lowered_goal_export_round_trips(
    training_traces: Traces, tmp_path: pathlib.Path
) -> None:
    chakra = load(training_traces, "chakra")
    settings = config_lib.SimConfig()
    goal_export.write_goal(
        engine.lower(chakra, settings)[0],
        chakra.world_size,
        tmp_path / "lowered.goal",
    )
    replayed = engine.run_graph(
        graph_lib.load_graph(tmp_path / "lowered.goal", cache_dir=tmp_path)
    )
    assert (
        replayed.makespan_ns == engine.run_graph(chakra, settings).makespan_ns
    )


def test_analytical_level(training_traces: Traces) -> None:
    result = engine.run_graph(
        load(training_traces, "chakra"), level="analytical"
    )
    assert result.completed and result.summary()["cp_collective_ns"] > 0
    assert (
        result.collectives.height
        == (
            load(training_traces, "chakra").nodes["kind"]
            == graph_lib.KIND_COLLECTIVE
        ).sum()
    )


def test_cross_comparison(
    training_traces: Traces, tmp_path: pathlib.Path
) -> None:
    comparison = compare.compare(
        [training_traces["goal"], training_traces["chakra"]],
        labels=["goal", "chakra"],
        cache_dir=training_traces["cache"],
    )
    assert comparison.overview["run"].to_list() == [
        "goal",
        "chakra [lowered]",
        "chakra [analytical]",
    ]
    assert comparison.overview["delta_ms"][1] == 0
    for run, delta_ms in comparison.overview.select("run", "delta_ms").rows()[
        1:
    ]:
        deltas = comparison.attribution.filter(pl.col("run") == run)[
            "delta_ns"
        ].sum()
        assert deltas == pytest.approx(delta_ms * 1e6, abs=1)
    assert comparison.traffic_mismatch() == {"chakra [lowered]": 0.0}
    phases = comparison.phases.filter(pl.col("run") == "chakra [lowered]")
    assert (
        phases.height
        == (
            load(training_traces, "goal").nodes["kind"] == graph_lib.KIND_CALC
        ).sum()
    )
    assert (phases["span_delta_ns"] == 0).all() and (
        phases["bytes_delta"] == 0
    ).all()

    report = comparison.to_report()
    assert writers.write_report(report, tmp_path, ["markdown", "json"])
    for kind in ("sim_breakdown", "sim_attribution", "sim_ranks", "sim_phases"):
        report.plot(kind=kind, path=tmp_path / f"{kind}.png")
    assert any(
        f.name == "phases.parquet"
        for f in comparison.write(tmp_path / "tables")
    )


def test_all_to_all_size_conventions(training_traces: Traces) -> None:
    # The GOAL generator sends the whole buffer to every all-to-all peer;
    # Chakra's comm_size (the default) is the whole buffer, B/P per peer.
    def mismatch(**overrides: str) -> compare.SimComparison:
        return compare.compare(
            [training_traces["moe_goal"], training_traces["moe_chakra"]],
            config_lib.SimConfig().with_overrides(overrides),
            labels=["goal", "chakra"],
            levels="lowered",
            cache_dir=training_traces["cache"],
        )

    total = mismatch()
    assert total.traffic_mismatch()["chakra [lowered]"] > 0
    moe = total.phases.filter(
        pl.col("run_collectives").str.contains("ALL_TO_ALL")
    )
    assert moe.height and (moe["bytes_delta"] < 0).all()
    per_peer = mismatch(**{"collectives.all_to_all_size": "per_peer"})
    assert per_peer.traffic_mismatch()["chakra [lowered]"] == 0
    assert (per_peer.phases["bytes_delta"] == 0).all()


def test_simulation_aspect(training_traces: Traces) -> None:
    report = cta.compare(
        training_traces["goal"],
        training_traces["goal_bin"],
        aspects=["simulation"],
        cache_dir=training_traces["cache"],
    )
    makespan = (
        report.to_polars("simulation")
        .filter(pl.col("metric") == "sim_makespan_ms")
        .row(0, named=True)
    )
    assert makespan["baseline"] == pytest.approx(makespan["candidate"])
    assert not report.to_polars("simulation", table="breakdown").is_empty()


def test_chrome_trace(training_traces: Traces, tmp_path: pathlib.Path) -> None:
    result = engine.run_graph(load(training_traces, "chakra"))
    count = trace_export.write_chrome_trace(
        result, tmp_path / "t.json", ranks=[0, 1]
    )
    events = json.loads((tmp_path / "t.json").read_text())["traceEvents"]
    assert count == len(events)
    assert {e["ph"] for e in events} >= {"M", "X", "s", "f"}
    assert {
        e["pid"]
        for e in events
        if e["ph"] == "X" and e["cat"] != "critical_path"
    } == {0, 1}


def test_public_api(training_traces: Traces) -> None:
    settings = cta.load_sim_config(
        preset="ib-ndr400", overrides=["collectives.level=analytical"]
    )
    result = cta.simulate(
        training_traces["chakra"], settings, cache_dir=training_traces["cache"]
    )
    assert isinstance(result, cta.SimResult) and result.level == "analytical"
    comparison = cta.compare_simulations(
        [training_traces["goal"], training_traces["goal_bin"]],
        cache_dir=training_traces["cache"],
    )
    assert isinstance(comparison, cta.SimComparison)
