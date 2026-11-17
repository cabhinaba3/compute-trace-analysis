"""Tests of compute models, what-if scaling, sweeps and Monte Carlo runs."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest

from cta.sim import compute
from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib
from cta.sim import sweep


def tiny_graph() -> graph_lib.SimGraph:
    nodes = pl.DataFrame(
        {
            "rank": [0, 0, 1],
            "node_id": [0, 1, 0],
            "name": ["attn", "mlp", "attn"],
            "op_class": ["compute", "compute", "compute"],
            "comm_type": ["", "", ""],
            "duration_ns": [1000, 2000, 1000],
            "size_bytes": [4_000, 0, 0],
            "src_rank": [-1, -1, -1],
            "dst_rank": [-1, -1, -1],
            "message_tag": [0, 0, 0],
            "num_ops": [2_000_000, 0, 0],
        }
    ).cast(
        {
            "rank": pl.UInt32,
            "node_id": pl.UInt64,
            "duration_ns": pl.UInt64,
            "size_bytes": pl.UInt64,
            "num_ops": pl.UInt64,
            "message_tag": pl.UInt32,
            "src_rank": pl.Int32,
            "dst_rank": pl.Int32,
        }
    )
    edges = pl.DataFrame(
        {"rank": [0], "src_id": [0], "dst_id": [1], "dep_kind": ["data"]}
    ).cast({"rank": pl.UInt32, "src_id": pl.UInt64, "dst_id": pl.UInt64})
    return graph_lib.from_tables(
        nodes, edges, trace_format="chakra", label="tiny"
    )


def test_recorded_durations_with_scaling() -> None:
    settings = config_lib.ComputeConfig(
        scale=2.0, rank_scale={1: 1.5}, name_scale={"^mlp": 0.25}
    )
    assert compute.durations(tiny_graph(), settings, seed=1).tolist() == [
        2000,
        1000,
        3000,
    ]


def test_roofline() -> None:
    settings = config_lib.ComputeConfig(
        model="roofline", peak_flops=1e12, memory_bandwidth=1.0
    )
    # Op 0: 2 MFLOP at 1 TFLOP/s = 2000 ns; 4000 bytes at 1 B/ns = 4000 ns.
    # Ops without FLOPs or bytes keep their recorded time.
    assert compute.durations(tiny_graph(), settings, seed=1).tolist() == [
        4000,
        2000,
        1000,
    ]
    missing = config_lib.ComputeConfig(
        model="roofline_if_missing", peak_flops=1e12
    )
    assert compute.durations(tiny_graph(), missing, seed=1).tolist() == [
        1000,
        2000,
        1000,
    ]
    with pytest.raises(ValueError, match="needs compute.peak_flops"):
        compute.durations(
            tiny_graph(), config_lib.ComputeConfig(model="roofline"), seed=1
        )


def test_jitter_has_mean_one_and_follows_the_seed() -> None:
    graph = tiny_graph()
    settings = config_lib.ComputeConfig(jitter=0.2)
    a = compute.durations(graph, settings, seed=5)
    assert np.array_equal(a, compute.durations(graph, settings, seed=5))
    assert not np.array_equal(a, compute.durations(graph, settings, seed=6))


def test_straggler_moves_the_makespan() -> None:
    graph = tiny_graph()
    base = engine.run_graph(graph).makespan_ns
    slow = engine.run_graph(
        graph,
        config_lib.SimConfig().with_overrides({"compute.rank_scale.1": 4}),
    ).makespan_ns
    assert (base, slow) == (3000, 4000)


def test_sweep_and_monte_carlo(
    training_traces: dict[str, pathlib.Path],
) -> None:
    graph = graph_lib.load_graph(
        training_traces["goal"], cache_dir=training_traces["cache"]
    )
    grid = sweep.parse_grid(["loggops.G=6,3", "loggops.L=1us,bogus"])
    table = sweep.sweep(graph, config_lib.SimConfig(), grid)
    assert table.height == 4
    ok = table.filter(pl.col("error").is_null()).sort("loggops.G")
    assert ok.height == 2
    # Halving G (doubling bandwidth) speeds the run up.
    assert ok["makespan_ms"][0] < ok["makespan_ms"][1]
    assert table.filter(pl.col("error").is_not_null()).height == 2

    noisy = config_lib.SimConfig().with_overrides(
        {"noise.frequency": "10kHz", "noise.duration": "50us"}
    )
    trials, stats = sweep.monte_carlo(graph, noisy, 3)
    assert trials["seed"].to_list() == [1, 2, 3]
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["max_ms"]
    quiet = engine.run_graph(graph).makespan_ns / 1e6
    assert stats["min_ms"] > quiet
