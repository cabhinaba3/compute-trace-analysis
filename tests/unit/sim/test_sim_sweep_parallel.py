"""Sweeps and Monte Carlo runs use several CPUs and give the serial answer.

One simulation is a sequential event loop, so it never runs on a GPU; the
points of a sweep and the seeds of a Monte Carlo run are independent, so they
run on a thread pool (the native simulator releases the GIL). The rows must
come back in the order of the grid and the seeds, whatever the thread count.
"""

from __future__ import annotations

import pathlib

import polars as pl
from polars import testing as pl_testing
import pytest

from cta.sim import config as config_lib
from cta.sim import graph as graph_lib
from cta.sim import sweep


@pytest.fixture(scope="module")
def graph(training_traces: dict[str, pathlib.Path]) -> graph_lib.SimGraph:
    return graph_lib.load_graph(
        training_traces["goal"], cache_dir=training_traces["cache"]
    )


NOISY = config_lib.SimConfig().with_overrides(
    {"noise.frequency": "10kHz", "noise.duration": "50us"}
)


def test_a_parallel_sweep_equals_the_serial_one(
    graph: graph_lib.SimGraph,
) -> None:
    grid = sweep.parse_grid(["loggops.G=6,3,1.5", "loggops.L=1us,2us,bad"])

    serial = sweep.sweep(graph, config_lib.SimConfig(), grid, workers=1)
    parallel = sweep.sweep(graph, config_lib.SimConfig(), grid, workers=4)

    pl_testing.assert_frame_equal(serial, parallel)
    assert serial["loggops.G"].to_list()[:3] == ["6", "6", "6"]  # grid order
    assert serial.filter(pl.col("error").is_not_null()).height == 3


def test_parallel_monte_carlo_equals_the_serial_one(
    graph: graph_lib.SimGraph,
) -> None:
    serial_rows, serial_stats = sweep.monte_carlo(graph, NOISY, 6, workers=1)
    rows, stats = sweep.monte_carlo(graph, NOISY, 6, workers=3)

    pl_testing.assert_frame_equal(serial_rows, rows)
    assert serial_stats == stats
    assert rows["seed"].to_list() == [1, 2, 3, 4, 5, 6]


def test_the_default_uses_the_cpus_and_still_gives_the_serial_answer(
    graph: graph_lib.SimGraph,
) -> None:
    serial_rows, _ = sweep.monte_carlo(graph, NOISY, 4, workers=1)

    default_rows, _ = sweep.monte_carlo(graph, NOISY, 4)

    pl_testing.assert_frame_equal(serial_rows, default_rows)


@pytest.mark.parametrize("workers", [0, -1])
def test_workers_must_be_positive(
    graph: graph_lib.SimGraph, workers: int
) -> None:
    with pytest.raises(ValueError, match="workers"):
        sweep.monte_carlo(graph, NOISY, 2, workers=workers)


def test_an_error_in_a_trial_is_raised_not_lost(
    graph: graph_lib.SimGraph,
) -> None:
    bad = config_lib.SimConfig()
    with pytest.raises(ValueError, match="trials"):
        sweep.monte_carlo(graph, bad, 0, workers=2)
