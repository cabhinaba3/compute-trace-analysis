"""Parameter sweeps and Monte Carlo runs over one graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import itertools
import os
from typing import Any

import numpy as np
import polars as pl

from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib

METRICS = (
    "makespan_ms",
    "cp_compute_pct",
    "cp_comm_pct",
    "cp_contention_pct",
    "rank_finish_spread_ns",
    "mean_utilization",
)


def usable_cpus() -> int:
    """Returns the CPUs this process may run on (at least 1)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _map_ordered[T](
    work: Callable[[Any], T], items: Sequence[Any], workers: int | None
) -> list[T]:
    """Runs ``work`` over ``items`` on threads and returns results in order.

    One simulation is a sequential event loop in native code that releases
    the GIL, so independent runs overlap on a thread pool. An exception in
    any run is raised here, as in a serial loop.

    Args:
        work: Called once per item.
        items: The inputs.
        workers: Threads; None uses the usable CPUs, never more than items.

    Raises:
        ValueError: If ``workers`` is below 1.
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    count = min(workers or usable_cpus(), len(items))
    if count <= 1:
        return [work(item) for item in items]
    with futures.ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(work, items))


def parse_grid(items: Sequence[str]) -> dict[str, list[str]]:
    """Reads ``key=v1,v2,...`` items."""
    grid = {
        key: [v.strip() for v in values.split(",") if v.strip()]
        for key, values in config_lib.parse_assignments(items).items()
    }
    if empty := [k for k, v in grid.items() if not v]:
        raise ValueError(f"no values for {', '.join(empty)}")
    return grid


def sweep(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig,
    grid: Mapping[str, Sequence[Any]],
    *,
    level: str | None = None,
    workers: int | None = None,
) -> pl.DataFrame:
    """Simulates every combination of the grid's values.

    One row per point, in grid order: its values, METRICS, and the ``error``
    of a point whose settings did not parse. Points run on ``workers``
    threads (default: the usable CPUs) and give what a serial run gives.
    """

    def run_point(values: tuple[Any, ...]) -> dict[str, Any]:
        point = dict(zip(grid, values, strict=True))
        row: dict[str, Any] = {k: str(v) for k, v in point.items()}
        try:
            result = engine.run_graph(
                graph,
                settings.with_overrides(point),
                level=level,
                keep_ops=False,
            )
        except ValueError as error:
            return row | {"error": str(error)}
        summary = result.summary()
        return (
            row
            | {m: summary[m] for m in METRICS}
            | {"completed": result.completed, "error": None}
        )

    rows = _map_ordered(
        run_point, list(itertools.product(*grid.values())), workers
    )
    schema = (
        dict.fromkeys(grid, pl.String)
        | dict.fromkeys(METRICS, pl.Float64)
        | {"completed": pl.Boolean, "error": pl.String}
    )
    return pl.DataFrame(rows, schema=schema)


def monte_carlo(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig,
    trials: int,
    *,
    level: str | None = None,
    workers: int | None = None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Reruns the graph with seeds seed, seed + 1, ...

    Trials run on ``workers`` threads (default: the usable CPUs) and the rows
    come back in seed order, as a serial run gives them.

    Returns one row per trial, and the makespan's mean, std, min, p50, p95,
    p99 and max in ms.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")

    def run_trial(seed: int) -> dict[str, Any]:
        result = engine.run_graph(
            graph,
            settings.with_overrides({"seed": seed}),
            level=level,
            keep_ops=False,
        )
        summary = result.summary()
        return {
            "seed": seed,
            "makespan_ms": summary["makespan_ms"],
            "noise_ns": summary["total_noise_ns"],
        }

    rows = _map_ordered(
        run_trial,
        list(range(settings["seed"], settings["seed"] + trials)),
        workers,
    )
    spans = np.array([row["makespan_ms"] for row in rows])
    stats = {
        "mean_ms": float(spans.mean()),
        "std_ms": float(spans.std(ddof=1)) if trials > 1 else 0.0,
        "min_ms": float(spans.min()),
        **{f"p{q}_ms": float(np.percentile(spans, q)) for q in (50, 95, 99)},
        "max_ms": float(spans.max()),
    }
    return pl.DataFrame(rows), stats
