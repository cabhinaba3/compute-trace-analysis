"""Parameter sweeps and Monte Carlo studies over one graph.

The graph is built once and simulated once per point. ``sweep`` runs the
Cartesian product of the values given per setting (``--set`` keys); a point
whose settings are invalid gets its error instead of metrics. ``monte_carlo``
reruns the graph with seeds seed, seed + 1, ... so OS noise and compute
jitter vary, and summarises the makespan distribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
from typing import Any

import numpy as np
import polars as pl

from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib

# The metrics each point of a sweep reports.
SWEEP_METRICS = (
    "makespan_ms",
    "cp_compute_pct",
    "cp_comm_pct",
    "cp_contention_pct",
    "rank_finish_spread_ns",
    "mean_utilization",
)


def parse_grid(items: Sequence[str]) -> dict[str, list[str]]:
    """Reads ``key=v1,v2,...`` items into a grid.

    Raises:
        ValueError: An item without ``=`` or without values.
    """
    grid: dict[str, list[str]] = {}
    for key, values in config_lib.parse_assignments(items).items():
        parts = [v.strip() for v in values.split(",") if v.strip()]
        if not parts:
            raise ValueError(f"{key} has no values")
        grid[key] = parts
    return grid


def sweep(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig,
    grid: Mapping[str, Sequence[Any]],
    *,
    level: str | None = None,
) -> pl.DataFrame:
    """Simulates every combination of the grid's values.

    Returns:
        One row per point: the grid's keys (as given), the run's level,
        SWEEP_METRICS and ``error`` (null when the point ran).
    """
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        row: dict[str, Any] = {
            k: str(v) for k, v in zip(keys, values, strict=True)
        }
        try:
            point = settings.with_overrides(
                dict(zip(keys, values, strict=True))
            )
            result = engine.run_graph(graph, point, level=level, keep_ops=False)
        except ValueError as error:
            row["error"] = str(error)
            rows.append(row)
            continue
        summary = result.summary()
        row["level"] = result.level
        for metric in SWEEP_METRICS:
            row[metric] = summary.get(metric)
        row["completed"] = result.completed
        row["error"] = None
        rows.append(row)
    schema: dict[str, Any] = {k: pl.String for k in keys}
    schema.update({"level": pl.String})
    schema.update({m: pl.Float64 for m in SWEEP_METRICS})
    schema.update({"completed": pl.Boolean, "error": pl.String})
    return pl.DataFrame(rows, schema=schema)


def monte_carlo(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig,
    trials: int,
    *,
    level: str | None = None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Reruns the graph with ``trials`` consecutive seeds.

    Returns:
        One row per trial (``seed``, ``makespan_ms``, ``cp_comm_pct``,
        ``total_noise_ns``) and the makespan's mean, std, min, p50, p95, p99
        and max in ms.

    Raises:
        ValueError: ``trials`` is less than 1.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    rows = []
    for k in range(trials):
        seed = settings.seed + k
        result = engine.run_graph(
            graph,
            settings.with_overrides({"seed": seed}),
            level=level,
            keep_ops=False,
        )
        summary = result.summary()
        rows.append(
            {
                "seed": seed,
                "makespan_ms": summary["makespan_ms"],
                "cp_comm_pct": summary["cp_comm_pct"],
                "total_noise_ns": summary.get("total_noise_ns", 0.0),
            }
        )
    table = pl.DataFrame(rows)
    spans = table["makespan_ms"].to_numpy()
    stats = {
        "mean_ms": float(np.mean(spans)),
        "std_ms": float(np.std(spans, ddof=1)) if len(spans) > 1 else 0.0,
        "min_ms": float(np.min(spans)),
        "p50_ms": float(np.percentile(spans, 50)),
        "p95_ms": float(np.percentile(spans, 95)),
        "p99_ms": float(np.percentile(spans, 99)),
        "max_ms": float(np.max(spans)),
    }
    return table, stats
