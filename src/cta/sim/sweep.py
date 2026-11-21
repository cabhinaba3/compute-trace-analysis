"""Parameter sweeps and Monte Carlo runs over one graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
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
) -> pl.DataFrame:
    """Simulates every combination of the grid's values.

    One row per point: its values, METRICS, and the ``error`` of a point
    whose settings did not parse.
    """
    rows = []
    for values in itertools.product(*grid.values()):
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
            rows.append(row | {"error": str(error)})
            continue
        summary = result.summary()
        rows.append(
            row
            | {m: summary[m] for m in METRICS}
            | {"completed": result.completed, "error": None}
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
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Reruns the graph with seeds seed, seed + 1, ...

    Returns one row per trial, and the makespan's mean, std, min, p50, p95,
    p99 and max in ms.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    rows = []
    for seed in range(settings["seed"], settings["seed"] + trials):
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
                "noise_ns": summary["total_noise_ns"],
            }
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
