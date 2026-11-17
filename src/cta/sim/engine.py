"""Runs simulations.

A graph with collectives runs ``lowered`` (each collective becomes its
algorithm's sends, recvs and reductions, which contend with everything else;
like for like with GOAL) or ``analytical`` (each collective stays whole, starts
when its last member is ready and lasts its contention-free time). A graph
without collectives runs at level ``p2p``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from cta import _native as native
from cta.sim import compute
from cta.sim import config as config_lib
from cta.sim import graph as graph_lib
from cta.sim import result as result_lib


def resolve_level(
    graph: graph_lib.SimGraph, level: str | None, settings: config_lib.SimConfig
) -> str:
    """The level a graph runs at."""
    if not graph.has_collectives:
        return "p2p"
    return config_lib.SETTINGS["collectives.level"].parse(
        level or settings["collectives.level"]
    )


def prepared_columns(
    graph: graph_lib.SimGraph, settings: config_lib.SimConfig
) -> dict[str, np.ndarray]:
    """The graph's arrays with the simulated compute durations."""
    return {**graph.columns, "duration": compute.durations(graph, settings)}


def lower(
    graph: graph_lib.SimGraph, settings: config_lib.SimConfig
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """The graph with its collectives lowered, and lowering counters.

    The arrays add ``origin``, ``role`` and ``action`` columns.
    """
    lowered, stats = native.lower_collectives(
        prepared_columns(graph, settings),
        graph.group_offsets,
        graph.group_members,
        settings.to_native(graph.world_size),
    )
    return lowered, dict(stats)


def run_graph(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig | None = None,
    *,
    level: str | None = None,
    keep_ops: bool = True,
) -> result_lib.SimResult:
    """Simulates a graph; ``keep_ops=False`` skips per-op and message tables."""
    settings = settings or config_lib.SimConfig()
    level = resolve_level(graph, level, settings)
    lowering: dict[str, int] = {}
    if level == "lowered":
        run, lowering = lower(graph, settings)
    else:
        run = prepared_columns(graph, settings)
    raw: dict[str, Any] = native.simulate(
        run,
        graph.group_offsets,
        graph.group_members,
        settings.to_native(graph.world_size),
    )
    return result_lib.from_native(
        graph, settings, level, run, raw, lowering, keep_ops=keep_ops
    )


def simulate(
    source: str | pathlib.Path | graph_lib.SimGraph,
    settings: config_lib.SimConfig | None = None,
    *,
    level: str | None = None,
    trace_format: str | None = None,
    label: str | None = None,
    cache_dir: str | pathlib.Path = ".cta_cache",
    keep_ops: bool = True,
) -> result_lib.SimResult:
    """Simulates a GOAL or Chakra trace (or a SimGraph)."""
    settings = settings or config_lib.SimConfig()
    graph = (
        source
        if isinstance(source, graph_lib.SimGraph)
        else graph_lib.load_graph(
            source,
            trace_format=trace_format,
            label=label,
            groups=settings["collectives.groups"],
            world_size=settings["world_size"],
            cache_dir=cache_dir,
        )
    )
    return run_graph(graph, settings, level=level, keep_ops=keep_ops)
