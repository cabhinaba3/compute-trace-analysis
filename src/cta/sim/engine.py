"""Runs simulations: ``simulate`` for a trace, ``run_graph`` for a SimGraph.

A graph with collectives runs at one of two levels:

* ``lowered``: each collective becomes its algorithm's sends, recvs and
  reductions (``cta.sim.collectives``), and those run on the network model
  with everything else, contention included. This is the level at which a
  Chakra trace and a GOAL schedule are like for like.
* ``analytical``: each collective stays one op per member; it starts when
  the last member is ready and lasts the algorithm's contention-free time
  under the network model (or the trace's recorded time).

A graph without collectives (any GOAL schedule) runs at level ``p2p``.
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
    """Returns the level a graph runs at: p2p without collectives."""
    if not graph.has_collectives:
        return "p2p"
    chosen = level or settings.collectives.level
    if chosen not in config_lib.COLLECTIVE_LEVELS:
        raise ValueError(
            f"unknown level {chosen!r}; use one of "
            f"{', '.join(config_lib.COLLECTIVE_LEVELS)}"
        )
    return chosen


def prepared_columns(
    graph: graph_lib.SimGraph, settings: config_lib.SimConfig
) -> dict[str, np.ndarray]:
    """Returns the graph's arrays with the simulated compute durations."""
    columns = dict(graph.columns)
    columns["duration"] = compute.durations(
        graph, settings.compute, settings.seed
    )
    return columns


def lower(
    graph: graph_lib.SimGraph, settings: config_lib.SimConfig
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Returns the graph with its collectives lowered, and lowering counters.

    The arrays carry ``origin`` (the op of ``graph`` each op came from),
    ``role`` (result.ROLES) and ``action`` (index in the schedule).
    """
    lowered, stats = native.lower_collectives(
        prepared_columns(graph, settings),
        graph.group_offsets,
        graph.group_members,
        settings.to_native(graph.world_size),
    )
    return lowered, {k: int(v) for k, v in stats.items()}


def run_graph(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig | None = None,
    *,
    level: str | None = None,
    keep_ops: bool = True,
) -> result_lib.SimResult:
    """Simulates a graph.

    Args:
        graph: The graph.
        settings: The config; None uses the defaults.
        level: ``lowered`` or ``analytical`` for graphs with collectives;
            None uses ``settings.collectives.level``.
        keep_ops: Whether the result keeps per-op and per-message tables
            (sweeps over large graphs turn this off).

    Returns:
        The result.

    Raises:
        ValueError: Invalid settings, or a graph the engine rejects.
    """
    settings = settings or config_lib.SimConfig()
    chosen = resolve_level(graph, level, settings)
    params = settings.to_native(graph.world_size)
    lowering: dict[str, int] = {}
    if chosen == "lowered":
        run, lowering = lower(graph, settings)
    else:
        run = prepared_columns(graph, settings)
    raw: dict[str, Any] = native.simulate(
        run, graph.group_offsets, graph.group_members, params
    )
    return result_lib.from_native(
        graph, settings, chosen, run, raw, lowering, keep_ops=keep_ops
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
    """Simulates a GOAL or Chakra trace.

    Args:
        source: A trace file, directory, glob or bundle, or a SimGraph.
        settings: The config (``config.load_config`` builds one); None uses
            the defaults (LogGOPSim's parameters, lowered collectives).
        level: ``lowered`` or ``analytical``; None uses the config's.
        trace_format: Which format to read if the source holds several.
        label: Name of the run; defaults to the bundle's label.
        cache_dir: Directory of the Parquet ingest cache.
        keep_ops: Whether to keep per-op and per-message tables.

    Returns:
        The result.
    """
    settings = settings or config_lib.SimConfig()
    graph = (
        source
        if isinstance(source, graph_lib.SimGraph)
        else graph_lib.load_graph(
            source,
            trace_format=trace_format,
            label=label,
            groups=settings.collectives.groups,
            world_size=settings.world_size,
            cache_dir=cache_dir,
        )
    )
    return run_graph(graph, settings, level=level, keep_ops=keep_ops)
