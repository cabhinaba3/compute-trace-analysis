"""Simulated durations of compute ops.

From the trace's durations, ``durations`` applies the compute model (a
roofline max(FLOPs / peak, bytes / bandwidth), scaled by the efficiency, for
every calc with FLOPs or bytes, or only for those without a recorded time),
then the global, per-rank and per-name factors, then log-normal jitter of
mean 1. Only calcs change; collectives keep their recorded time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cta.sim import graph as graph_lib


def durations(
    graph: graph_lib.SimGraph, settings: Mapping[str, Any]
) -> np.ndarray:
    """Every op's duration in ns, as the engine should use it."""
    nodes = graph.nodes
    recorded = nodes["recorded_ns"].to_numpy().astype(np.float64)
    is_calc = nodes["kind"].to_numpy() == graph_lib.KIND_CALC
    out = recorded.copy()
    if settings["compute.model"] != "recorded":
        peak, bandwidth = (
            settings["compute.peak_flops"],
            settings["compute.memory_bandwidth"],
        )
        efficiency = settings["compute.efficiency"]
        if peak <= 0 and bandwidth <= 0:
            raise ValueError(
                "the roofline needs compute.peak_flops or "
                "compute.memory_bandwidth"
            )
        if not 0 < efficiency <= 1:
            raise ValueError("compute.efficiency must be in (0, 1]")
        flops = nodes["num_ops"].to_numpy().astype(np.float64)
        size = nodes["size_bytes"].to_numpy().astype(np.float64)
        modeled = np.maximum(
            flops / (peak * efficiency) * 1e9 if peak > 0 else 0.0,
            size / (bandwidth * efficiency) if bandwidth > 0 else 0.0,
        )
        applies = is_calc & ((flops > 0) | (size > 0))
        if settings["compute.model"] == "roofline_if_missing":
            applies &= recorded <= 0
        out = np.where(applies, modeled, out)

    factor = np.full(nodes.height, float(settings["compute.scale"]))
    ranks = nodes["rank"].to_numpy()
    for rank, value in settings["compute.rank_scale"].items():
        factor[ranks == rank] *= value
    for pattern, value in settings["compute.name_scale"].items():
        factor[
            nodes["name"].str.contains(pattern).fill_null(False).to_numpy()
        ] *= value
    if np.any(factor < 0):
        raise ValueError("compute scale factors must not be negative")
    if (sigma := settings["compute.jitter"]) > 0:
        rng = np.random.default_rng(settings["seed"])
        factor *= rng.lognormal(-sigma * sigma / 2, sigma, size=nodes.height)
    out = np.where(is_calc, out * factor, out)
    return np.rint(np.maximum(out, 0.0)).astype(np.int64)
