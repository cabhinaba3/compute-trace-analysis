"""Simulated durations of compute ops.

``durations`` starts from what the trace recorded and applies, in order:

1. The model: ``recorded`` keeps the trace's durations; ``roofline`` times
   every op that has FLOPs or bytes as max(FLOPs / (peak * efficiency),
   bytes / (memory bandwidth * efficiency)); ``roofline_if_missing`` does so
   only for ops that recorded no duration. Chakra gives FLOPs (num_ops) and
   bytes (tensor_size); GOAL gives neither, so its calcs keep their length.
2. ``scale``, then ``rank_scale`` (stragglers), then ``name_scale`` (regex
   on the op name, e.g. "a 2x faster attention kernel").
3. ``jitter``: each op is multiplied by an independent log-normal factor of
   mean 1 and the given sigma, drawn from the config's seed.

Only calcs change; the recorded durations of collectives (used by
``collectives.cost = "recorded"``) are kept.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cta.sim import config as config_lib
from cta.sim import graph as graph_lib


def durations(
    graph: graph_lib.SimGraph, settings: config_lib.ComputeConfig, seed: int
) -> np.ndarray:
    """Returns every op's duration in ns as the engine should use it.

    Raises:
        ValueError: The roofline model is chosen without a peak FLOP rate or
            memory bandwidth, or a scale factor is negative.
    """
    nodes = graph.nodes
    recorded = nodes["recorded_ns"].to_numpy().astype(np.float64)
    is_calc = nodes["kind"].to_numpy() == graph_lib.KIND_CALC
    out = recorded.copy()

    if settings.model != "recorded":
        out = np.where(is_calc, _roofline(nodes, settings, recorded), out)

    factor = np.full(nodes.height, settings.scale, dtype=np.float64)
    ranks = nodes["rank"].to_numpy()
    for rank, value in settings.rank_scale.items():
        factor[ranks == rank] *= value
    names = nodes["name"]
    for pattern, value in settings.name_scale.items():
        factor[names.str.contains(pattern).fill_null(False).to_numpy()] *= value
    if np.any(factor < 0):
        raise ValueError("compute scale factors must not be negative")
    out = np.where(is_calc, out * factor, out)

    if settings.jitter > 0:
        rng = np.random.default_rng(seed)
        sigma = settings.jitter
        noise = rng.lognormal(-sigma * sigma / 2, sigma, size=nodes.height)
        out = np.where(is_calc, out * noise, out)
    return np.rint(np.maximum(out, 0.0)).astype(np.int64)


def _roofline(
    nodes: pl.DataFrame,
    settings: config_lib.ComputeConfig,
    recorded: np.ndarray,
) -> np.ndarray:
    """Returns roofline times where the model applies, else ``recorded``."""
    if settings.peak_flops <= 0 and settings.memory_bandwidth <= 0:
        raise ValueError(
            "the roofline compute model needs compute.peak_flops or "
            "compute.memory_bandwidth"
        )
    if not 0 < settings.efficiency <= 1:
        raise ValueError("compute.efficiency must be in (0, 1]")
    flops = nodes["num_ops"].to_numpy().astype(np.float64)
    size = nodes["size_bytes"].to_numpy().astype(np.float64)
    compute_ns = (
        flops / (settings.peak_flops * settings.efficiency) * 1e9
        if settings.peak_flops > 0
        else np.zeros_like(flops)
    )
    # memory_bandwidth is in bytes per ns.
    memory_ns = (
        size / (settings.memory_bandwidth * settings.efficiency)
        if settings.memory_bandwidth > 0
        else np.zeros_like(size)
    )
    modeled = np.maximum(compute_ns, memory_ns)
    applies = (flops > 0) | (size > 0)
    if settings.model == "roofline_if_missing":
        applies &= recorded <= 0
    return np.where(applies, modeled, recorded)
