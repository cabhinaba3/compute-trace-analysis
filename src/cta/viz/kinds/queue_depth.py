"""Kernels running at the same time on a device, over time."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers


def depth_steps(
    starts: np.ndarray, ends: np.ndarray, edges: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Counts the kernels running at once, from their start and end times.

    A kernel runs on ``[start, end)``, so one that starts as another ends does
    not overlap it. Kernels with no duration are ignored.

    Args:
        starts: Kernel start times.
        ends: Kernel end times, in the same order.
        edges: Increasing bin edges. With them, the result has one value per
            bin: the most kernels that run together at any moment inside it.
            Without them, one value per start or end event.

    Returns:
        The times and the depth from each time on: the event times, or the
        bin starts.
    """
    keep = ends > starts
    times = np.concatenate([starts[keep], ends[keep]])
    count = int(keep.sum())
    deltas = np.concatenate(
        [np.ones(count, dtype=int), -np.ones(count, dtype=int)]
    )
    # An end sorts before a start at the same time.
    order = np.lexsort((deltas, times))
    times, depth = times[order], np.cumsum(deltas[order])
    if edges is None:
        return times, depth
    bins = len(edges) - 1
    if not len(times):
        return edges[:-1], np.zeros(bins, dtype=int)
    # Depth carried in from before each bin, then the peak of its events.
    before = np.searchsorted(times, edges[:-1], side="left")
    peak = np.where(before > 0, depth[np.maximum(before - 1, 0)], 0)
    inside = np.clip(
        np.searchsorted(edges, times, side="right") - 1, 0, bins - 1
    )
    np.maximum.at(peak, inside, depth)
    return edges[:-1], peak


@spec.register_plot(
    "queue_depth",
    group="timeline",
    title="Kernels running at once, over time",
    aliases=("concurrency", "stream_concurrency", "active_kernels"),
    needs="kernels table",
    options={
        "by": spec.Opt(
            "str",
            "category",
            "Count all kernels, or one line per category or stream.",
            ("category", "stream", "total"),
        ),
        "devices": spec.Opt(
            "str",
            "max",
            "With several GPUs: draw the busiest one at each time, or one "
            "line per GPU.",
            ("max", "split"),
        ),
        "bins": spec.Opt(
            "int",
            200,
            "Time bins; each shows its peak depth (0 = every start and end).",
        ),
        "max_events": spec.Opt(
            "int", 20000, "Most kernels read per group (0 = all)."
        ),
    },
)
def queue_depth(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots how many kernels run at once on a GPU.

    The count is per GPU: kernels of different ranks or devices never add up.
    It counts kernels that are executing, not launches waiting to start.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one panel per group.

    Raises:
        spec.NotApplicableError: If the kernels table is empty or has no
            timestamps.
    """
    kernels = ctx.frame("kernels", options)
    if "t_ms" not in kernels.columns or "end_ms" not in kernels.columns:
        raise spec.NotApplicableError("the kernels table has no timestamps")

    cap = int(options["max_events"])
    n_bins = int(options["bins"])
    panels: list[ir.Panel] = []
    capped: list[str] = []
    drawn = 0
    for label in chart_helpers.unique_in_order(kernels, "label"):
        group = kernels.filter(pl.col("label") == label)
        if 0 < cap < group.height:
            group = group.sort("t_ms").head(cap)
            capped.append(str(label))
        drawn += group.height
        traces = _group_traces(group, options, n_bins, str(label))
        panels.append(
            ir.Panel(
                traces=traces,
                title=str(label),
                xlabel="time since the first step (ms)",
                ylabel="kernels running at once",
            )
        )
    notes = (
        f"Only the first {cap} kernels of {', '.join(capped)} were read "
        "(max_events)."
        if capped
        else ""
    )
    return chart_helpers.grid(
        "Kernels running at once",
        panels,
        max_cols=1,
        notes=notes,
        meta={"kernel_count": drawn},
    )


def _group_traces(
    group: pl.DataFrame, options: dict[str, Any], n_bins: int, label: str
) -> list[ir.Trace]:
    """Builds the depth lines of one group's kernels."""
    t_min = float(group["t_ms"].to_numpy().min())
    t_max = float(group["end_ms"].to_numpy().max())
    edges = (
        np.linspace(t_min, max(t_max, t_min + 1e-6), n_bins + 1)
        if n_bins > 0
        else None
    )
    split = {"category": "category", "stream": "stream"}.get(options["by"])
    parts: list[tuple[str, pl.DataFrame]] = [("kernels", group)]
    if split in group.columns:
        parts = [
            (f"stream {key[0]}" if split == "stream" else str(key[0]), rows)
            for key, rows in group.group_by(split, maintain_order=True)
        ]
    gpu_columns = [c for c in ("rank", "device") if c in group.columns]
    traces: list[ir.Trace] = []
    for name, rows in parts:
        gpus = (
            [gpu for _, gpu in rows.group_by(gpu_columns, maintain_order=True)]
            if gpu_columns
            else [rows]
        )
        per_gpu = [
            depth_steps(
                gpu["t_ms"].to_numpy().astype(float),
                gpu["end_ms"].to_numpy().astype(float),
                edges,
            )
            for gpu in gpus
        ]
        if options["devices"] == "split" and len(per_gpu) > 1:
            lines = [
                (f"{name} / GPU {number}", times, depth)
                for number, (times, depth) in enumerate(per_gpu)
            ]
        else:
            lines = [(name, *_busiest(per_gpu, edges))]
        for line_name, times, depth in lines:
            traces.append(
                ir.Trace(
                    "step",
                    x=times.tolist(),
                    y=depth.tolist(),
                    name=line_name,
                    style={"ci": len(traces)},
                    meta={"label": label},
                )
            )
    return traces


def _busiest(
    per_device: list[tuple[np.ndarray, np.ndarray]], edges: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Takes the deepest device at each time.

    Binned lines share their x, so the peaks combine directly. Event lines
    are resampled at every event time of any device.
    """
    if len(per_device) == 1:
        return per_device[0]
    if edges is not None:
        return per_device[0][0], np.max(
            [depth for _, depth in per_device], axis=0
        )
    times = np.unique(np.concatenate([t for t, _ in per_device]))
    levels = [
        np.where(
            (at := np.searchsorted(t, times, side="right")) > 0,
            d[np.maximum(at - 1, 0)],
            0,
        )
        for t, d in per_device
    ]
    return times, np.max(levels, axis=0)
