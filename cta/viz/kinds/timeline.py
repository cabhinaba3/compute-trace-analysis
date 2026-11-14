"""Timeline plot kinds.

Covers the GPU Gantt chart, the compute vs communication busy fraction over
time, and per-stream utilisation.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers


def _lane_expr(kernels: pl.DataFrame) -> pl.Expr:
    """Returns the expression naming each kernel's rank/device/stream lane.

    When the frame holds several trials, lanes get a trial prefix so trials
    never share a lane (their clocks are unrelated).

    Args:
        kernels: The kernels the expression will be applied to.

    Returns:
        An expression aliased to ``lane``.
    """
    base = pl.format(
        "r{}/gpu{}/s{}", pl.col("rank"), pl.col("device"), pl.col("stream")
    )
    if kernels["trial"].n_unique() > 1:
        return (pl.format("t{}/", pl.col("trial")) + base).alias("lane")
    return base.alias("lane")


@spec.register_plot(
    "timeline",
    group="timeline",
    title="GPU timeline (Gantt)",
    needs="kernels table (nsys, kineto or jax)",
    options={
        "by": spec.Opt(
            "str",
            "category",
            "What colours the bars.",
            ("category", "op_family", "name", "scope", "step"),
        ),
        "max_events": spec.Opt(
            "int",
            20000,
            "Cap on drawn kernels; the earliest steps are kept and the chart"
            " says so.",
        ),
        "max_lanes": spec.Opt(
            "int", 40, "Cap on rank/device/stream rows (the busiest are kept)."
        ),
    },
)
def timeline(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots kernels as bars on rank/device/stream lanes.

    Times are relative to each rank's first step. Without a `steps` option
    and with five or more steps, only steps 1-3 are drawn.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one panel per group; its notes say when the
        steps or kernels were capped.
    """
    kernels = ctx.frame("kernels", options)
    notes = []
    if not options.get("steps"):
        # Step 0 is usually a cold start that dwarfs the rest: show three
        # steady steps instead.
        last_step = cast(int | None, kernels["idx"].drop_nulls().max())
        n_steps = int(last_step or 0) + 1
        if n_steps >= 5:
            kernels = ctx.frame("kernels", {**options, "steps": (1, 4)})
            notes.append(
                f"showing steps 1-3 of {n_steps} (pass steps=a:b for another"
                " window)"
            )
    color_column = options["by"]
    color_values = sorted(
        str(value) for value in kernels[color_column].unique().to_list()
    )
    color_index_by_value = {value: i for i, value in enumerate(color_values)}
    lane = _lane_expr(kernels)
    panels = []
    labels = chart_helpers.unique_in_order(kernels, "label")
    for panel_index, label in enumerate(labels):
        group_kernels = kernels.filter(pl.col("label") == label).with_columns(
            lane
        )
        if group_kernels.height > options["max_events"]:
            group_kernels = group_kernels.sort("t_ms").head(
                options["max_events"]
            )
            notes.append(
                f"{label}: showing the first {group_kernels.height} kernels"
                f" (max_events={options['max_events']})"
            )
        busiest_lanes = (
            group_kernels.group_by("lane")
            .agg(pl.col("duration_ms").sum().alias("busy_ms"))
            .sort("busy_ms", descending=True)
            .head(int(options["max_lanes"]))["lane"]
            .to_list()
        )
        lanes = sorted(busiest_lanes)
        group_kernels = group_kernels.filter(pl.col("lane").is_in(lanes))
        traces = []
        for value in color_values:
            value_kernels = group_kernels.filter(
                pl.col(color_column).cast(pl.String) == value
            )
            if value_kernels.is_empty():
                continue
            hover_texts = [
                f"{name} ({duration_ms:.3f} ms)"
                for name, duration_ms in zip(
                    value_kernels["name"].to_list(),
                    value_kernels["duration_ms"].to_list(),
                    strict=True,
                )
            ]
            traces.append(
                ir.Trace(
                    "segment",
                    x=value_kernels["t_ms"].to_list(),
                    x1=value_kernels["end_ms"].to_list(),
                    y=value_kernels["lane"].to_list(),
                    name=value,
                    text=hover_texts,
                    style={
                        "ci": color_index_by_value[value],
                        "lanes": lanes,
                        "nolegend": panel_index > 0,
                        "width": 5,
                    },
                    meta={"label": label},
                )
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=label,
                xlabel="time since rank's first step (ms)",
                ylabel="",
                xkind="",
                legend=panel_index == 0,
            )
        )
    return chart_helpers.grid(
        "GPU timeline", panels, max_cols=1, notes="; ".join(notes)
    )


@spec.register_plot(
    "overlap_timeline",
    group="timeline",
    title="Compute vs communication busy fraction over time",
    needs="kernels table",
    options={"bins": spec.Opt("int", 120, "Time bins.")},
)
def overlap_timeline(
    ctx: data.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots how busy compute and comm/memcpy kernels keep the GPUs over time.

    Each time bin shows the kernels' busy time divided by the bin width
    times the group's device count, capped at 1.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one panel per group.
    """
    kernels = ctx.frame("kernels", options)
    panels = []
    for label in chart_helpers.unique_in_order(kernels, "label"):
        group_kernels = kernels.filter(pl.col("label") == label)
        end_ms = float(cast(float, group_kernels["end_ms"].max()))
        n_bins = int(options["bins"])
        bin_edges = np.linspace(0.0, max(end_ms, 1e-9), n_bins + 1)
        bin_width_ms = bin_edges[1] - bin_edges[0]
        n_devices = max(
            1,
            group_kernels.select(pl.struct("rank", "device").n_unique()).item(),
        )
        kernels_per_kind = (
            (
                "compute",
                group_kernels.filter(pl.col("category") == "compute"),
            ),
            (
                "comm / memcpy",
                group_kernels.filter(pl.col("category") != "compute"),
            ),
        )
        traces = []
        for kind_index, (name, kind_kernels) in enumerate(kernels_per_kind):
            busy_ms = _busy_ms_per_bin(
                kind_kernels["t_ms"].to_numpy(),
                kind_kernels["end_ms"].to_numpy(),
                bin_edges,
            )
            busy_fraction = busy_ms / (bin_width_ms * n_devices)
            traces.append(
                ir.Trace(
                    "step",
                    x=bin_edges[:-1].tolist(),
                    y=np.minimum(busy_fraction, 1.0).tolist(),
                    name=name,
                    style={"ci": kind_index * 2},
                    meta={"label": label},
                )
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=label,
                xlabel="time since first step (ms)",
                ylabel="busy fraction",
            )
        )
    return chart_helpers.grid(
        "Compute / communication busy fraction", panels, max_cols=1
    )


def _busy_ms_per_bin(
    start_ms: np.ndarray, end_ms: np.ndarray, bin_edges: np.ndarray
) -> np.ndarray:
    """Sums the kernel time (ms) that falls into each time bin.

    A kernel that spans several bins is split between them.

    Args:
        start_ms: Kernel start times.
        end_ms: Kernel end times.
        bin_edges: Increasing bin edges; times outside are clipped to the
            first or last bin.

    Returns:
        The busy time per bin.
    """
    n_bins = len(bin_edges) - 1
    busy_ms = np.zeros(n_bins)
    if len(start_ms) == 0:
        return busy_ms
    first_bin = np.clip(
        np.searchsorted(bin_edges, start_ms, side="right") - 1, 0, n_bins - 1
    )
    last_bin = np.clip(
        np.searchsorted(bin_edges, end_ms, side="left") - 1, 0, n_bins - 1
    )
    in_one_bin = first_bin == last_bin
    np.add.at(busy_ms, first_bin[in_one_bin], (end_ms - start_ms)[in_one_bin])
    for i in np.flatnonzero(~in_one_bin):
        for bin_index in range(first_bin[i], last_bin[i] + 1):
            overlap_start = max(start_ms[i], bin_edges[bin_index])
            overlap_end = min(end_ms[i], bin_edges[bin_index + 1])
            busy_ms[bin_index] += max(0.0, overlap_end - overlap_start)
    return busy_ms


@spec.register_plot(
    "stream_util",
    group="timeline",
    title="Per-stream GPU busy time",
    needs="kernels table",
    options={"top_k": spec.Opt("int", 16, "Busiest lanes shown.")},
)
def stream_util(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots the busy time of each rank/device/stream lane.

    Busy time is a share of the profiled step window of the lane's rank;
    only the top K lanes (by their highest share) are shown.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one bar series per group.
    """
    kernels = ctx.frame("kernels", options)
    steps = ctx.frame("steps", {**options, "where": None})
    window_per_rank = steps.group_by("label", "trial", "rank").agg(
        (pl.col("end_ms").max() - pl.col("t_ms").min()).alias("window_ms")
    )
    busy_per_lane = (
        kernels.with_columns(_lane_expr(kernels))
        .group_by("label", "trial", "rank", "lane")
        .agg(pl.col("duration_ms").sum().alias("busy_ms"))
        .join(window_per_rank, on=["label", "trial", "rank"])
        .with_columns(
            (100 * pl.col("busy_ms") / pl.col("window_ms"))
            .clip(0, 100)
            .alias("busy_pct")
        )
    )
    top_lanes = (
        busy_per_lane.group_by("lane")
        .agg(pl.col("busy_pct").max())
        .sort("busy_pct", descending=True)
        .head(int(options["top_k"]))["lane"]
        .to_list()
    )
    traces = []
    labels = chart_helpers.unique_in_order(kernels, "label")
    for i, label in enumerate(labels):
        lane_means = (
            busy_per_lane.filter(
                (pl.col("label") == label) & pl.col("lane").is_in(top_lanes)
            )
            .group_by("lane")
            .agg(pl.col("busy_pct").mean())
            .sort("lane")
        )
        traces.append(
            ir.Trace(
                "barh",
                x=lane_means["busy_pct"].to_list(),
                y=lane_means["lane"].to_list(),
                name=label,
                style={"ci": i},
                meta={"label": label},
            )
        )
    return chart_helpers.single(
        "Per-stream busy time",
        ir.Panel(
            traces=traces, xlabel="busy time / step window (%)", invert_y=True
        ),
    )
