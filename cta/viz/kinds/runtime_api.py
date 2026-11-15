"""Plots of CUDA runtime and OS API calls made by the host."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

_CLASSES = (
    ("sync", re.compile(r"synchronize|streamwait|eventquery|waitevent", re.I)),
    ("launch", re.compile(r"launch", re.I)),
    ("memory", re.compile(r"malloc|free|memcpy|memset|hostalloc", re.I)),
)
_CLASS_ORDER = ("launch", "sync", "memory", "other")


def api_class(name: str) -> str:
    """Sorts a CUDA runtime call into launch, sync, memory or other.

    Args:
        name: The API name, such as ``cudaStreamSynchronize``.

    Returns:
        ``sync`` for calls that wait, ``launch`` for kernel and graph
        launches, ``memory`` for allocation and copies, else ``other``.
    """
    for kind, pattern in _CLASSES:
        if pattern.search(name):
            return kind
    return "other"


def _classified(calls: pl.DataFrame) -> pl.DataFrame:
    """Adds an ``api_class`` column to a runtime-call table."""
    classes = {name: api_class(name) for name in calls["name"].unique()}
    return calls.with_columns(
        pl.col("name")
        .replace_strict(classes, return_dtype=pl.String)
        .alias("api_class")
    )


@spec.register_plot(
    "runtime_top",
    group="overview",
    title="Time in CUDA runtime or OS calls, by call",
    needs="runtime or osrt table (nsys, Kineto)",
    aliases=("api_top", "api_time"),
    options={
        "source": spec.Opt(
            "str",
            "runtime",
            "CUDA runtime calls, or OS runtime calls.",
            ("runtime", "osrt"),
        ),
        "metric": spec.Opt(
            "str",
            "total_ms",
            "Total time, mean time per call, or call count.",
            ("total_ms", "mean_us", "calls"),
        ),
        "top_k": spec.Opt("int", 15, "Calls with the largest total time."),
    },
)
def runtime_top(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Ranks API calls by the host time they take.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One horizontal grouped-bar panel, the costliest call at the top.

    Raises:
        spec.NotApplicableError: The report has no such API table.
    """
    table = ctx.frame(options["source"], options)
    totals = table.group_by("label", "name", maintain_order=True).agg(
        pl.col("duration_ns").sum(), pl.col("calls").sum()
    )
    top = (
        totals.group_by("name")
        .agg(pl.col("duration_ns").max())
        .top_k(max(1, options["top_k"]), by="duration_ns")
        .sort("duration_ns")
    )
    names = top["name"].to_list()
    metric = options["metric"]
    value = {
        "total_ms": pl.col("duration_ns") / 1e6,
        "mean_us": pl.col("duration_ns") / pl.col("calls") / 1e3,
        "calls": pl.col("calls"),
    }[metric]
    traces = []
    for index, label in enumerate(
        chart_helpers.unique_in_order(totals, "label")
    ):
        mine = totals.filter(pl.col("label") == label).select(
            "name", value.alias("value")
        )
        by_name = dict(mine.iter_rows())
        traces.append(
            ir.Trace(
                "barh",
                x=[by_name.get(name, float("nan")) for name in names],
                y=names,
                name=str(label),
                style={"ci": index},
                meta={"label": str(label)},
            )
        )
    unit = {"total_ms": "ms", "mean_us": "us per call", "calls": "calls"}[
        metric
    ]
    return chart_helpers.single(
        "Time in API calls",
        ir.Panel(traces=traces, xlabel=unit, ylabel=options["source"]),
    )


def _lanes_for(
    calls: pl.DataFrame, count: int
) -> tuple[list[str], dict[int, str]]:
    """Names the busiest threads of a table of calls, longest time first."""
    busiest = (
        calls.group_by("thread")
        .agg(pl.col("duration_ns").sum())
        .top_k(count, by="duration_ns")["thread"]
        .to_list()
    )
    names = {thread: f"host thread {i}" for i, thread in enumerate(busiest)}
    return list(names.values()), names


@spec.register_plot(
    "runtime_timeline",
    group="timeline",
    title="CUDA API calls and GPU kernels on one timeline",
    needs="runtime_calls table (nsys, Kineto)",
    aliases=("api_timeline", "host_timeline"),
    options={
        "start_ms": spec.Opt(
            "float", 0.0, "Window start, in ms since the rank's first event."
        ),
        "window_ms": spec.Opt("float", 20.0, "Window length in ms."),
        "threads": spec.Opt("int", 4, "Host threads to draw, busiest first."),
        "gpu_streams": spec.Opt(
            "int", 3, "GPU streams to draw below the threads (0 = none)."
        ),
    },
)
def runtime_timeline(
    ctx: data.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Draws host API calls above the GPU work they launch.

    One panel per group, for its lowest rank: a lane per busy host thread
    with each call coloured by class (launch, sync, memory, other), then
    lanes for the busiest GPU streams. Long ``sync`` bars with an empty GPU
    below them are the host waiting on the device; gaps in the GPU lanes
    under short ``launch`` calls are the device waiting on the host.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column grid of timelines.

    Raises:
        spec.NotApplicableError: The report has no runtime-call table, or
            the window holds no call.
    """
    calls = _classified(ctx.frame("runtime_calls", options))
    kernels = ctx.frame("kernels", options)
    low, high = (
        float(options["start_ms"]),
        float(options["start_ms"] + options["window_ms"]),
    )
    panels = []
    for label in chart_helpers.unique_in_order(calls, "label"):
        mine = calls.filter(pl.col("label") == label)
        rank = int(mine["rank"].to_numpy().min())
        window = mine.filter(
            pl.col("rank") == rank,
            pl.col("end_ms") > low,
            pl.col("t_ms") < high,
        )
        if window.is_empty():
            continue
        lane_names, thread_lane = _lanes_for(window, max(1, options["threads"]))
        window = window.filter(pl.col("thread").is_in(list(thread_lane)))
        window = window.with_columns(
            pl.col("thread")
            .replace_strict(thread_lane, return_dtype=pl.String)
            .alias("lane")
        )
        lanes = list(lane_names)
        gpu = kernels.filter(
            pl.col("label") == label,
            pl.col("rank") == rank,
            pl.col("end_ms") > low,
            pl.col("t_ms") < high,
        )
        if options["gpu_streams"] > 0 and not gpu.is_empty():
            streams = (
                gpu.group_by("stream")
                .agg(pl.col("duration_ms").sum())
                .top_k(options["gpu_streams"], by="duration_ms")["stream"]
                .to_list()
            )
            gpu = gpu.filter(pl.col("stream").is_in(streams)).with_columns(
                ("GPU stream " + pl.col("stream").cast(pl.String)).alias("lane")
            )
            lanes += [f"GPU stream {stream}" for stream in sorted(streams)]
        else:
            gpu = gpu.head(0)
        traces = []
        for index, kind in enumerate(_CLASS_ORDER):
            part = window.filter(pl.col("api_class") == kind)
            if part.is_empty():
                continue
            traces.append(_segments(part, lanes, kind, index, label, low, high))
        if not gpu.is_empty():
            traces.append(
                _segments(gpu, lanes, "GPU kernel", 5, label, low, high)
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=f"{label} (rank {rank})",
                xlabel="ms since the rank's first event",
                xrange=(low, high),
                invert_y=True,
            )
        )
    if not panels:
        raise spec.NotApplicableError("no API call falls in the window")
    return chart_helpers.grid("API calls and kernels", panels, max_cols=1)


def _segments(
    rows: pl.DataFrame,
    lanes: list[str],
    name: str,
    color: int,
    label: str,
    low: float,
    high: float,
) -> ir.Trace:
    """Builds one coloured segment trace, clipped to the window."""
    return ir.Trace(
        "segment",
        x=rows["t_ms"].clip(lower_bound=low).to_list(),
        x1=rows["end_ms"].clip(upper_bound=high).to_list(),
        y=rows["lane"].to_list(),
        name=name,
        text=rows["name"].to_list(),
        style={"lanes": lanes, "ci": color, "width": 9},
        meta={"label": label},
    )


@spec.register_plot(
    "sync_waits",
    group="timeline",
    title="Host time spent waiting on the device",
    needs="runtime_calls table (nsys, Kineto)",
    aliases=("sync", "host_waits"),
    options={
        "bins": spec.Opt("int", 30, "Bins of the wait-length histogram."),
        "top_k": spec.Opt("int", 8, "Sync calls to list, longest total first."),
    },
)
def sync_waits(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots how long the host blocks in synchronising calls.

    Left, how long each wait lasts, on a log axis; right, which calls the
    waiting time goes to.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        Two panels: wait lengths, and total wait by call.

    Raises:
        spec.NotApplicableError: The trace has no synchronising call.
    """
    calls = _classified(ctx.frame("runtime_calls", options)).filter(
        pl.col("api_class") == "sync"
    )
    if calls.is_empty():
        raise spec.NotApplicableError("the trace has no synchronising calls")
    waits = calls["duration_ns"].to_numpy().astype(float) / 1e3
    bins = max(2, int(options["bins"]))
    edges = np.geomspace(
        waits.min(), max(waits.max(), waits.min() * 2), bins + 1
    )
    names = (
        calls.group_by("name")
        .agg(pl.col("duration_ns").sum())
        .top_k(max(1, options["top_k"]), by="duration_ns")["name"]
        .to_list()
    )
    lengths, totals = [], []
    for index, label in enumerate(
        chart_helpers.unique_in_order(calls, "label")
    ):
        mine = calls.filter(pl.col("label") == label)
        histogram, _ = np.histogram(
            mine["duration_ns"].to_numpy() / 1e3, bins=edges
        )
        style = {"ci": index}
        meta = {"label": str(label)}
        lengths.append(
            ir.Trace(
                "step",
                x=edges[:-1].tolist(),
                y=histogram.tolist(),
                name=str(label),
                style=style,
                meta=meta,
            )
        )
        by_name = dict(
            mine.group_by("name").agg(pl.col("duration_ns").sum()).iter_rows()
        )
        totals.append(
            ir.Trace(
                "barh",
                x=[by_name.get(name, 0) / 1e6 for name in names],
                y=names,
                name=str(label),
                style=style,
                meta=meta,
            )
        )
    return chart_helpers.grid(
        "Host waits",
        [
            ir.Panel(
                traces=lengths,
                title="Length of a wait",
                xlabel="wait (us)",
                ylabel="calls",
                xscale="log",
            ),
            ir.Panel(
                traces=totals,
                title="Total wait by call",
                xlabel="ms",
                legend=False,
            ),
        ],
        max_cols=2,
        meta={"skip_suitability": True},
    )


@spec.register_plot(
    "runtime_step_mix",
    group="steps",
    title="Host API time per step, by kind of call",
    needs="runtime_calls table (nsys, Kineto)",
    aliases=("api_per_step",),
    options={},
)
def runtime_step_mix(
    ctx: data.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Stacks each step's API call time into launch, sync, memory and other.

    Calls of all threads add up, so a step can show more API time than its
    length. A step whose sync share grows is waiting on the GPU more.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One stacked-bar panel per group.

    Raises:
        spec.NotApplicableError: No call falls inside a step.
    """
    calls = _classified(ctx.frame("runtime_calls", options))
    calls = calls.filter(pl.col("idx").is_not_null())
    if calls.is_empty():
        raise spec.NotApplicableError("no API call falls inside a step")
    panels = []
    for label in chart_helpers.unique_in_order(calls, "label"):
        mine = calls.filter(pl.col("label") == label)
        per_step = (
            mine.group_by("idx", "api_class")
            .agg(pl.col("duration_ms").sum())
            .sort("idx")
        )
        steps = sorted(per_step["idx"].unique().to_list())
        traces = []
        for index, kind in enumerate(_CLASS_ORDER):
            by_step = dict(
                per_step.filter(pl.col("api_class") == kind)
                .select("idx", "duration_ms")
                .iter_rows()
            )
            if not by_step:
                continue
            traces.append(
                ir.Trace(
                    "bar",
                    x=[str(step) for step in steps],
                    y=[by_step.get(step, 0.0) for step in steps],
                    name=kind,
                    style={"ci": index},
                    meta={"label": str(label)},
                )
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=str(label),
                xlabel="step",
                ylabel="API time (ms)",
                stacked=True,
            )
        )
    return chart_helpers.grid("API time per step", panels, max_cols=1)
