"""Plots of copies between host and devices, and between devices."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers

_DIRECTIONS = (
    ("memset", re.compile(r"memset", re.I)),
    ("H2D", re.compile(r"htod|h2d", re.I)),
    ("D2H", re.compile(r"dtoh|d2h", re.I)),
    ("D2D", re.compile(r"dtod|d2d", re.I)),
    ("P2P", re.compile(r"ptop|p2p|peer", re.I)),
    ("H2H", re.compile(r"htoh|h2h", re.I)),
)
_ORDER = [name for name, _ in _DIRECTIONS] + ["other"]
_COMMON = {
    "direction": registry.Opt(
        "str",
        "",
        "Only this direction: H2D, D2H, D2D, P2P, H2H or memset.",
    ),
    "by": registry.Opt(
        "str",
        "label",
        "One series per group label, or per label and rank.",
        ("label", "rank"),
    ),
}


def direction_of(name: str) -> str:
    """Names the direction of a copy from its kernel name.

    Args:
        name: A copy's name, such as ``Memcpy_H2D`` (nsys) or
            ``Memcpy HtoD (Pinned -> Device)`` (Kineto).

    Returns:
        ``H2D``, ``D2H``, ``D2D``, ``P2P``, ``H2H`` or ``memset``; ``other``
        when the name says none of them.
    """
    for direction, pattern in _DIRECTIONS:
        if pattern.search(name):
            return direction
    return "other"


def _copies(ctx: context.DataContext, options: dict[str, Any]) -> pl.DataFrame:
    """Reads the copy rows of ``kernels`` with a ``direction`` column.

    Raises:
        NotApplicableError: The trace has no copies, or none in the
            requested direction.
    """
    kernels = ctx.frame("kernels", options)
    copies = kernels.filter(pl.col("category") == "memcpy")
    if copies.is_empty():
        raise registry.NotApplicableError("the trace has no memory copies")
    names = copies["name"].unique().to_list()
    directions = {name: direction_of(name) for name in names}
    copies = copies.with_columns(
        pl.col("name")
        .replace_strict(directions, return_dtype=pl.String)
        .alias("direction")
    )
    if options["direction"]:
        copies = copies.filter(pl.col("direction") == options["direction"])
        if copies.is_empty():
            raise registry.NotApplicableError(
                f"no copies in direction {options['direction']}"
            )
    return copies


def _series(copies: pl.DataFrame, by: str) -> list[tuple[str, pl.DataFrame]]:
    """Splits the copies into one series per label, or label and rank."""
    keys = ["label", "rank"] if by == "rank" and "rank" in copies else ["label"]
    return [
        (" / ".join(str(part) for part in key), rows)
        for key, rows in copies.group_by(keys, maintain_order=True)
    ]


@registry.register_plot(
    "memcpy_bandwidth",
    group="overview",
    title="Copy bandwidth, time or volume by direction",
    needs="kernels table with memory copies",
    aliases=("copy_bandwidth", "memcpy"),
    options={
        **_COMMON,
        "metric": registry.Opt(
            "str",
            "bandwidth",
            "GB/s over the copies, total time, total volume, or call count.",
            ("bandwidth", "time_ms", "megabytes", "calls"),
        ),
    },
)
def memcpy_bandwidth(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots what copies achieve in each direction.

    Bandwidth is bytes over the time the copies took, so it is an average
    over the calls, and copies with an unknown size are left out of it.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One grouped-bar panel with a bar per series and direction.

    Raises:
        NotApplicableError: There are no copies to measure.
    """
    copies = _copies(ctx, options)
    metric = options["metric"]
    if metric == "bandwidth":
        copies = copies.filter(pl.col("bytes") > 0)
        if copies.is_empty():
            raise registry.NotApplicableError("no copy records its size")
    directions = [
        d for d in _ORDER if d in copies["direction"].unique().to_list()
    ]
    traces = []
    for index, (name, rows) in enumerate(_series(copies, options["by"])):
        totals = rows.group_by("direction").agg(
            pl.col("bytes").sum(),
            pl.col("duration_ns").sum(),
            pl.len().alias("calls"),
        )
        value = {
            "bandwidth": pl.col("bytes") / pl.col("duration_ns"),
            "time_ms": pl.col("duration_ns") / 1e6,
            "megabytes": pl.col("bytes") / 1e6,
            "calls": pl.col("calls"),
        }[metric]
        by_direction = dict(
            totals.select("direction", value.alias("value")).iter_rows()
        )
        traces.append(
            ir.Trace(
                "bar",
                x=directions,
                y=[by_direction.get(d, float("nan")) for d in directions],
                name=name,
                style={"ci": index},
                meta={"label": name.split(" / ")[0]},
            )
        )
    unit = {
        "bandwidth": "GB/s",
        "time_ms": "ms",
        "megabytes": "MB",
        "calls": "calls",
    }[metric]
    return chart_helpers.single(
        "Copies by direction",
        ir.Panel(traces=traces, xlabel="direction", ylabel=unit),
    )


@registry.register_plot(
    "memcpy_sizes",
    group="overview",
    title="Copy sizes",
    needs="kernels table with memory copies of known size",
    aliases=("copy_sizes",),
    options={
        **_COMMON,
        "bins": registry.Opt("int", 24, "Bins, evenly spaced on a log axis."),
        "weight": registry.Opt(
            "str",
            "calls",
            "Count the calls in each bin, or the bytes they moved.",
            ("calls", "bytes"),
        ),
    },
)
def memcpy_sizes(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots how copy sizes are spread, on a log size axis.

    Many tiny copies cost latency and a few large ones cost bandwidth; the
    two look alike in a plain total.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One panel with a step line per series.

    Raises:
        NotApplicableError: No copy records its size.
    """
    copies = _copies(ctx, options).filter(pl.col("bytes") > 0)
    if copies.is_empty():
        raise registry.NotApplicableError("no copy records its size")
    sizes = copies["bytes"].to_numpy().astype(float)
    bins = max(2, int(options["bins"]))
    edges = np.geomspace(
        sizes.min(), max(sizes.max(), sizes.min() * 2), bins + 1
    )
    traces = []
    for index, (name, rows) in enumerate(_series(copies, options["by"])):
        values = rows["bytes"].to_numpy().astype(float)
        weights = values if options["weight"] == "bytes" else None
        counts, _ = np.histogram(values, bins=edges, weights=weights)
        traces.append(
            ir.Trace(
                "step",
                x=edges[:-1].tolist(),
                y=counts.tolist(),
                name=name,
                style={"ci": index},
                meta={"label": name.split(" / ")[0]},
            )
        )
    return chart_helpers.single(
        "Copy sizes",
        ir.Panel(
            traces=traces,
            xlabel="copy size (bytes)",
            ylabel="bytes moved" if options["weight"] == "bytes" else "calls",
            xscale="log",
        ),
        meta={"skip_suitability": True},
    )


@registry.register_plot(
    "memcpy_overlap",
    group="overview",
    title="Copy time hidden behind compute",
    needs="kernels table with memory copies and compute kernels",
    aliases=("copy_overlap",),
    options={**_COMMON},
)
def memcpy_overlap(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots how much copy time runs while a compute kernel does.

    Overlap is measured on each device on its own: the share of a copy's
    time that falls inside the union of that device's compute kernels. The
    rest is exposed copy time, which lengthens the step.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        Two panels: the overlapped share per direction, and the exposed copy
        time.

    Raises:
        NotApplicableError: There are no copies or no compute kernels.
    """
    copies = _copies(ctx, options)
    compute = ctx.frame("kernels", options).filter(
        pl.col("category") == "compute"
    )
    if compute.is_empty():
        raise registry.NotApplicableError("the trace has no compute kernels")
    overlapped = np.zeros(copies.height)
    copy_start = copies["t_ms"].to_numpy().astype(float)
    copy_end = copies["end_ms"].to_numpy().astype(float)
    labels = copies["label"].to_numpy()
    ranks = copies["rank"].to_numpy()
    devices = copies["device"].to_numpy()
    for (label, rank, device), kernels in compute.group_by(
        ["label", "rank", "device"]
    ):
        merged = chart_helpers.merge_intervals(
            kernels["t_ms"].to_numpy().astype(float),
            kernels["end_ms"].to_numpy().astype(float),
        )
        here = np.flatnonzero(
            (labels == label) & (ranks == rank) & (devices == device)
        )
        if len(here):
            overlapped[here] = chart_helpers.covered_length(
                *merged, copy_start[here], copy_end[here]
            )
    copies = copies.with_columns(
        pl.Series("overlap_ms", overlapped),
        (pl.col("duration_ms").cast(pl.Float64)).alias("copy_ms"),
    )
    directions = [
        d for d in _ORDER if d in copies["direction"].unique().to_list()
    ]
    share_traces, exposed_traces = [], []
    for index, (name, rows) in enumerate(_series(copies, options["by"])):
        totals = rows.group_by("direction").agg(
            pl.col("overlap_ms").sum(), pl.col("copy_ms").sum()
        )
        by_direction = {
            row["direction"]: row for row in totals.iter_rows(named=True)
        }
        nan = float("nan")
        style = {"ci": index}
        meta = {"label": name.split(" / ")[0]}
        share_traces.append(
            ir.Trace(
                "bar",
                x=directions,
                y=[
                    100.0
                    * by_direction[d]["overlap_ms"]
                    / by_direction[d]["copy_ms"]
                    if d in by_direction and by_direction[d]["copy_ms"] > 0
                    else nan
                    for d in directions
                ],
                name=name,
                style=style,
                meta=meta,
            )
        )
        exposed_traces.append(
            ir.Trace(
                "bar",
                x=directions,
                y=[
                    (by_direction[d]["copy_ms"] - by_direction[d]["overlap_ms"])
                    if d in by_direction
                    else nan
                    for d in directions
                ],
                name=name,
                style=style,
                meta=meta,
            )
        )
    return chart_helpers.grid(
        "Copies and compute",
        [
            ir.Panel(
                traces=share_traces,
                title="Copy time that overlaps compute",
                xlabel="direction",
                ylabel="% of copy time",
            ),
            ir.Panel(
                traces=exposed_traces,
                title="Exposed copy time",
                xlabel="direction",
                ylabel="ms",
                legend=False,
            ),
        ],
        max_cols=2,
    )
