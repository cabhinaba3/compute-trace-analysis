"""Plot kinds for communication calls: bandwidth, wait skew and size sweeps.

They read the ``network_ops`` source and reuse the aspect's arithmetic, so a
plot and the ``network_api`` summary agree.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import polars as pl

from cta.analysis import network
from cta.aspects import network as network_aspect
from cta.viz import context
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

_BANDWIDTH_CHOICES = ("busbw", "algbw")
_SPLIT_CHOICES = ("label", "api")
_GAP_CHOICES = ("wait_skew", "launch_gap")
_BUCKET_ORDER = {
    name: index
    for index, name in enumerate([*network.SIZE_BUCKET_LABELS, "unknown"])
}

# Options every network kind shares.
_COMMON_OPTIONS = {
    "api": spec.Opt("str", "", "Only this API (nccl, mpi, ucx, p2p)."),
    "op": spec.Opt("str", "", "Only this logical operation (allreduce, ...)."),
    "by": spec.Opt(
        "str",
        "label",
        "One series per group label, or per group label and API.",
        _SPLIT_CHOICES,
    ),
}


def _network_frame(
    ctx: context.DataContext, options: dict[str, Any]
) -> pl.DataFrame:
    """Reads the ``network_ops`` source with the API and operation filters.

    Raises:
        spec.NotApplicableError: There are no calls, or none after filtering.
    """
    ops = ctx.frame("network_ops", options)
    if options["api"]:
        ops = ops.filter(pl.col("api") == options["api"])
    if options["op"]:
        ops = ops.filter(pl.col("op") == options["op"])
    if ops.is_empty():
        raise spec.NotApplicableError("no network calls match api/op")
    return ops


def _series(
    ops: pl.DataFrame, split: str
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yields (series name, calls): one series per label, or label and API."""
    keys = ["label", "api"] if split == "api" else ["label"]
    for key, rows in ops.group_by(keys, maintain_order=True):
        yield " / ".join(str(part) for part in key), rows


def _per_trial(ops: pl.DataFrame) -> Iterator[pl.DataFrame]:
    """Splits one series' calls by trial, which the skew match needs."""
    for _, rows in ops.group_by("trial", maintain_order=True):
        yield rows


def _bucket_sort(frame: pl.DataFrame) -> pl.DataFrame:
    """Orders rows by operation, then by message-size bucket."""
    return frame.sort(
        "op",
        pl.col("bucket").replace_strict(
            _BUCKET_ORDER, default=len(_BUCKET_ORDER), return_dtype=pl.Int32
        ),
    )


def _pooled(ops: pl.DataFrame) -> pl.DataFrame:
    """Sums one series over its trials: per API, operation and bucket."""
    return (
        pl.concat([network_aspect.per_operation(t) for t in _per_trial(ops)])
        .group_by("api", "op", "bucket")
        .agg(pl.exclude("api", "op", "bucket").sum())
    )


@spec.register_plot(
    "network_bandwidth",
    group="distributed",
    title="Communication bandwidth by operation and size",
    needs="network_ops (nsys MPI/UCX tables, or Kineto NCCL kernels)",
    aliases=("netbw", "comm_bandwidth"),
    options={
        **_COMMON_OPTIONS,
        "metric": spec.Opt(
            "str",
            "busbw",
            "Bus bandwidth (nccl-tests) or algorithm bandwidth.",
            _BANDWIDTH_CHOICES,
        ),
        "top_k": spec.Opt(
            "int", 12, "Most time-consuming operation/size pairs (0 = all)."
        ),
    },
)
def network_bandwidth(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots bandwidth per operation and message-size bucket.

    Calls are matched across groups by logical operation and size bucket, so
    a group that ran NCCL lines up with one that ran MPI. Only calls that run
    for the whole transfer are timed; posted calls do not count.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of grouped bars, one bar per series and pair.

    Raises:
        spec.NotApplicableError: No timed call has a known message size.
    """
    ops = _network_frame(ctx, options)
    metric = options["metric"]
    column = "weighted_bytes" if metric == "busbw" else "timed_bytes"
    rates: dict[str, dict[str, float]] = {}
    time_by_key: dict[str, float] = {}
    for name, rows in _series(ops, options["by"]):
        pooled = (
            _pooled(rows)
            .group_by("op", "bucket")
            .agg(
                pl.col(column).sum(),
                pl.col("timed_ns").sum(),
                pl.col("duration_ms").sum(),
            )
            .filter(pl.col("timed_ns") > 0, pl.col("bucket") != "unknown")
        )
        for row in pooled.iter_rows(named=True):
            key = f"{row['op']}|{row['bucket']}"
            rates.setdefault(name, {})[key] = row[column] / row["timed_ns"]
            time_by_key[key] = (
                time_by_key.get(key, float("nan")) + row["duration_ms"]
            )
    if not rates:
        raise spec.NotApplicableError(
            "no timed network call has a known message size (nsys cannot "
            "size NCCL calls; use a Kineto trace for those)"
        )
    keys = sorted(time_by_key, key=lambda key: -time_by_key[key])
    if options["top_k"] > 0:
        keys = keys[: options["top_k"]]
    keys.sort(
        key=lambda key: (
            key.split("|")[0],
            _BUCKET_ORDER.get(key.split("|")[1], len(_BUCKET_ORDER)),
        )
    )
    traces = [
        ir.Trace(
            "bar",
            x=keys,
            y=[rates[name].get(key, float("nan")) for key in keys],
            name=name,
            style={"ci": index},
            meta={"label": name.split(" / ")[0]},
        )
        for index, name in enumerate(rates)
    ]
    label = "bus bandwidth" if metric == "busbw" else "algorithm bandwidth"
    return chart_helpers.single(
        f"Communication {label}",
        ir.Panel(
            traces=traces,
            xlabel="operation | message size",
            ylabel=f"{label} (GB/s)",
        ),
    )


@spec.register_plot(
    "network_skew",
    group="distributed",
    title="Cross-rank wait skew and launch gap by operation",
    needs="network_ops from two or more ranks",
    aliases=("comm_skew", "wait_skew"),
    options={
        **_COMMON_OPTIONS,
        "metric": spec.Opt(
            "str",
            "wait_skew",
            "Spread of call durations across ranks, or time between the "
            "host launch and the GPU start.",
            _GAP_CHOICES,
        ),
    },
)
def network_skew(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots, per operation, how unevenly ranks wait in the same call.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of grouped bars in microseconds.

    Raises:
        spec.NotApplicableError: The chosen metric cannot be computed, such as
            skew from a single rank or launch gap from an nsys-less trace.
    """
    ops = _network_frame(ctx, options)
    wait = options["metric"] == "wait_skew"
    column = "skew_us" if wait else "gap_us"
    measure = (
        network_aspect.wait_skew_us if wait else network_aspect.launch_gap_us
    )
    values: dict[str, dict[str, float]] = {}
    keys: set[str] = set()
    for name, rows in _series(ops, options["by"]):
        frames = [measure(trial) for trial in _per_trial(rows)]
        frames = [frame for frame in frames if frame.height]
        if not frames:
            continue
        mean = (
            pl.concat(frames).group_by("api", "op").agg(pl.col(column).mean())
        )
        for row in mean.iter_rows(named=True):
            key = f"{row['api']}|{row['op']}"
            values.setdefault(name, {})[key] = row[column]
            keys.add(key)
    if not values:
        raise spec.NotApplicableError(
            "needs collectives seen on two or more ranks"
            if wait
            else "no call has a known launch time (nsys and Kineto only)"
        )
    ordered = sorted(keys)
    traces = [
        ir.Trace(
            "bar",
            x=ordered,
            y=[values[name].get(key, float("nan")) for key in ordered],
            name=name,
            style={"ci": index},
            meta={"label": name.split(" / ")[0]},
        )
        for index, name in enumerate(values)
    ]
    title = "Cross-rank wait skew" if wait else "Launch-to-start gap"
    return chart_helpers.single(
        title,
        ir.Panel(
            traces=traces,
            xlabel="API | operation",
            ylabel=f"{title.lower()} (us)",
        ),
    )


@spec.register_plot(
    "network_size_sweep",
    group="distributed",
    title="Bandwidth against message size, per operation",
    needs="network_ops with message sizes",
    aliases=("size_sweep", "comm_sweep"),
    options={
        **_COMMON_OPTIONS,
        "metric": spec.Opt(
            "str",
            "busbw",
            "Bus bandwidth (nccl-tests) or algorithm bandwidth.",
            _BANDWIDTH_CHOICES,
        ),
        "max_ops": spec.Opt(
            "int", 4, "Operations to draw, most time-consuming first."
        ),
    },
)
def network_size_sweep(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots bandwidth against mean message size, one panel per operation.

    Each point is a size bucket; its x is the mean message size of the calls
    in it. Libraries running the same workload draw on the same axes.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A grid of log-log panels with one line per series.

    Raises:
        spec.NotApplicableError: No timed call has a known message size.
    """
    ops = _network_frame(ctx, options)
    column = "weighted_bytes" if options["metric"] == "busbw" else "timed_bytes"
    lines: dict[str, list[tuple[str, str, float, float]]] = {}
    time_by_op: dict[str, float] = {}
    for name, rows in _series(ops, options["by"]):
        pooled = (
            _bucket_sort(_pooled(rows))
            .filter(pl.col("timed_ns") > 0, pl.col("bucket") != "unknown")
            .filter(pl.col("calls") > 0)
        )
        for row in pooled.iter_rows(named=True):
            time_by_op[row["op"]] = (
                time_by_op.get(row["op"], 0.0) + row["duration_ms"]
            )
            lines.setdefault(name, []).append(
                (
                    row["op"],
                    row["bucket"],
                    row["bytes"] / row["calls"],
                    row[column] / row["timed_ns"],
                )
            )
    if not lines:
        raise spec.NotApplicableError(
            "no timed network call has a known message size (nsys cannot "
            "size NCCL calls; use a Kineto trace for those)"
        )
    chosen = sorted(time_by_op, key=lambda op: -time_by_op[op])
    chosen = chosen[: max(1, options["max_ops"])]
    label = "bus bandwidth" if options["metric"] == "busbw" else "algorithm bw"
    panels = []
    for op in chosen:
        traces = []
        for index, (name, points) in enumerate(lines.items()):
            mine = [point for point in points if point[0] == op]
            if not mine:
                continue
            traces.append(
                ir.Trace(
                    "line",
                    x=[point[2] for point in mine],
                    y=[point[3] for point in mine],
                    name=name,
                    text=[point[1] for point in mine],
                    style={"ci": index, "marker": "o"},
                    meta={"label": name.split(" / ")[0]},
                )
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=op,
                xlabel="mean message size (bytes)",
                ylabel=f"{label} (GB/s)",
                xscale="log",
                yscale="log",
            )
        )
    return chart_helpers.grid("Bandwidth against message size", panels, 2)


_MATRIX_VALUES = ("bytes", "calls", "time_ms")


@spec.register_plot(
    "network_matrix",
    group="distributed",
    title="Point-to-point traffic between ranks",
    needs="network_ops with a known peer (nsys MPI send/recv)",
    aliases=("p2p_matrix", "comm_matrix"),
    options={
        **_COMMON_OPTIONS,
        "value": spec.Opt(
            "str",
            "bytes",
            "What each cell sums: megabytes, calls, or milliseconds.",
            _MATRIX_VALUES,
        ),
    },
)
def network_matrix(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots who sends to whom, one heatmap per group label.

    Rows are the sending rank and columns the receiving rank. Sends are
    counted; when a trace has receives only, those are read the other way
    round. Calls whose peer the trace does not record are left out, so a
    collective never shows up here.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A grid of heatmaps that share one colour scale.

    Raises:
        spec.NotApplicableError: No call has a known peer.
    """
    ops = _network_frame(ctx, options).filter(pl.col("peer") >= 0)
    if ops.is_empty():
        raise spec.NotApplicableError(
            "no call records its peer rank (nsys MPI send/recv do; NCCL "
            "kernels and UCX submits do not)"
        )
    sends = ops.filter(pl.col("op") == "send")
    if sends.is_empty():
        sends = ops.filter(pl.col("op") == "recv").rename(
            {"rank": "peer", "peer": "rank"}
        )
    if sends.is_empty():
        raise spec.NotApplicableError("no point-to-point send or recv call")
    value = options["value"]
    column = {"bytes": "bytes", "calls": "bytes", "time_ms": "duration_ns"}[
        value
    ]
    cells = (
        sends.group_by("label", "rank", "peer")
        .agg(
            pl.col(column).sum().alias("total"),
            pl.len().alias("calls"),
        )
        .with_columns(
            (
                pl.col("calls") if value == "calls" else pl.col("total") / 1e6
            ).alias("cell")
        )
    )
    ranks = sorted(set(cells["rank"].to_list()) | set(cells["peer"].to_list()))
    index = {rank: position for position, rank in enumerate(ranks)}
    top = max([*cells["cell"].to_list(), 1e-9])
    unit = {"bytes": "MB", "calls": "calls", "time_ms": "ms"}[value]
    panels = []
    for label in chart_helpers.unique_in_order(cells, "label"):
        grid = [[0.0] * len(ranks) for _ in ranks]
        for row in cells.filter(pl.col("label") == label).iter_rows(named=True):
            grid[index[row["rank"]]][index[row["peer"]]] = row["cell"]
        panels.append(
            ir.Panel(
                traces=[
                    ir.Trace(
                        "heatmap",
                        x=ranks,
                        y=ranks,
                        z=grid,
                        style={"cmap": "Blues", "vmin": 0.0, "vmax": top},
                        meta={"label": label},
                    )
                ],
                title=label,
                xlabel="Destination rank",
                ylabel="Source rank",
                colorbar=unit,
                legend=False,
            )
        )
    return chart_helpers.grid("Point-to-point traffic", panels, 2)
