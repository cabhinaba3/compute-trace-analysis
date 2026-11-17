"""Plot kinds of a raw-table metric over time."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers

_TIME_COL_CHOICES = ("t_ms", "start_ns", "elapsed_s", "idx", "step")


_BY_TRACE_CHOICES = (
    "category",
    "op_family",
    "rank",
    "device",
    "stream",
    "label",
    "none",
)


_METRIC_TRACE_OPTIONS = {
    "source": registry.Opt(
        "str",
        "kernels",
        "Source raw table.",
        chart_helpers.TIMED_SOURCES,
    ),
    "metric": registry.Opt(
        "str",
        "duration_ms",
        "Metric column to plot over time.",
    ),
    "time_col": registry.Opt(
        "str",
        "t_ms",
        "Time or step column for X axis.",
        _TIME_COL_CHOICES,
    ),
    "by": registry.Opt(
        "str",
        "category",
        "Line grouping column.",
        _BY_TRACE_CHOICES,
    ),
    "rolling": registry.Opt(
        "int",
        0,
        "Rolling-median smoothing window in samples (0 = disabled).",
    ),
    "downsample": registry.Opt(
        "int",
        4000,
        "Maximum points plotted per line (0 = disabled).",
    ),
    "yscale": registry.Opt(
        "str",
        "linear",
        "Scale of Y axis.",
        ("linear", "log"),
    ),
    "unit": registry.Opt(
        "str",
        "ms",
        "Time axis display unit.",
        ("ms", "s", "ns", "step"),
    ),
}


@registry.register_plot(
    "metric_trace",
    group="timeline",
    title="Metric trace over time",
    needs="raw table (kernels, gpu_metrics, steps, network_ops, or jtop)",
    aliases=("metric_timeseries", "time_series"),
    options=_METRIC_TRACE_OPTIONS,
)
def metric_trace(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots metric trends over continuous time or steps.

    Supports line grouping, rolling-median smoothing, and downsampling.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart showing time series metric trends.

    Raises:
        NotApplicableError: The table or required columns are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise registry.NotApplicableError(f"{source} table is empty")

    requested_time = options["time_col"]
    if requested_time in df.columns:
        time_col = requested_time
    elif source == "jtop" and "elapsed_s" in df.columns:
        time_col = "elapsed_s"
    elif "t_ms" in df.columns:
        time_col = "t_ms"
    elif "idx" in df.columns:
        time_col = "idx"
    elif "step" in df.columns:
        time_col = "step"
    elif "start_ns" in df.columns:
        time_col = "start_ns"
    else:
        raise registry.NotApplicableError(
            f"No valid time column '{requested_time}' in {source}"
        )

    metric = options["metric"]
    if metric not in df.columns:
        if source == "gpu_metrics" and "value" in df.columns:
            metric = "value"
        elif source == "jtop" and "power_total_w" in df.columns:
            metric = "power_total_w"
        else:
            raise registry.NotApplicableError(
                f"Metric '{metric}' not found in {source}"
            )

    df = df.filter(
        pl.col(time_col).is_not_null() & pl.col(metric).is_not_null()
    )
    if df.is_empty():
        raise registry.NotApplicableError("No non-null data for metric trace")

    if options["yscale"] == "log":
        df = df.filter(pl.col(metric) > 0)
        if df.is_empty():
            raise registry.NotApplicableError(
                "Log scale requested but no positive values"
            )

    by_col = options["by"]
    has_labels = "label" in df.columns and df["label"].n_unique() > 1

    group_keys: list[tuple[str, pl.DataFrame]] = []
    if by_col == "none" or by_col not in df.columns:
        if has_labels:
            for label in chart_helpers.unique_in_order(df, "label"):
                group_keys.append((label, df.filter(pl.col("label") == label)))
        else:
            group_keys.append(("all", df))
    else:
        if has_labels:
            for label in chart_helpers.unique_in_order(df, "label"):
                sub_df = df.filter(pl.col("label") == label)
                for val in chart_helpers.unique_in_order(sub_df, by_col):
                    group_keys.append(
                        (
                            f"{label} {val}",
                            sub_df.filter(pl.col(by_col) == val),
                        )
                    )
        else:
            for val in chart_helpers.unique_in_order(df, by_col):
                group_keys.append((str(val), df.filter(pl.col(by_col) == val)))

    downsample = int(options["downsample"])
    rolling = int(options["rolling"])
    traces: list[ir.Trace] = []

    for color_idx, (name, grp) in enumerate(group_keys):
        sorted_grp = grp.sort(time_col)
        x = sorted_grp[time_col].to_numpy().astype(float)
        y = sorted_grp[metric].to_numpy().astype(float)
        if options["unit"] == "s" and time_col == "t_ms":
            x = x / 1000.0

        if downsample > 0 and len(x) > downsample:
            indices = np.linspace(0, len(x) - 1, downsample, dtype=int)
            x = x[indices]
            y = y[indices]

        traces.append(
            ir.Trace(
                mark="line",
                x=x.tolist(),
                y=y.tolist(),
                name=name,
                style={"ci": color_idx, "width": 1.2},
            )
        )

        if rolling > 1 and len(y) >= rolling:
            y_smooth = chart_helpers.rolling_median(y, rolling)
            traces.append(
                ir.Trace(
                    mark="line",
                    x=x.tolist(),
                    y=y_smooth.tolist(),
                    name=f"{name} (median-{rolling})",
                    style={"ci": color_idx, "width": 2.2, "alpha": 0.85},
                )
            )

    if not traces:
        raise registry.NotApplicableError("No valid lines to plot")

    if time_col in ("idx", "step"):
        xlabel = "Step"
    else:
        xlabel = f"Time ({options['unit']})"

    panel = ir.Panel(
        traces=traces,
        xlabel=xlabel,
        ylabel=metric,
        yscale=options["yscale"],
    )
    return chart_helpers.single("Metric trace over time", panel)


@registry.register_plot(
    "line_plot",
    group="timeline",
    title="Metric line plot over time",
    needs="raw table (kernels, gpu_metrics, steps, network_ops, or jtop)",
    aliases=("line", "lineplot"),
    options=_METRIC_TRACE_OPTIONS,
)
def line_plot(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots metric trends as a line plot over time or steps."""
    return metric_trace(ctx, options)
