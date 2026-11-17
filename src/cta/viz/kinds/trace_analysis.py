"""Trace analysis plot kinds for time series, distributions, and comparisons.

Covers metric time series trends, statistical distributions, categorical
metric comparisons, execution heatmaps, and metric correlations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from cta.viz import context
from cta.viz import ir
from cta.viz import spec
from cta.viz import theme
from cta.viz.kinds import chart_helpers

_SOURCE_CHOICES = (
    "kernels",
    "gpu_metrics",
    "steps",
    "network_ops",
    "jtop",
)
_DIST_SOURCE_CHOICES = (
    "kernels",
    "steps",
    "network_ops",
    "gpu_metrics",
    "jtop",
)
_CAT_SOURCE_CHOICES = (
    "kernels",
    "steps",
    "network_ops",
    "runtime",
    "osrt",
)
_CORR_SOURCE_CHOICES = (
    "kernels",
    "network_ops",
    "steps",
    "gpu_metrics",
    "ncu_kernels",
)
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
_BY_DIST_CHOICES = (
    "label",
    "category",
    "op_family",
    "rank",
    "device",
    "none",
)
_BY_CAT_CHOICES = ("label", "rank", "device", "trial", "none")
_BY_CORR_CHOICES = (
    "category",
    "op_family",
    "rank",
    "device",
    "stream",
    "label",
    "none",
)
_DIST_MODES = ("hist", "ecdf", "box", "percentile", "violin")
_CAT_CHOICES = (
    "op_family",
    "category",
    "name",
    "scope",
    "stream",
    "rank",
    "device",
)
_AGG_CHOICES = ("sum", "mean", "median", "p50", "p90", "p99", "count")
_SORT_CHOICES = ("desc", "asc", "none")
_ORIENTATION_CHOICES = ("v", "h")
_HEATMAP_X_CHOICES = ("step", "time")
_HEATMAP_Y_CHOICES = ("rank", "device", "stream", "category", "op_family")
_HEATMAP_AGG_CHOICES = ("sum", "mean", "max", "count")
_CMAP_CHOICES = ("magma", "viridis", "plasma", "cividis", "Blues", "YlOrRd")


_METRIC_TRACE_OPTIONS = {
    "source": spec.Opt(
        "str",
        "kernels",
        "Source raw table.",
        _SOURCE_CHOICES,
    ),
    "metric": spec.Opt(
        "str",
        "duration_ms",
        "Metric column to plot over time.",
    ),
    "time_col": spec.Opt(
        "str",
        "t_ms",
        "Time or step column for X axis.",
        _TIME_COL_CHOICES,
    ),
    "by": spec.Opt(
        "str",
        "category",
        "Line grouping column.",
        _BY_TRACE_CHOICES,
    ),
    "rolling": spec.Opt(
        "int",
        0,
        "Rolling-median smoothing window in samples (0 = disabled).",
    ),
    "downsample": spec.Opt(
        "int",
        4000,
        "Maximum points plotted per line (0 = disabled).",
    ),
    "yscale": spec.Opt(
        "str",
        "linear",
        "Scale of Y axis.",
        ("linear", "log"),
    ),
    "unit": spec.Opt(
        "str",
        "ms",
        "Time axis display unit.",
        ("ms", "s", "ns", "step"),
    ),
}


@spec.register_plot(
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
        spec.NotApplicableError: The table or required columns are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise spec.NotApplicableError(f"{source} table is empty")

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
        raise spec.NotApplicableError(
            f"No valid time column '{requested_time}' in {source}"
        )

    metric = options["metric"]
    if metric not in df.columns:
        if source == "gpu_metrics" and "value" in df.columns:
            metric = "value"
        elif source == "jtop" and "power_total_w" in df.columns:
            metric = "power_total_w"
        else:
            raise spec.NotApplicableError(
                f"Metric '{metric}' not found in {source}"
            )

    df = df.filter(
        pl.col(time_col).is_not_null() & pl.col(metric).is_not_null()
    )
    if df.is_empty():
        raise spec.NotApplicableError("No non-null data for metric trace")

    if options["yscale"] == "log":
        df = df.filter(pl.col(metric) > 0)
        if df.is_empty():
            raise spec.NotApplicableError(
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
        raise spec.NotApplicableError("No valid lines to plot")

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


@spec.register_plot(
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


_DIST_BASE_OPTIONS = {
    "source": spec.Opt(
        "str",
        "kernels",
        "Source table.",
        _DIST_SOURCE_CHOICES,
    ),
    "metric": spec.Opt(
        "str",
        "duration_ms",
        "Numeric column for distribution analysis.",
    ),
    "by": spec.Opt(
        "str",
        "label",
        "Grouping column.",
        _BY_DIST_CHOICES,
    ),
    "logx": spec.Opt(
        "bool",
        False,
        "Log-scale for metric values / X axis.",
    ),
    "logy": spec.Opt(
        "bool",
        False,
        "Log-scale for counts/density / Y axis.",
    ),
}


@spec.register_plot(
    "metric_dist",
    group="overview",
    title="Metric distribution",
    needs="any numeric column from kernels, steps, network_ops, or gpu_metrics",
    aliases=("dist", "distribution", "distributions"),
    options={
        **_DIST_BASE_OPTIONS,
        "mode": spec.Opt(
            "str",
            "hist",
            "Distribution presentation format.",
            _DIST_MODES,
        ),
        "bins": spec.Opt("int", 50, "Histogram bin count."),
        "density": spec.Opt(
            "bool",
            True,
            "Normalize histogram to probability density.",
        ),
        "percentiles": spec.Opt(
            "list",
            ["50", "90", "95", "99", "99.9"],
            "Percentile cutoffs to compute and display.",
        ),
    },
)
def metric_dist(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots statistical distributions across trace events.

    Supports histograms, empirical CDFs, box-and-whisker plots, and percentiles.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart showing statistical metric distributions.

    Raises:
        spec.NotApplicableError: The table or numeric column is missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise spec.NotApplicableError(f"{source} table is empty")

    metric = options["metric"]
    if metric not in df.columns:
        raise spec.NotApplicableError(
            f"Metric '{metric}' not found in {source}"
        )

    df = df.filter(pl.col(metric).is_not_null())
    if df.is_empty():
        raise spec.NotApplicableError("No valid numeric values")

    by_col = options["by"]
    groups: list[tuple[str, pl.DataFrame]] = []
    if by_col == "none" or by_col not in df.columns:
        groups.append(("all", df))
    else:
        for val in chart_helpers.unique_in_order(df, by_col):
            groups.append((str(val), df.filter(pl.col(by_col) == val)))

    mode = options["mode"]
    traces: list[ir.Trace] = []
    notes: list[str] = []

    if mode == "hist":
        for color_idx, (name, grp) in enumerate(groups):
            vals = grp[metric].to_numpy().astype(float)
            if options["logx"]:
                vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            traces.append(
                ir.Trace(
                    mark="hist",
                    y=vals.tolist(),
                    name=name,
                    style={
                        "bins": int(options["bins"]),
                        "density": bool(options["density"]),
                        "alpha": 0.55,
                        "ci": color_idx,
                    },
                )
            )
            notes.append(
                f"{name}: N={len(vals)}, median={np.median(vals):.3g}, "
                f"p99={np.percentile(vals, 99):.3g}"
            )
        ylabel = "Density" if options["density"] else "Count"
        panel = ir.Panel(
            traces=traces,
            xlabel=metric,
            ylabel=ylabel,
            xscale="log" if options["logx"] else "linear",
            yscale="log" if options["logy"] else "linear",
        )

    elif mode == "ecdf":
        for color_idx, (name, grp) in enumerate(groups):
            vals = np.sort(grp[metric].to_numpy().astype(float))
            if options["logx"]:
                vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            traces.append(
                ir.Trace(
                    mark="step",
                    x=vals.tolist(),
                    y=cdf.tolist(),
                    name=name,
                    style={"ci": color_idx, "width": 1.5},
                )
            )
            notes.append(
                f"{name}: median={np.median(vals):.3g}, "
                f"p90={np.percentile(vals, 90):.3g}, "
                f"p99={np.percentile(vals, 99):.3g}"
            )
        panel = ir.Panel(
            traces=traces,
            xlabel=metric,
            ylabel="Cumulative Probability (ECDF)",
            xscale="log" if options["logx"] else "linear",
            yscale="log" if options["logy"] else "linear",
        )

    elif mode == "box":
        for color_idx, (name, grp) in enumerate(groups):
            vals = grp[metric].to_numpy().astype(float)
            if options["logy"]:
                vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            traces.append(
                ir.Trace(
                    mark="box",
                    y=vals.tolist(),
                    name=name,
                    style={"ci": color_idx},
                )
            )
            notes.append(
                f"{name}: median={np.median(vals):.3g}, "
                f"IQR={np.percentile(vals, 75) - np.percentile(vals, 25):.3g}"
            )
        panel = ir.Panel(
            traces=traces,
            ylabel=metric,
            yscale="log" if options["logy"] else "linear",
        )

    elif mode == "violin":
        for color_idx, (name, grp) in enumerate(groups):
            vals = grp[metric].to_numpy().astype(float)
            if options["logy"]:
                vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            traces.append(
                ir.Trace(
                    mark="violin",
                    y=vals.tolist(),
                    name=name,
                    style={"ci": color_idx},
                )
            )
            notes.append(
                f"{name}: median={np.median(vals):.3g}, "
                f"IQR={np.percentile(vals, 75) - np.percentile(vals, 25):.3g}"
            )
        panel = ir.Panel(
            traces=traces,
            ylabel=metric,
            yscale="log" if options["logy"] else "linear",
        )

    elif mode == "percentile":
        plist = [float(p) for p in options["percentiles"]]
        p_labels = [f"p{p:g}" for p in plist]
        for color_idx, (name, grp) in enumerate(groups):
            vals = grp[metric].to_numpy().astype(float)
            if len(vals) == 0:
                continue
            p_vals = [float(np.percentile(vals, p)) for p in plist]
            traces.append(
                ir.Trace(
                    mark="bar",
                    x=p_labels,
                    y=p_vals,
                    name=name,
                    style={"ci": color_idx},
                )
            )
        panel = ir.Panel(
            traces=traces,
            xlabel="Percentile",
            ylabel=metric,
            yscale="log" if options["logy"] else "linear",
        )

    if not traces:
        raise spec.NotApplicableError("No valid values after filtering")

    title = (
        "Violin plot"
        if mode == "violin"
        else ("Box plot" if mode == "box" else "Metric distribution")
    )
    return chart_helpers.single(title, panel, notes="; ".join(notes))


@spec.register_plot(
    "violin",
    group="overview",
    title="Violin plot of metric distributions",
    needs="any numeric column from kernels, steps, network_ops, or gpu_metrics",
    aliases=("violinplot", "violin_plot"),
    options=_DIST_BASE_OPTIONS,
)
def violin_plot(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots metric distributions as violin plots."""
    opts = dict(options)
    opts["mode"] = "violin"
    opts.setdefault("bins", 50)
    opts.setdefault("density", True)
    opts.setdefault("percentiles", ["50", "90", "95", "99", "99.9"])
    return metric_dist(ctx, opts)


@spec.register_plot(
    "boxplot",
    group="overview",
    title="Boxplot of metric distributions",
    needs="any numeric column from kernels, steps, network_ops, or gpu_metrics",
    aliases=("box", "box_plot"),
    options=_DIST_BASE_OPTIONS,
)
def box_plot(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots metric distributions as box plots."""
    opts = dict(options)
    opts["mode"] = "box"
    opts.setdefault("bins", 50)
    opts.setdefault("density", True)
    opts.setdefault("percentiles", ["50", "90", "95", "99", "99.9"])
    return metric_dist(ctx, opts)


@spec.register_plot(
    "category_compare",
    group="compare",
    title="Categorical metric comparison",
    needs="categorical data from kernels, steps, network_ops, or report tables",
    aliases=("cat_compare", "categorical_bars"),
    options={
        "source": spec.Opt(
            "str",
            "kernels",
            "Source table.",
            _CAT_SOURCE_CHOICES,
        ),
        "category": spec.Opt(
            "str",
            "op_family",
            "Categorical breakdown column.",
            _CAT_CHOICES,
        ),
        "metric": spec.Opt(
            "str",
            "duration_ms",
            "Metric to aggregate.",
        ),
        "agg": spec.Opt(
            "str",
            "sum",
            "Aggregation operation.",
            _AGG_CHOICES,
        ),
        "top_k": spec.Opt(
            "int",
            10,
            "Number of top categories to display (0 = all).",
        ),
        "sort": spec.Opt(
            "str",
            "desc",
            "Sorting order.",
            _SORT_CHOICES,
        ),
        "normalize": spec.Opt(
            "bool",
            False,
            "Normalize to percentage share per group (100% stacked).",
        ),
        "orientation": spec.Opt(
            "str",
            "v",
            "Bar orientation.",
            _ORIENTATION_CHOICES,
        ),
        "stack": spec.Opt(
            "bool",
            False,
            "Stack bars instead of grouped side-by-side.",
        ),
        "by": spec.Opt(
            "str",
            "label",
            "Group comparison column.",
            _BY_CAT_CHOICES,
        ),
    },
)
def category_compare(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots categorical metric comparisons across categories and groups.

    Supports top-K filtering, sorting, stacked or grouped layouts, and
    horizontal or vertical orientations.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with comparative bars.

    Raises:
        spec.NotApplicableError: The table or required columns are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise spec.NotApplicableError(f"{source} table is empty")

    cat_col = options["category"]
    if cat_col not in df.columns:
        raise spec.NotApplicableError(
            f"Category column '{cat_col}' not found in {source}"
        )

    metric = options["metric"]
    if metric not in df.columns:
        raise spec.NotApplicableError(
            f"Metric column '{metric}' not found in {source}"
        )

    df = df.filter(pl.col(cat_col).is_not_null() & pl.col(metric).is_not_null())
    if df.is_empty():
        raise spec.NotApplicableError("No valid categorical rows")

    agg_name = options["agg"]
    if agg_name == "sum":
        agg_expr = pl.col(metric).sum()
    elif agg_name == "mean":
        agg_expr = pl.col(metric).mean()
    elif agg_name in ("median", "p50"):
        agg_expr = pl.col(metric).median()
    elif agg_name == "p90":
        agg_expr = pl.col(metric).quantile(0.90)
    elif agg_name == "p99":
        agg_expr = pl.col(metric).quantile(0.99)
    else:
        agg_expr = pl.col(metric).count()

    cat_totals = df.group_by(cat_col).agg(pl.col(metric).sum())
    top_k = int(options["top_k"])
    sort_order = options["sort"]
    if sort_order == "desc":
        cat_totals = cat_totals.sort(metric, descending=True)
    elif sort_order == "asc":
        cat_totals = cat_totals.sort(metric, descending=False)

    if top_k > 0:
        cat_totals = cat_totals.head(top_k)

    categories = [str(c) for c in cat_totals[cat_col].to_list()]
    if not categories:
        raise spec.NotApplicableError("No categories found")

    by_col = options["by"]
    groups: list[tuple[str, pl.DataFrame]] = []
    if by_col == "none" or by_col not in df.columns:
        groups.append(("total", df))
    else:
        for val in chart_helpers.unique_in_order(df, by_col):
            groups.append((str(val), df.filter(pl.col(by_col) == val)))

    traces: list[ir.Trace] = []
    orientation = options["orientation"]
    normalize = bool(options["normalize"])

    for color_idx, (name, grp) in enumerate(groups):
        grp_agg = grp.group_by(cat_col).agg(agg_expr)
        val_map = {
            str(row[cat_col]): float(row[metric])
            for row in grp_agg.iter_rows(named=True)
            if row[metric] is not None
        }
        vals = np.array([val_map.get(c, 0.0) for c in categories], dtype=float)
        if normalize:
            tot = float(np.sum(vals))
            if tot > 0:
                vals = (vals / tot) * 100.0

        if orientation == "v":
            traces.append(
                ir.Trace(
                    mark="bar",
                    x=categories,
                    y=vals.tolist(),
                    name=name,
                    style={"ci": color_idx},
                )
            )
        else:
            traces.append(
                ir.Trace(
                    mark="barh",
                    x=vals.tolist(),
                    y=categories,
                    name=name,
                    style={"ci": color_idx},
                )
            )

    val_label = f"{metric} (% share)" if normalize else f"{metric} ({agg_name})"
    if orientation == "v":
        xlabel = cat_col
        ylabel = val_label
    else:
        xlabel = val_label
        ylabel = cat_col

    stacked = bool(options["stack"]) or normalize
    panel = ir.Panel(
        traces=traces,
        xlabel=xlabel,
        ylabel=ylabel,
        stacked=stacked,
    )
    return chart_helpers.single("Categorical metric comparison", panel)


@spec.register_plot(
    "execution_heatmap",
    group="distributed",
    title="Execution heatmap",
    needs="kernels or steps with 2+ ranks or multi-stream execution",
    aliases=("exec_heatmap", "rank_time_heatmap"),
    options={
        "source": spec.Opt(
            "str",
            "kernels",
            "Source table.",
            _SOURCE_CHOICES[:4],
        ),
        "x": spec.Opt(
            "str",
            "step",
            "X axis dimension.",
            _HEATMAP_X_CHOICES,
        ),
        "y": spec.Opt(
            "str",
            "rank",
            "Y axis dimension.",
            _HEATMAP_Y_CHOICES,
        ),
        "metric": spec.Opt(
            "str",
            "duration_ms",
            "Intensity metric.",
        ),
        "bins": spec.Opt(
            "int",
            100,
            "Number of uniform time bins when x='time'.",
        ),
        "cmap": spec.Opt(
            "str",
            "magma",
            "Colormap.",
            _CMAP_CHOICES,
        ),
        "clip_p": spec.Opt(
            "float",
            98.0,
            "Percentile to clip colorbar maximum to avoid washout.",
        ),
        "zscale": spec.Opt(
            "str",
            "linear",
            "Scale for heatmap values.",
            ("linear", "log"),
        ),
        "agg": spec.Opt(
            "str",
            "sum",
            "Aggregation per cell.",
            _HEATMAP_AGG_CHOICES,
        ),
    },
)
def execution_heatmap(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots 2D execution intensity heatmaps over time or steps.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart with one heatmap panel per trial group.

    Raises:
        spec.NotApplicableError: The table or dimensions are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise spec.NotApplicableError(f"{source} table is empty")

    y_col = options["y"]
    if y_col not in df.columns:
        raise spec.NotApplicableError(
            f"Y dimension '{y_col}' not found in {source}"
        )

    metric = options["metric"]
    if metric not in df.columns:
        raise spec.NotApplicableError(
            f"Metric '{metric}' not found in {source}"
        )

    if options["x"] == "step":
        if "idx" in df.columns:
            x_col = "idx"
        elif "step" in df.columns:
            x_col = "step"
        else:
            raise spec.NotApplicableError("Step column not found in table")
    else:
        if "t_ms" in df.columns:
            x_col = "t_ms"
        elif "start_ns" in df.columns:
            x_col = "start_ns"
        elif "elapsed_s" in df.columns:
            x_col = "elapsed_s"
        else:
            raise spec.NotApplicableError("Time column not found in table")

    df = df.filter(
        pl.col(y_col).is_not_null()
        & pl.col(x_col).is_not_null()
        & pl.col(metric).is_not_null()
    )
    if df.is_empty():
        raise spec.NotApplicableError("No valid rows for execution heatmap")

    has_labels = "label" in df.columns and df["label"].n_unique() > 1
    labels = (
        chart_helpers.unique_in_order(df, "label") if has_labels else ["all"]
    )

    panels: list[ir.Panel] = []
    agg_name = options["agg"]
    bins = int(options["bins"])
    zscale = options["zscale"]
    clip_p = float(options["clip_p"])

    for label in labels:
        label_df = df.filter(pl.col("label") == label) if has_labels else df
        if label_df.is_empty():
            continue

        raw_y = sorted(label_df[y_col].unique().to_list())
        if not raw_y:
            continue
        y_names = [
            f"r{v}"
            if y_col == "rank"
            else (f"s{v}" if y_col == "stream" else str(v))
            for v in raw_y
        ]
        y_map = {raw_v: i for i, raw_v in enumerate(raw_y)}

        if options["x"] == "step":
            raw_x = sorted(label_df[x_col].unique().to_list())
            x_names = [str(v) for v in raw_x]
            x_map = {raw_v: j for j, raw_v in enumerate(raw_x)}
            z_grid = np.zeros((len(raw_y), len(raw_x)), dtype=float)

            if agg_name == "sum":
                agg_expr = pl.col(metric).sum()
            elif agg_name == "mean":
                agg_expr = pl.col(metric).mean()
            elif agg_name == "max":
                agg_expr = pl.col(metric).max()
            else:
                agg_expr = pl.col(metric).count()

            agg_df = label_df.group_by(y_col, x_col).agg(agg_expr)
            for row in agg_df.iter_rows(named=True):
                r_val = row[metric]
                if r_val is not None:
                    z_grid[y_map[row[y_col]], x_map[row[x_col]]] = float(r_val)

        else:
            t_arr = label_df[x_col].to_numpy().astype(float)
            t_min, t_max = float(np.min(t_arr)), float(np.max(t_arr))
            if t_max <= t_min:
                t_max = t_min + 1.0
            bin_edges = np.linspace(t_min, t_max, bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            x_names = [f"{bc:.1f}" for bc in bin_centers]
            bin_indices = np.clip(
                np.floor(
                    (t_arr - t_min) / (t_max - t_min + 1e-9) * bins
                ).astype(int),
                0,
                bins - 1,
            )
            binned_df = label_df.with_columns(pl.Series("bin_idx", bin_indices))

            z_grid = np.zeros((len(raw_y), bins), dtype=float)
            if agg_name == "sum":
                agg_expr = pl.col(metric).sum()
            elif agg_name == "mean":
                agg_expr = pl.col(metric).mean()
            elif agg_name == "max":
                agg_expr = pl.col(metric).max()
            else:
                agg_expr = pl.col(metric).count()

            agg_df = binned_df.group_by(y_col, "bin_idx").agg(agg_expr)
            for row in agg_df.iter_rows(named=True):
                r_val = row[metric]
                if r_val is not None:
                    b_idx = int(row["bin_idx"])
                    if 0 <= b_idx < bins:
                        z_grid[y_map[row[y_col]], b_idx] = float(r_val)

        if zscale == "log":
            pos = z_grid > 0
            min_pos = float(np.min(z_grid[pos])) if np.any(pos) else 1e-3
            z_grid = np.where(
                z_grid > 0,
                np.log10(np.maximum(z_grid, min_pos * 0.1)),
                0.0,
            )

        pos_z = z_grid[z_grid > 0]
        vmax = (
            float(np.percentile(pos_z, clip_p))
            if len(pos_z) > 0
            else float(np.max(z_grid) if len(z_grid) > 0 else 1.0)
        )

        trace = ir.Trace(
            mark="heatmap",
            x=x_names,
            y=y_names,
            z=z_grid.tolist(),
            style={
                "cmap": options["cmap"],
                "vmin": 0.0,
                "vmax": vmax,
            },
        )
        panel_title = f"{label} ({metric})" if has_labels else metric
        xlabel = "Step" if options["x"] == "step" else "Time (ms)"
        colorbar = (
            f"{metric} ({agg_name})"
            if zscale == "linear"
            else f"log10({metric})"
        )
        panels.append(
            ir.Panel(
                traces=[trace],
                title=panel_title,
                xlabel=xlabel,
                ylabel=y_col.capitalize(),
                colorbar=colorbar,
            )
        )

    if not panels:
        raise spec.NotApplicableError("No valid heatmap panels")

    return chart_helpers.grid("Execution heatmap", panels, max_cols=2)


@spec.register_plot(
    "metric_correlation",
    group="compare",
    title="Metric correlation",
    needs=(
        "at least 2 numeric columns from kernels, network_ops, steps, or "
        "gpu_metrics"
    ),
    aliases=("correlation", "metric_corr", "corr"),
    options={
        "source": spec.Opt(
            "str",
            "kernels",
            "Source table.",
            _CORR_SOURCE_CHOICES,
        ),
        "x": spec.Opt("str", "t_ms", "X metric column."),
        "y": spec.Opt("str", "duration_ms", "Y metric column."),
        "color": spec.Opt(
            "str",
            "category",
            "Point coloring column.",
            _BY_CORR_CHOICES,
        ),
        "logx": spec.Opt("bool", False, "Log scale on X axis."),
        "logy": spec.Opt("bool", False, "Log scale on Y axis."),
        "trend": spec.Opt(
            "bool",
            True,
            "Compute and draw regression / Theil-Sen trend line.",
        ),
        "annotate_top": spec.Opt(
            "int",
            5,
            "Annotate top N outlier points (0 = off).",
        ),
        "report_stats": spec.Opt(
            "bool",
            True,
            "Include Pearson r and Spearman rho in chart notes.",
        ),
        "max_points": spec.Opt(
            "int",
            5000,
            "Maximum scatter points to plot.",
        ),
    },
)
def metric_correlation(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots metric correlations, scatter points, and trend fits.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with correlation scatter, trendline, and stats.

    Raises:
        spec.NotApplicableError: The table or numeric columns are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise spec.NotApplicableError(f"{source} table is empty")

    x_col = options["x"]
    y_col = options["y"]
    if x_col not in df.columns:
        raise spec.NotApplicableError(f"Column '{x_col}' not found in {source}")
    if y_col not in df.columns:
        raise spec.NotApplicableError(f"Column '{y_col}' not found in {source}")

    df = df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
    if options["logx"]:
        df = df.filter(pl.col(x_col) > 0)
    if options["logy"]:
        df = df.filter(pl.col(y_col) > 0)

    if df.height < 2:
        raise spec.NotApplicableError(
            "Needs at least 2 data points for correlation"
        )

    x_arr = df[x_col].to_numpy().astype(float)
    y_arr = df[y_col].to_numpy().astype(float)

    if np.ptp(x_arr) == 0 or np.ptp(y_arr) == 0:
        notes = (
            f"N = {len(x_arr)} (constant values detected; correlation "
            "undefined)"
        )
    else:
        r_val, r_p = sp_stats.pearsonr(x_arr, y_arr)
        rho_val, rho_p = sp_stats.spearmanr(x_arr, y_arr)
        notes = (
            f"Pearson r = {r_val:.3f} (p={r_p:.2e}), "
            f"Spearman rho = {rho_val:.3f} (p={rho_p:.2e}), N = {len(x_arr)}"
        )

    max_points = int(options["max_points"])
    if 0 < max_points < df.height:
        df = df.sample(n=max_points, seed=42)
        x_arr = df[x_col].to_numpy().astype(float)
        y_arr = df[y_col].to_numpy().astype(float)

    color_col = options["color"]
    groups: list[tuple[str, pl.DataFrame]] = []
    if color_col == "none" or color_col not in df.columns:
        groups.append(("points", df))
    else:
        for val in chart_helpers.unique_in_order(df, color_col):
            groups.append((str(val), df.filter(pl.col(color_col) == val)))

    traces: list[ir.Trace] = []
    for color_idx, (name, grp) in enumerate(groups):
        gx = grp[x_col].to_numpy().astype(float).tolist()
        gy = grp[y_col].to_numpy().astype(float).tolist()
        gtext = (
            [str(n)[:32] for n in grp["name"].to_list()]
            if "name" in grp.columns
            else None
        )
        traces.append(
            ir.Trace(
                mark="scatter",
                x=gx,
                y=gy,
                name=name,
                text=gtext,
                style={"ci": color_idx, "s": 24, "alpha": 0.7},
            )
        )

    trend_drawn = False
    slope = 0.0
    intercept = 0.0
    if (
        options["trend"]
        and len(x_arr) >= 4
        and np.ptp(x_arr) > 0
        and np.ptp(y_arr) > 0
    ):
        slope, intercept, _, _ = sp_stats.theilslopes(y_arr, x_arr)
        x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
        x_trend = np.linspace(x_min, x_max, 50)
        y_trend = slope * x_trend + intercept
        traces.append(
            ir.Trace(
                mark="line",
                x=x_trend.tolist(),
                y=y_trend.tolist(),
                name=f"Theil-Sen trend (slope={slope:.2g})",
                color=theme.NEUTRAL,
                style={"dash": "--", "width": 1.5},
            )
        )
        trend_drawn = True

    annotate_top = int(options["annotate_top"])
    if annotate_top > 0 and "name" in df.columns and len(x_arr) > 0:
        if trend_drawn:
            residuals = np.abs(y_arr - (slope * x_arr + intercept))
        else:
            residuals = y_arr
        top_k_ann = min(annotate_top, len(residuals))
        top_indices = np.argsort(residuals)[-top_k_ann:]
        ox = [float(x_arr[i]) for i in top_indices]
        oy = [float(y_arr[i]) for i in top_indices]
        names_list = df["name"].to_list()
        otext = [str(names_list[int(i)])[:20] for i in top_indices]
        traces.append(
            ir.Trace(
                mark="text",
                x=ox,
                y=oy,
                text=otext,
                color=theme.BAD,
            )
        )

    panel = ir.Panel(
        traces=traces,
        xlabel=x_col,
        ylabel=y_col,
        xscale="log" if options["logx"] else "linear",
        yscale="log" if options["logy"] else "linear",
    )
    return chart_helpers.single(
        "Metric correlation",
        panel,
        notes=notes if options["report_stats"] else "",
    )
