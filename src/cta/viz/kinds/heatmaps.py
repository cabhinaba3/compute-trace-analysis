"""Plot kinds that draw metrics as matrices: heatmaps, correlations."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz import theme
from cta.viz.kinds import chart_helpers

_CORR_SOURCE_CHOICES = (
    "kernels",
    "network_ops",
    "steps",
    "gpu_metrics",
    "ncu_kernels",
)


_BY_CORR_CHOICES = (
    "category",
    "op_family",
    "rank",
    "device",
    "stream",
    "label",
    "none",
)


_HEATMAP_X_CHOICES = ("step", "time")


_HEATMAP_Y_CHOICES = ("rank", "device", "stream", "category", "op_family")


_HEATMAP_AGG_CHOICES = ("sum", "mean", "max", "count")


_CMAP_CHOICES = ("magma", "viridis", "plasma", "cividis", "Blues", "YlOrRd")


@registry.register_plot(
    "execution_heatmap",
    group="distributed",
    title="Execution heatmap",
    needs="kernels or steps with 2+ ranks or multi-stream execution",
    aliases=("exec_heatmap", "rank_time_heatmap"),
    options={
        "source": registry.Opt(
            "str",
            "kernels",
            "Source table.",
            chart_helpers.TIMED_SOURCES[:4],
        ),
        "x": registry.Opt(
            "str",
            "step",
            "X axis dimension.",
            _HEATMAP_X_CHOICES,
        ),
        "y": registry.Opt(
            "str",
            "rank",
            "Y axis dimension.",
            _HEATMAP_Y_CHOICES,
        ),
        "metric": registry.Opt(
            "str",
            "duration_ms",
            "Intensity metric.",
        ),
        "bins": registry.Opt(
            "int",
            100,
            "Number of uniform time bins when x='time'.",
        ),
        "cmap": registry.Opt(
            "str",
            "magma",
            "Colormap.",
            _CMAP_CHOICES,
        ),
        "clip_p": registry.Opt(
            "float",
            98.0,
            "Percentile to clip colorbar maximum to avoid washout.",
        ),
        "zscale": registry.Opt(
            "str",
            "linear",
            "Scale for heatmap values.",
            ("linear", "log"),
        ),
        "agg": registry.Opt(
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
        NotApplicableError: The table or dimensions are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise registry.NotApplicableError(f"{source} table is empty")

    y_col = options["y"]
    if y_col not in df.columns:
        raise registry.NotApplicableError(
            f"Y dimension '{y_col}' not found in {source}"
        )

    metric = options["metric"]
    if metric not in df.columns:
        raise registry.NotApplicableError(
            f"Metric '{metric}' not found in {source}"
        )

    if options["x"] == "step":
        if "idx" in df.columns:
            x_col = "idx"
        elif "step" in df.columns:
            x_col = "step"
        else:
            raise registry.NotApplicableError("Step column not found in table")
    else:
        if "t_ms" in df.columns:
            x_col = "t_ms"
        elif "start_ns" in df.columns:
            x_col = "start_ns"
        elif "elapsed_s" in df.columns:
            x_col = "elapsed_s"
        else:
            raise registry.NotApplicableError("Time column not found in table")

    df = df.filter(
        pl.col(y_col).is_not_null()
        & pl.col(x_col).is_not_null()
        & pl.col(metric).is_not_null()
    )
    if df.is_empty():
        raise registry.NotApplicableError("No valid rows for execution heatmap")

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
        raise registry.NotApplicableError("No valid heatmap panels")

    return chart_helpers.grid("Execution heatmap", panels, max_cols=2)


@registry.register_plot(
    "metric_correlation",
    group="compare",
    title="Metric correlation",
    needs=(
        "at least 2 numeric columns from kernels, network_ops, steps, or "
        "gpu_metrics"
    ),
    aliases=("correlation", "metric_corr", "corr"),
    options={
        "source": registry.Opt(
            "str",
            "kernels",
            "Source table.",
            _CORR_SOURCE_CHOICES,
        ),
        "x": registry.Opt("str", "t_ms", "X metric column."),
        "y": registry.Opt("str", "duration_ms", "Y metric column."),
        "color": registry.Opt(
            "str",
            "category",
            "Point coloring column.",
            _BY_CORR_CHOICES,
        ),
        "logx": registry.Opt("bool", False, "Log scale on X axis."),
        "logy": registry.Opt("bool", False, "Log scale on Y axis."),
        "trend": registry.Opt(
            "bool",
            True,
            "Compute and draw regression / Theil-Sen trend line.",
        ),
        "annotate_top": registry.Opt(
            "int",
            5,
            "Annotate top N outlier points (0 = off).",
        ),
        "report_stats": registry.Opt(
            "bool",
            True,
            "Include Pearson r and Spearman rho in chart notes.",
        ),
        "max_points": registry.Opt(
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
        NotApplicableError: The table or numeric columns are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise registry.NotApplicableError(f"{source} table is empty")

    x_col = options["x"]
    y_col = options["y"]
    if x_col not in df.columns:
        raise registry.NotApplicableError(
            f"Column '{x_col}' not found in {source}"
        )
    if y_col not in df.columns:
        raise registry.NotApplicableError(
            f"Column '{y_col}' not found in {source}"
        )

    df = df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
    if options["logx"]:
        df = df.filter(pl.col(x_col) > 0)
    if options["logy"]:
        df = df.filter(pl.col(y_col) > 0)

    if df.height < 2:
        raise registry.NotApplicableError(
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
