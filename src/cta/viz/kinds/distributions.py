"""Plot kinds of how a raw-table metric is distributed."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers

_DIST_SOURCE_CHOICES = (
    "kernels",
    "steps",
    "network_ops",
    "gpu_metrics",
    "jtop",
)


_BY_DIST_CHOICES = (
    "label",
    "category",
    "op_family",
    "rank",
    "device",
    "none",
)


_DIST_MODES = ("hist", "ecdf", "box", "percentile", "violin")


_DIST_BASE_OPTIONS = {
    "source": registry.Opt(
        "str",
        "kernels",
        "Source table.",
        _DIST_SOURCE_CHOICES,
    ),
    "metric": registry.Opt(
        "str",
        "duration_ms",
        "Numeric column for distribution analysis.",
    ),
    "by": registry.Opt(
        "str",
        "label",
        "Grouping column.",
        _BY_DIST_CHOICES,
    ),
    "logx": registry.Opt(
        "bool",
        False,
        "Log-scale for metric values / X axis.",
    ),
    "logy": registry.Opt(
        "bool",
        False,
        "Log-scale for counts/density / Y axis.",
    ),
}


@registry.register_plot(
    "metric_dist",
    group="overview",
    title="Metric distribution",
    needs="any numeric column from kernels, steps, network_ops, or gpu_metrics",
    aliases=("dist", "distribution", "distributions"),
    options={
        **_DIST_BASE_OPTIONS,
        "mode": registry.Opt(
            "str",
            "hist",
            "Distribution presentation format.",
            _DIST_MODES,
        ),
        "bins": registry.Opt("int", 50, "Histogram bin count."),
        "density": registry.Opt(
            "bool",
            True,
            "Normalize histogram to probability density.",
        ),
        "percentiles": registry.Opt(
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
        NotApplicableError: The table or numeric column is missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise registry.NotApplicableError(f"{source} table is empty")

    metric = options["metric"]
    if metric not in df.columns:
        raise registry.NotApplicableError(
            f"Metric '{metric}' not found in {source}"
        )

    df = df.filter(pl.col(metric).is_not_null())
    if df.is_empty():
        raise registry.NotApplicableError("No valid numeric values")

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
        raise registry.NotApplicableError("No valid values after filtering")

    title = (
        "Violin plot"
        if mode == "violin"
        else ("Box plot" if mode == "box" else "Metric distribution")
    )
    return chart_helpers.single(title, panel, notes="; ".join(notes))


@registry.register_plot(
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


@registry.register_plot(
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


# The most points a density estimate reads per series.
_KDE_POINTS = 20_000


def _density(
    values: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Evaluates a Gaussian kernel density estimate on a grid."""
    if len(values) > _KDE_POINTS:
        values = values[:: len(values) // _KDE_POINTS + 1]
    estimate = np.zeros(len(grid))
    for chunk in np.array_split(values, max(1, len(values) // 2_000)):
        scaled = (grid[:, None] - chunk[None, :]) / bandwidth
        estimate += np.exp(-0.5 * scaled * scaled).sum(axis=1)
    return estimate / (len(values) * bandwidth * np.sqrt(2 * np.pi))


@registry.register_plot(
    "ridgeline",
    group="overview",
    title="Stacked distributions",
    needs="a raw table with a numeric column",
    aliases=("joyplot", "ridges"),
    options={
        "source": registry.Opt(
            "str",
            "kernels",
            "Source table.",
            ("kernels", "steps", "network_ops", "runtime_calls", "gpu_metrics"),
        ),
        "metric": registry.Opt("str", "duration_ms", "Numeric column."),
        "by": registry.Opt(
            "str",
            "label",
            "One curve per value of this column.",
            ("label", "rank", "device", "stream", "category", "idx"),
        ),
        "max_series": registry.Opt("int", 12, "Most curves (largest first)."),
        "overlap": registry.Opt(
            "float", 1.8, "How many curve heights a ridge may rise."
        ),
        "logx": registry.Opt("bool", False, "Log-scale value axis."),
    },
)
def ridgeline(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Stacks one density curve per rank, group, step or other column value.

    All curves share one value axis, so a shift or a second peak shows as a
    sideways move between neighbouring ridges.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One panel with filled curves, the first at the top.

    Raises:
        NotApplicableError: The table lacks a column or has no values.
    """
    table = ctx.frame(options["source"], options)
    metric, by = options["metric"], options["by"]
    for column in (metric, by):
        if column not in table.columns:
            raise registry.NotApplicableError(
                f"the {options['source']} table has no '{column}'"
            )
    table = table.filter(pl.col(metric).is_finite())
    if options["logx"]:
        table = table.filter(pl.col(metric) > 0)
    if table.is_empty():
        raise registry.NotApplicableError("no finite values to estimate")
    series = sorted(
        table.group_by(by).len().iter_rows(), key=lambda row: -row[1]
    )[: max(1, options["max_series"])]
    keys = sorted(key for key, _ in series)
    samples = {
        key: table.filter(pl.col(by) == key)[metric].to_numpy().astype(float)
        for key in keys
    }
    if options["logx"]:
        samples = {key: np.log10(values) for key, values in samples.items()}
    everything = np.concatenate(list(samples.values()))
    low, high = float(everything.min()), float(everything.max())
    pad = (high - low) * 0.05 or 1.0
    grid = np.linspace(low - pad, high + pad, 240)
    densities = {}
    for key, values in samples.items():
        spread = float(np.std(values))
        # Silverman's rule; a constant series still gets a visible spike.
        bandwidth = (
            1.06 * spread * len(values) ** -0.2 if spread > 0 else pad / 5
        )
        densities[key] = _density(values, grid, max(bandwidth, pad / 100))
    peak = max(float(d.max()) for d in densities.values()) or 1.0
    scale = float(options["overlap"]) / peak
    x = (10**grid if options["logx"] else grid).tolist()
    traces: list[ir.Trace] = []
    for position, key in enumerate(keys):
        offset = float(len(keys) - 1 - position)
        top = (offset + densities[key] * scale).tolist()
        traces.append(
            ir.Trace(
                "band",
                x=x,
                y=[offset] * len(x),
                y1=top,
                style={"ci": position, "alpha": 0.55},
                meta={"label": str(key)} if by == "label" else {},
            )
        )
        traces.append(ir.Trace("line", x=x, y=top, style={"ci": position}))
        traces.append(
            ir.Trace(
                "text",
                x=[x[0]],
                y=[offset + 0.15],
                text=[str(key)],
                style={"ci": position},
            )
        )
    panel = ir.Panel(
        traces=traces,
        xlabel=metric,
        ylabel=by,
        xscale="log" if options["logx"] else "linear",
        legend=False,
    )
    return chart_helpers.single(
        "Stacked distributions", panel, meta=chart_helpers.SKIP_SUITABILITY
    )
