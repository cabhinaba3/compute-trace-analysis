"""Plot kinds that compare groups or categories on several metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz.kinds import chart_helpers

_CAT_SOURCE_CHOICES = (
    "kernels",
    "steps",
    "network_ops",
    "runtime",
    "osrt",
)


_BY_CAT_CHOICES = ("label", "rank", "device", "trial", "none")


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


@registry.register_plot(
    "category_compare",
    group="compare",
    title="Categorical metric comparison",
    needs="categorical data from kernels, steps, network_ops, or report tables",
    aliases=("cat_compare", "categorical_bars"),
    options={
        "source": registry.Opt(
            "str",
            "kernels",
            "Source table.",
            _CAT_SOURCE_CHOICES,
        ),
        "category": registry.Opt(
            "str",
            "op_family",
            "Categorical breakdown column.",
            _CAT_CHOICES,
        ),
        "metric": registry.Opt(
            "str",
            "duration_ms",
            "Metric to aggregate.",
        ),
        "agg": registry.Opt(
            "str",
            "sum",
            "Aggregation operation.",
            _AGG_CHOICES,
        ),
        "top_k": registry.Opt(
            "int",
            10,
            "Number of top categories to display (0 = all).",
        ),
        "sort": registry.Opt(
            "str",
            "desc",
            "Sorting order.",
            _SORT_CHOICES,
        ),
        "normalize": registry.Opt(
            "bool",
            False,
            "Normalize to percentage share per group (100% stacked).",
        ),
        "orientation": registry.Opt(
            "str",
            "v",
            "Bar orientation.",
            _ORIENTATION_CHOICES,
        ),
        "stack": registry.Opt(
            "bool",
            False,
            "Stack bars instead of grouped side-by-side.",
        ),
        "by": registry.Opt(
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
        NotApplicableError: The table or required columns are missing.
    """
    source = options["source"]
    df = ctx.frame(source, options)
    if df.is_empty():
        raise registry.NotApplicableError(f"{source} table is empty")

    cat_col = options["category"]
    if cat_col not in df.columns:
        raise registry.NotApplicableError(
            f"Category column '{cat_col}' not found in {source}"
        )

    metric = options["metric"]
    if metric not in df.columns:
        raise registry.NotApplicableError(
            f"Metric column '{metric}' not found in {source}"
        )

    df = df.filter(pl.col(cat_col).is_not_null() & pl.col(metric).is_not_null())
    if df.is_empty():
        raise registry.NotApplicableError("No valid categorical rows")

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
        raise registry.NotApplicableError("No categories found")

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


def _summary_rows(
    ctx: context.DataContext, aspect: str, match: str, count: int
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Picks the metrics that differ most between groups, with their means.

    Args:
        ctx: The report being plotted.
        aspect: One aspect to read, or "" for all of them.
        match: Regular expression a metric name must contain; "" keeps all.
        count: How many metrics to keep (0 = all).

    Returns:
        The kept metric names, most different first, and each one's mean in
        each group.

    Raises:
        NotApplicableError: There are fewer than two groups, or no
            metric has values in all of them.
    """
    import re

    report = ctx.require_report()
    if len(report.groups) < 2:
        raise registry.NotApplicableError("needs two or more groups")
    values: dict[str, dict[str, float]] = {}
    for name in [aspect] if aspect else list(report.aspects):
        table = ctx.report_table(name)
        for metric, by_label in chart_helpers.summary_values(
            table, ctx.labels
        ).items():
            if len(by_label) == len(ctx.labels) and (
                not match or re.search(match, metric)
            ):
                values[f"{name}: {metric}" if not aspect else metric] = by_label
    if not values:
        raise registry.NotApplicableError(
            "no metric has a value in every group"
        )

    def spread(by_label: dict[str, float]) -> float:
        base = by_label[ctx.labels[0]]
        scale = abs(base) if abs(base) > 1e-12 else 1.0
        return (max(by_label.values()) - min(by_label.values())) / scale

    ranked = sorted(values, key=lambda metric: -spread(values[metric]))
    return (ranked[:count] if count > 0 else ranked), values


@registry.register_plot(
    "metric_radar",
    group="compare",
    title="Metrics of every group against the baseline, on a radar",
    needs="a report with two or more groups",
    aliases=("radar", "spider"),
    options={
        "aspect": registry.Opt("str", "", "Only this aspect (default: all)."),
        "match": registry.Opt(
            "str", "", "Only metrics whose name matches this expression."
        ),
        "max_axes": registry.Opt(
            "int", 8, "Axes to draw: the metrics that differ most."
        ),
    },
)
def metric_radar(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Draws each group as a polygon of its metrics relative to the baseline.

    Every axis is a metric and the radius is the group's mean divided by the
    baseline's, so the baseline is a circle at 1 and a larger radius means a
    larger value, which is better for some metrics and worse for others.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One polar panel with a polygon per group.

    Raises:
        NotApplicableError: Fewer than two groups, or fewer than three
            metrics to span a polygon.
    """
    metrics, values = _summary_rows(
        ctx, options["aspect"], options["match"], options["max_axes"]
    )
    if len(metrics) < 3:
        raise registry.NotApplicableError(
            "a radar needs at least three metrics"
        )
    baseline = ctx.labels[0]
    ratios = {
        metric: {
            label: (
                values[metric][label] / values[metric][baseline]
                if abs(values[metric][baseline]) > 1e-12
                else 1.0
            )
            for label in ctx.labels
        }
        for metric in metrics
    }
    top = max(max(by_label.values()) for by_label in ratios.values())
    traces = [
        ir.Trace(
            "radar",
            x=metrics,
            y=[ratios[metric][label] for metric in metrics],
            name=label,
            style={"ci": index, "rmax": max(1.2, 1.1 * top)},
            meta={"label": label},
        )
        for index, label in enumerate(ctx.labels)
    ]
    panel = ir.Panel(traces=traces, polar=True)
    return chart_helpers.single(
        "Metrics against the baseline",
        panel,
        notes=f"Radius is the group's mean over {baseline}'s mean.",
        meta=chart_helpers.SKIP_SUITABILITY,
    )


@registry.register_plot(
    "parallel_coords",
    group="compare",
    title="Every group as a line across its metrics",
    needs="a report with two or more groups",
    aliases=("parallel", "parcoords"),
    options={
        "aspect": registry.Opt("str", "", "Only this aspect (default: all)."),
        "match": registry.Opt(
            "str", "", "Only metrics whose name matches this expression."
        ),
        "max_axes": registry.Opt(
            "int", 10, "Metrics to draw: the ones that differ most."
        ),
    },
)
def parallel_coords(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Draws one line per group through its metrics, each axis scaled 0 to 1.

    Each metric is scaled to the range its groups span, so lines that cross
    between two axes show metrics that trade off against each other.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One panel with a line per group.

    Raises:
        NotApplicableError: Fewer than two groups or no shared metric.
    """
    metrics, values = _summary_rows(
        ctx, options["aspect"], options["match"], options["max_axes"]
    )
    low = {m: min(values[m].values()) for m in metrics}
    high = {m: max(values[m].values()) for m in metrics}

    def scaled(metric: str, label: str) -> float:
        span = high[metric] - low[metric]
        return (values[metric][label] - low[metric]) / span if span else 0.5

    traces = [
        ir.Trace(
            "line",
            x=metrics,
            y=[scaled(metric, label) for metric in metrics],
            name=label,
            style={"ci": index, "marker": "o", "width": 1.6},
            meta={"label": label},
        )
        for index, label in enumerate(ctx.labels)
    ]
    return chart_helpers.single(
        "Groups across metrics",
        ir.Panel(
            traces=traces,
            xlabel="metric",
            ylabel="scaled to the groups' range (0 to 1)",
        ),
        notes="Each metric runs from its lowest to its highest group mean.",
        meta=chart_helpers.SKIP_SUITABILITY,
    )
