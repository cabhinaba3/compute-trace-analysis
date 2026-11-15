"""General-purpose charts: radar, ridgeline, parallel coordinates, facets."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

_SKIP = {"skip_suitability": True}
# The most points a density estimate reads per series.
_KDE_POINTS = 20_000


def _summary_rows(
    ctx: data.DataContext, aspect: str, match: str, count: int
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
        spec.NotApplicableError: There are fewer than two groups, or no
            metric has values in all of them.
    """
    import re

    report = ctx.require_report()
    if len(report.groups) < 2:
        raise spec.NotApplicableError("needs two or more groups")
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
        raise spec.NotApplicableError("no metric has a value in every group")

    def spread(by_label: dict[str, float]) -> float:
        base = by_label[ctx.labels[0]]
        scale = abs(base) if abs(base) > 1e-12 else 1.0
        return (max(by_label.values()) - min(by_label.values())) / scale

    ranked = sorted(values, key=lambda metric: -spread(values[metric]))
    return (ranked[:count] if count > 0 else ranked), values


@spec.register_plot(
    "metric_radar",
    group="compare",
    title="Metrics of every group against the baseline, on a radar",
    needs="a report with two or more groups",
    aliases=("radar", "spider"),
    options={
        "aspect": spec.Opt("str", "", "Only this aspect (default: all)."),
        "match": spec.Opt(
            "str", "", "Only metrics whose name matches this expression."
        ),
        "max_axes": spec.Opt(
            "int", 8, "Axes to draw: the metrics that differ most."
        ),
    },
)
def metric_radar(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
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
        spec.NotApplicableError: Fewer than two groups, or fewer than three
            metrics to span a polygon.
    """
    metrics, values = _summary_rows(
        ctx, options["aspect"], options["match"], options["max_axes"]
    )
    if len(metrics) < 3:
        raise spec.NotApplicableError("a radar needs at least three metrics")
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
        meta=_SKIP,
    )


@spec.register_plot(
    "parallel_coords",
    group="compare",
    title="Every group as a line across its metrics",
    needs="a report with two or more groups",
    aliases=("parallel", "parcoords"),
    options={
        "aspect": spec.Opt("str", "", "Only this aspect (default: all)."),
        "match": spec.Opt(
            "str", "", "Only metrics whose name matches this expression."
        ),
        "max_axes": spec.Opt(
            "int", 10, "Metrics to draw: the ones that differ most."
        ),
    },
)
def parallel_coords(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Draws one line per group through its metrics, each axis scaled 0 to 1.

    Each metric is scaled to the range its groups span, so lines that cross
    between two axes show metrics that trade off against each other.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One panel with a line per group.

    Raises:
        spec.NotApplicableError: Fewer than two groups or no shared metric.
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
        meta=_SKIP,
    )


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


@spec.register_plot(
    "ridgeline",
    group="overview",
    title="Stacked distributions",
    needs="a raw table with a numeric column",
    aliases=("joyplot", "ridges"),
    options={
        "source": spec.Opt(
            "str",
            "kernels",
            "Source table.",
            ("kernels", "steps", "network_ops", "runtime_calls", "gpu_metrics"),
        ),
        "metric": spec.Opt("str", "duration_ms", "Numeric column."),
        "by": spec.Opt(
            "str",
            "label",
            "One curve per value of this column.",
            ("label", "rank", "device", "stream", "category", "idx"),
        ),
        "max_series": spec.Opt("int", 12, "Most curves (largest first)."),
        "overlap": spec.Opt(
            "float", 1.8, "How many curve heights a ridge may rise."
        ),
        "logx": spec.Opt("bool", False, "Log-scale value axis."),
    },
)
def ridgeline(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Stacks one density curve per rank, group, step or other column value.

    All curves share one value axis, so a shift or a second peak shows as a
    sideways move between neighbouring ridges.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        One panel with filled curves, the first at the top.

    Raises:
        spec.NotApplicableError: The table lacks a column or has no values.
    """
    table = ctx.frame(options["source"], options)
    metric, by = options["metric"], options["by"]
    for column in (metric, by):
        if column not in table.columns:
            raise spec.NotApplicableError(
                f"the {options['source']} table has no '{column}'"
            )
    table = table.filter(pl.col(metric).is_finite())
    if options["logx"]:
        table = table.filter(pl.col(metric) > 0)
    if table.is_empty():
        raise spec.NotApplicableError("no finite values to estimate")
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
    return chart_helpers.single("Stacked distributions", panel, meta=_SKIP)


_FACETABLE = (
    "metric_trace",
    "metric_dist",
    "violin",
    "boxplot",
    "category_compare",
    "queue_depth",
    "memcpy_bandwidth",
    "memcpy_sizes",
    "idle_gaps",
    "ridgeline",
    "sync_waits",
    "runtime_top",
    "network_bandwidth",
    "kernel_scatter",
)


def _parse_inner(text: str) -> dict[str, str]:
    """Reads ``key=value,key=value`` into a dict of strings."""
    pairs = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError(f"inner option '{part}' needs key=value")
        pairs[key.strip()] = value.strip()
    return pairs


@spec.register_plot(
    "facet",
    group="overview",
    title="Any raw-table plot, one panel per rank, device or group",
    needs="a raw-table plot kind and a report",
    aliases=("small_multiples", "facets"),
    options={
        "kind": spec.Opt(
            "str",
            "metric_dist",
            "Plot kind to repeat.",
            _FACETABLE,
        ),
        "by": spec.Opt(
            "str",
            "rank",
            "What to split on.",
            ("rank", "device", "label"),
        ),
        "inner": spec.Opt(
            "str",
            "",
            "Options of the repeated kind as key=value,key=value.",
        ),
        "max_panels": spec.Opt("int", 8, "Most panels to draw."),
        "columns": spec.Opt("int", 4, "Panels per row."),
    },
)
def facet(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Repeats a plot kind once per rank, device or group.

    Each panel is the repeated kind drawn from only that slice of the data,
    so the panels are directly comparable. Only the first panel of a kind
    that draws several is kept.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A grid with one panel per value.

    Raises:
        spec.NotApplicableError: The split column has no value, or the
            repeated kind cannot be drawn from any slice.
        ValueError: ``inner`` is malformed or names an option the repeated
            kind lacks.
    """
    inner_kind = spec.get_kind(options["kind"])
    kernels = ctx.frame("kernels", options)
    by = options["by"]
    column = {"rank": "rank", "device": "device", "label": "label"}[by]
    values = sorted(kernels[column].unique().to_list())[
        : max(1, options["max_panels"])
    ]
    if not values:
        raise spec.NotApplicableError(f"the trace has no {by} values")
    given = _parse_inner(options["inner"])
    shared = {
        key: options[key] for key in ("labels", "rank", "device", "steps")
    }
    panels, skipped = [], []
    for value in values:
        narrowed = {
            **{k: v for k, v in shared.items() if v is not None},
            **given,
            {"rank": "rank", "device": "device", "label": "labels"}[by]: (
                [value]
            ),
        }
        resolved = spec.resolve_options(inner_kind, narrowed)
        try:
            chart = inner_kind.build(ctx, resolved)
        except spec.NotApplicableError as error:
            skipped.append(f"{by} {value}: {error}")
            continue
        panel = chart.panels[0]
        panel.title = f"{by} {value}"
        panels.append(panel)
    if not panels:
        raise spec.NotApplicableError(
            f"{inner_kind.name} cannot be drawn from any {by}: "
            + "; ".join(skipped)
        )
    return chart_helpers.grid(
        f"{inner_kind.name} by {by}",
        panels,
        max_cols=max(1, options["columns"]),
        notes="; ".join(skipped),
        meta=_SKIP,
    )
