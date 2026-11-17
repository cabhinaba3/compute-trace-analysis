"""Plot kinds across many groups: scaling, parameter sweeps, energy Pareto."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz import theme
from cta.viz.kinds import chart_helpers

if TYPE_CHECKING:
    pass


# Summary metrics tried, in order, when no metric is requested.
_PREFERRED_METRICS = (
    "mean_step_ms",
    "critical_path_us",
    "throughput_per_s",
    "energy_per_query_mj",
    "total_energy_j",
)


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _find_metric(
    ctx: context.DataContext, metric: str | None
) -> tuple[str, str, str]:
    """Finds the aspect and unit of a summary metric.

    Args:
        ctx: Data access for the report being plotted.
        metric: The metric to find; None picks the first of
            `_PREFERRED_METRICS` present, else the first metric of the first
            aspect with a summary.

    Returns:
        The aspect name, the metric name and its unit.

    Raises:
        NotApplicableError: The requested metric is in no summary, or the
            report has no summary metrics at all.
    """
    report = ctx.require_report()
    candidates = [metric] if metric else list(_PREFERRED_METRICS)
    for candidate in candidates:
        for aspect_name, aspect in report.aspects.items():
            if (
                not aspect.summary.is_empty()
                and candidate in aspect.summary["metric"].to_list()
            ):
                unit = aspect.summary.filter(pl.col("metric") == candidate)[
                    "unit"
                ][0]
                return aspect_name, candidate, str(unit)
    if metric:
        raise registry.NotApplicableError(
            f"metric '{metric}' not found in any aspect summary"
        )
    first_with_summary = next(
        (
            aspect
            for aspect in report.aspects.values()
            if not aspect.summary.is_empty()
        ),
        None,
    )
    if first_with_summary is None:
        raise registry.NotApplicableError("No summary metrics available")
    return (
        first_with_summary.name,
        str(first_with_summary.summary["metric"][0]),
        str(first_with_summary.summary["unit"][0]),
    )


@registry.register_plot(
    "scaling",
    group="sweep",
    title="Metric across groups",
    exported=True,
    options={"metric": registry.Opt("str", None, "Summary metric name.")},
)
def scaling(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots one summary metric across the groups.

    A bar for one group, a line for several.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart.

    Raises:
        NotApplicableError: The metric cannot be found (see
            `_find_metric`).
    """
    aspect, metric, unit = _find_metric(ctx, options.get("metric"))
    value_by_label = ctx.metric(aspect, metric)
    values = [value_by_label.get(label, 0.0) for label in ctx.labels]
    is_single_group = len(values) == 1
    trace = ir.Trace(
        "bar" if is_single_group else "line",
        x=ctx.labels,
        y=values,
        name=metric,
        style={"ci": 0, "marker": "o", "nolegend": True},
    )
    if is_single_group:
        title = f"Metric summary: {metric}"
    else:
        title = f"Scaling comparison: {metric}"
    return chart_helpers.single(
        title,
        ir.Panel(
            traces=[trace],
            ylabel=f"{metric} ({unit})" if unit else metric,
            legend=False,
        ),
    )


@registry.register_plot(
    "sweep",
    group="sweep",
    title="Metric vs configuration",
    needs="2+ groups",
    options={
        "metric": registry.Opt(
            "list", None, "Summary metric(s); default the first preferred one."
        ),
        "normalize": registry.Opt(
            "str",
            "none",
            "none, first (value / first group) or speedup (first / value).",
            ("none", "first", "speedup"),
        ),
        "x": registry.Opt(
            "str",
            "auto",
            "auto (a number in the label if every label has one, else the"
            " group index), index or label.",
            ("auto", "index", "label"),
        ),
        "logx": registry.Opt("bool", False, "Log x axis."),
        "logy": registry.Opt("bool", False, "Log y axis."),
    },
)
def sweep(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots one or more metrics across ordered groups, optionally normalised.

    Suits a sweep over batch size, GPUs, precision and the like; x is the
    number in each label when every label has a distinct one.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one line per metric.

    Raises:
        NotApplicableError: A metric cannot be found (see `_find_metric`).
    """
    labels = ctx.labels
    metrics = options.get("metric") or [None]
    number_matches = [_NUMBER_PATTERN.search(label) for label in labels]
    label_numbers = [float(match.group()) for match in number_matches if match]
    is_numeric_x = (
        options["x"] == "auto"
        and all(number_matches)
        and len(set(label_numbers)) == len(labels)
    )
    x_values: list[Any]
    if is_numeric_x:
        x_values = label_numbers
    elif options["x"] == "label":
        x_values = labels
    else:
        x_values = list(range(len(labels)))
    normalize = options["normalize"]
    traces = []
    for i, requested_metric in enumerate(metrics):
        aspect, metric, unit = _find_metric(ctx, requested_metric)
        value_by_label = ctx.metric(aspect, metric)
        y = [value_by_label.get(label, float("nan")) for label in labels]
        if normalize == "first" and y[0]:
            y = [value / y[0] for value in y]
        elif normalize == "speedup" and y[0]:
            y = [y[0] / value if value else float("nan") for value in y]
        unit_suffix = f" ({unit})" if unit and normalize == "none" else ""
        traces.append(
            ir.Trace(
                "line",
                x=x_values,
                y=y,
                name=metric + unit_suffix,
                style={"ci": i, "marker": "o"},
            )
        )
    if normalize != "none":
        traces.append(
            ir.Trace(
                "hline", y=[1.0], color=theme.NEUTRAL, style={"nolegend": True}
            )
        )
    ylabel = {
        "none": "value",
        "first": "relative to first group",
        "speedup": "speedup vs first group",
    }[normalize]
    if is_numeric_x:
        xlabel = "value from group label"
    elif options["x"] == "label":
        xlabel = "configuration"
    else:
        xlabel = (
            "group index ("
            + ", ".join(f"{i}={label}" for i, label in enumerate(labels))
            + ")"
        )
    panel = ir.Panel(
        traces=traces,
        xlabel=xlabel,
        ylabel=ylabel,
        xscale="log" if options["logx"] and is_numeric_x else "linear",
        yscale="log" if options["logy"] else "linear",
    )
    return chart_helpers.single("Metric vs configuration", panel)


@registry.register_plot(
    "energy_pareto",
    group="edge",
    title="Latency vs energy",
    needs="a latency and an energy metric (edge/jtop bundles)",
    options={
        "x": registry.Opt("str", "mean_step_ms", "Latency-like metric."),
        "y": registry.Opt("str", "energy_per_query_mj", "Energy-like metric."),
    },
)
def energy_pareto(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots the groups in the latency/energy plane with the Pareto frontier.

    Lower is better on both axes; the frontier links the groups that no
    other group matches or beats on both (with a different point).

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one point per group.

    Raises:
        NotApplicableError: A metric cannot be found, or no group has both.
    """
    x_aspect, x_metric, x_unit = _find_metric(ctx, options["x"])
    y_aspect, y_metric, y_unit = _find_metric(ctx, options["y"])
    x_by_label = ctx.metric(x_aspect, x_metric)
    y_by_label = ctx.metric(y_aspect, y_metric)
    labels = [
        label
        for label in ctx.labels
        if label in x_by_label and label in y_by_label
    ]
    if not labels:
        raise registry.NotApplicableError("no group has both metrics")
    points = [(x_by_label[label], y_by_label[label], label) for label in labels]
    frontier = sorted(
        point for point in points if not _is_dominated(point, points)
    )
    traces = [
        ir.Trace(
            "scatter",
            x=[x],
            y=[y],
            name=label,
            style={"ci": i, "s": 60},
            text=[label],
            meta={"label": label},
        )
        for i, (x, y, label) in enumerate(points)
    ]
    if len(frontier) > 1:
        traces.append(
            ir.Trace(
                "step",
                x=[x for x, _, _ in frontier],
                y=[y for _, y, _ in frontier],
                name="Pareto frontier",
                color=theme.NEUTRAL,
                style={"dash": "--"},
            )
        )
    return chart_helpers.single(
        "Latency vs energy",
        ir.Panel(
            traces=traces,
            xlabel=f"{x_metric} ({x_unit})",
            ylabel=f"{y_metric} ({y_unit})",
        ),
    )


def _is_dominated(
    point: tuple[float, float, str], points: list[tuple[float, float, str]]
) -> bool:
    """Tells whether another point is at least as good on both axes."""
    return any(
        other[0] <= point[0] and other[1] <= point[1] and other[:2] != point[:2]
        for other in points
    )
