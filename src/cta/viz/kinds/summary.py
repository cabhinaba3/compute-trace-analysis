"""Plot kinds that summarise a report: breakdown, waterfall, gates, scores."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz import theme
from cta.viz.kinds import chart_helpers

if TYPE_CHECKING:
    from cta import models


# Legend name and summary metric of each stacked series of `breakdown`.
_TIMELINE_OVERLAP_SERIES = (
    ("Compute only", "compute_only_ms"),
    ("Overlapped comm", "overlapped_comm_ms"),
    ("Exposed comm", "exposed_comm_ms"),
    ("GPU idle", "gpu_idle_ms"),
)


_CRITICAL_PATH_SERIES = (
    ("CP compute", "cp_compute_us"),
    ("CP exposed comm", "cp_exposed_comm_us"),
)


@registry.register_plot(
    "breakdown",
    group="overview",
    title="Timeline execution breakdown",
    exported=True,
    needs="aspect overlap or critical_path",
)
def breakdown(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots stacked compute, overlapped comm, exposed comm and idle time.

    Uses the overlap aspect, or else the critical-path composition
    from critical_path.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one stacked bar per group.

    Raises:
        NotApplicableError: Neither aspect is in the report.
    """
    series: tuple[tuple[str, str], ...]
    if ctx.has_aspect("overlap"):
        aspect, series = "overlap", _TIMELINE_OVERLAP_SERIES
        ylabel, title = "Time (ms)", "Timeline execution breakdown"
    elif ctx.has_aspect("critical_path"):
        aspect, series = "critical_path", _CRITICAL_PATH_SERIES
        ylabel, title = "Critical-path time (us)", "Critical-path composition"
    else:
        raise registry.NotApplicableError(
            "Breakdown plot requires 'overlap' or 'critical_path' aspect"
        )
    labels = ctx.labels
    traces = []
    for i, (series_name, metric) in enumerate(series):
        value_by_label = ctx.metric(aspect, metric)
        traces.append(
            ir.Trace(
                "bar",
                x=labels,
                y=[value_by_label.get(label, 0.0) for label in labels],
                name=series_name,
                style={"ci": i},
            )
        )
    return chart_helpers.single(
        title, ir.Panel(traces=traces, ylabel=ylabel, stacked=True)
    )


@registry.register_plot(
    "waterfall",
    group="compare",
    title="Delta waterfall",
    needs="2 groups; a per-entity table of kernels",
    options={
        "top_k": registry.Opt(
            "int", 8, "Entities shown; the rest collapse into 'other'."
        ),
        "table": registry.Opt(
            "str",
            "by_family",
            "kernels table: by_family, by_kernel or by_scope.",
        ),
    },
)
def waterfall(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots how per-entity time changes add up from baseline to candidate.

    The entities with the largest absolute change are shown; the rest are
    summed into ``other``.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of floating bars between the two totals.

    Raises:
        NotApplicableError: The report does not have exactly two groups, or
            the table lacks per-group columns.
    """
    report = ctx.require_report()
    if len(report.groups) != 2:
        raise registry.NotApplicableError(
            "Waterfall needs exactly two groups (baseline and candidate)"
        )
    per_entity = ctx.report_table("kernels", options["table"])
    baseline_label, candidate_label = ctx.labels
    key_column = per_entity.columns[0]
    if not {baseline_label, candidate_label, "delta"} <= set(
        per_entity.columns
    ):
        raise registry.NotApplicableError(
            f"table '{options['table']}' has no per-group columns"
        )
    per_entity = per_entity.sort(pl.col("delta").abs(), descending=True)
    top_k = int(options["top_k"])
    top = per_entity.head(top_k)
    rest = per_entity.tail(max(0, per_entity.height - top_k))
    names = [str(name)[:28] for name in top[key_column].to_list()]
    deltas = top["delta"].to_list()
    if rest.height:
        names.append("other")
        deltas.append(float(rest["delta"].sum()))
    start_ms = float(per_entity[baseline_label].sum())
    # Each change floats on the running total: its bar spans the total
    # before and after it.
    running_ms = start_ms
    bar_bases, slower_heights, faster_heights = [], [], []
    for delta in deltas:
        bar_bases.append(min(running_ms, running_ms + delta))
        slower_heights.append(abs(delta) if delta > 0 else 0.0)
        faster_heights.append(abs(delta) if delta <= 0 else 0.0)
        running_ms += delta
    x = [baseline_label, *names, candidate_label]
    traces = [
        ir.Trace(
            "bar",
            x=x,
            y=[start_ms, *[0.0] * len(names), running_ms],
            y1=[0.0] * len(x),
            name="total",
            color=theme.NEUTRAL,
        ),
        ir.Trace(
            "bar",
            x=names,
            y=slower_heights,
            y1=bar_bases,
            name="slower",
            color=theme.BAD,
        ),
        ir.Trace(
            "bar",
            x=names,
            y=faster_heights,
            y1=bar_bases,
            name="faster",
            color=theme.GOOD,
        ),
    ]
    panel = ir.Panel(
        traces=traces,
        ylabel=f"Kernel time (ms), {options['table']}",
        overlay=True,
    )
    return chart_helpers.single(
        f"Where the {running_ms - start_ms:+.3g} ms change comes from", panel
    )


@registry.register_plot(
    "gates",
    group="compare",
    title="Regression gates",
    needs="2+ groups and --gate thresholds",
)
def gates(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each gated metric's observed change (%) against its threshold.

    Violated gates are drawn in red, with a vertical line per threshold.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of horizontal bars.

    Raises:
        NotApplicableError: No gates are configured, or no gated metric has
            a delta in the report.
    """
    report = ctx.require_report()
    config = ctx.config
    if not config.gates:
        raise registry.NotApplicableError(
            "no gates configured (use --gate 'metric:+5%' or a manifest"
            " 'gates:' block)"
        )
    gate_rows = _gate_rows(report, config.gates)
    if not gate_rows:
        raise registry.NotApplicableError(
            "none of the gated metrics has a delta in this report"
        )
    # A non-negative threshold caps an increase; a negative one a decrease.
    is_violated = [
        change_pct > threshold if threshold >= 0 else change_pct < threshold
        for _, change_pct, threshold in gate_rows
    ]
    rows_and_status = list(zip(gate_rows, is_violated, strict=True))
    passed = [row for row, violated in rows_and_status if not violated]
    failed = [row for row, violated in rows_and_status if violated]
    traces = [
        ir.Trace(
            "barh",
            x=[change_pct for _, change_pct, _ in rows],
            y=[metric for metric, _, _ in rows],
            name=series_name,
            color=color,
        )
        for series_name, color, rows in (
            ("within gate", theme.GOOD, passed),
            ("violated", theme.BAD, failed),
        )
        if rows
    ]
    for metric, _, threshold in gate_rows:
        traces.append(
            ir.Trace(
                "vline",
                x=[threshold],
                color="#333333",
                name=f"{metric} limit" if len(gate_rows) == 1 else None,
                style={"nolegend": len(gate_rows) > 1},
            )
        )
    return chart_helpers.single(
        "Gate results",
        ir.Panel(traces=traces, xlabel="Change vs baseline (%)", invert_y=True),
    )


def _gate_rows(
    report: models.Report, thresholds: dict[str, float]
) -> list[tuple[str, float, float]]:
    """Pairs each gated metric's delta (%) with its threshold.

    Args:
        report: The report being plotted.
        thresholds: Gate threshold (%) by metric name.

    Returns:
        (metric, change %, threshold) for every gated metric that the first
        aspect reporting it has a ``delta_pct`` for, in gate order.
    """
    rows = []
    for metric, threshold in thresholds.items():
        for aspect in report.aspects.values():
            summary = aspect.summary
            if (
                "delta_pct" in summary.columns
                and metric in summary["metric"].to_list()
            ):
                metric_row = summary.filter(pl.col("metric") == metric)
                change_pct = float(metric_row["delta_pct"][0])
                rows.append((metric, change_pct, float(threshold)))
                break
    return rows


@registry.register_plot(
    "scorecard",
    group="compare",
    title="Metric scorecard (% change vs first group)",
    needs="2+ groups",
    options={
        "max_metrics": registry.Opt("int", 40, "Cap on metrics shown."),
        "aspects": registry.Opt("list", None, "Only these aspects."),
    },
)
def scorecard(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots every summary metric's % change relative to the first group.

    Metrics whose first-group value is zero or missing for a group are left
    out; the largest changes come first.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel heatmap of metric x group.

    Raises:
        NotApplicableError: The report has fewer than two groups or no
            comparable metric.
    """
    report = ctx.require_report()
    if len(report.groups) < 2:
        raise registry.NotApplicableError("Scorecard needs at least two groups")
    labels = ctx.labels
    rows: list[tuple[str, list[float]]] = []
    for aspect_name, aspect in report.aspects.items():
        if options.get("aspects") and aspect_name not in options["aspects"]:
            continue
        if aspect.summary.is_empty():
            continue
        for metric in aspect.summary["metric"].to_list():
            value_by_label = ctx.metric(aspect_name, metric)
            if len(value_by_label) != len(labels):
                continue
            base = value_by_label[labels[0]]
            if base == 0.0:
                continue
            changes_pct = [
                100.0 * (value_by_label[label] - base) / abs(base)
                for label in labels[1:]
            ]
            rows.append((f"{aspect_name}.{metric}", changes_pct))
    if not rows:
        raise registry.NotApplicableError("no comparable metrics")
    rows = sorted(rows, key=lambda row: -max(abs(change) for change in row[1]))[
        : int(options["max_metrics"])
    ]
    color_limit = max(
        1e-6, max(abs(change) for _, changes in rows for change in changes)
    )
    heatmap = ir.Trace(
        "heatmap",
        x=labels[1:],
        y=[name for name, _ in rows],
        z=[changes for _, changes in rows],
        style={"cmap": "RdBu_r", "vmin": -color_limit, "vmax": color_limit},
    )
    return chart_helpers.single(
        f"% change vs {labels[0]}",
        ir.Panel(
            traces=[heatmap], colorbar="% change", legend=False, ylabel=""
        ),
    )
