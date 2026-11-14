"""Plot kinds drawn from the aggregated report tables.

These are the original six (breakdown, topk, traffic_delta, skew, roofline,
scaling) plus waterfall, gates, scorecard, sweep and the latency/energy
Pareto.
"""

from __future__ import annotations

import re
from typing import Any, cast, TYPE_CHECKING

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz import theme
from cta.viz.kinds import chart_helpers

if TYPE_CHECKING:
    from cta.core import models

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
# Per-entity tables `topk` tries in order: (aspect, table, key column, unit).
_TOPK_SOURCES = (
    ("kernel_diff", "by_family", "op_family", "ms"),
    ("layer_fusion", "by_layer", "name", "ms"),
    ("comm_collectives", "by_comm_type", "comm_type", "MB"),
    ("os_runtime", "by_syscall", "name", "ms"),
    ("roofline_diff", "by_kernel", "canonical_name", "us"),
)
# Key column of the kernel_diff tables other than by_kernel.
_KERNEL_DIFF_KEY_BY_TABLE = {"by_family": "op_family", "by_scope": "scope"}
# Summary metrics tried, in order, when no metric is requested.
_PREFERRED_METRICS = (
    "mean_step_ms",
    "critical_path_us",
    "throughput_per_s",
    "energy_per_query_mj",
    "total_energy_j",
)
# Smallest value drawn on the roofline's log axes.
_LOG_AXIS_FLOOR = 1e-3
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


@spec.register_plot(
    "breakdown",
    group="overview",
    title="Timeline execution breakdown",
    legacy=True,
    needs="aspect timeline_overlap or dag_critical_path",
)
def breakdown(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots stacked compute, overlapped comm, exposed comm and idle time.

    Uses the timeline_overlap aspect, or else the critical-path composition
    from dag_critical_path.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one stacked bar per group.

    Raises:
        spec.NotApplicableError: Neither aspect is in the report.
    """
    series: tuple[tuple[str, str], ...]
    if ctx.has_aspect("timeline_overlap"):
        aspect, series = "timeline_overlap", _TIMELINE_OVERLAP_SERIES
        ylabel, title = "Time (ms)", "Timeline execution breakdown"
    elif ctx.has_aspect("dag_critical_path"):
        aspect, series = "dag_critical_path", _CRITICAL_PATH_SERIES
        ylabel, title = "Critical-path time (us)", "Critical-path composition"
    else:
        raise spec.NotApplicableError(
            "Breakdown plot requires 'timeline_overlap' or 'dag_critical_path'"
            " aspect"
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


@spec.register_plot(
    "topk",
    group="compare",
    title="Top contributors / hotspots",
    aliases=("tornado", "top_k", "hotspots"),
    legacy=True,
    options={
        "top_k": spec.Opt("int", 10, "How many entities."),
        "table": spec.Opt(
            "str", None, "Table of kernel_diff, e.g. by_kernel or by_scope."
        ),
    },
    needs="a per-entity report table (kernel_diff, layer_fusion, ...)",
)
def topk(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots the top-K entities by time (one group) or by delta (two groups).

    The entities come from the first per-entity report table that has the
    needed column; the `table` option puts a kernel_diff table first.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of horizontal bars: hotspots for one group, slower
        and faster entities for a pair.

    Raises:
        spec.NotApplicableError: The report has more than two groups, or no
            per-entity table has the needed column.
    """
    report = ctx.require_report()
    if len(report.groups) not in (1, 2):
        raise spec.NotApplicableError(
            "Tornado / Top-K plot requires 1 trace (hotspots) or 2 traces"
            " (Baseline vs Candidate)"
        )
    is_single_group = len(report.groups) == 1
    first_label = report.groups[0].label
    value_column = first_label if is_single_group else "delta"
    table, key_column, unit = _topk_table(report, options, value_column)
    names = [str(name)[:32] for name in table[key_column].to_list()]
    values = table[value_column].to_list()
    if is_single_group:
        hotspot_bars = ir.Trace(
            "barh",
            x=values,
            y=names,
            name=first_label,
            style={"ci": 0, "nolegend": True},
        )
        return chart_helpers.single(
            f"Top hotspots ({first_label})",
            ir.Panel(
                traces=[hotspot_bars],
                xlabel=f"{first_label} [{unit}]",
                invert_y=True,
                legend=False,
            ),
        )
    named_values = list(zip(names, values, strict=True))
    slower = [(name, value) for name, value in named_values if value > 0]
    faster = [(name, value) for name, value in named_values if value <= 0]
    traces = [
        ir.Trace(
            "barh",
            x=[value for _, value in bars],
            y=[name for name, _ in bars],
            name=series_name,
            color=color,
        )
        for series_name, color, bars in (
            ("slower", theme.BAD, slower),
            ("faster", theme.GOOD, faster),
        )
        if bars
    ]
    candidate_label = report.groups[1].label
    return chart_helpers.single(
        "Top contributors to net change",
        ir.Panel(
            traces=traces,
            xlabel=f"Delta ({candidate_label} - {first_label}) [{unit}]",
            invert_y=True,
        ),
    )


def _topk_table(
    report: models.ComparisonReport,
    options: dict[str, Any],
    value_column: str,
) -> tuple[pl.DataFrame, str, str]:
    """Finds the first per-entity table with rows and the value column.

    Args:
        report: The report being plotted.
        options: Resolved options of `topk` (reads `table` and `top_k`).
        value_column: Column the bars show.

    Returns:
        The table cut to its first `top_k` rows, its key column (the first
        column when the expected key is missing) and the value unit.

    Raises:
        spec.NotApplicableError: No candidate table qualifies.
    """
    candidates = list(_TOPK_SOURCES)
    if options.get("table"):
        table_name = options["table"]
        key_column = _KERNEL_DIFF_KEY_BY_TABLE.get(table_name, "canonical_name")
        candidates = [
            ("kernel_diff", table_name, key_column, "ms"),
            *candidates,
        ]
    for aspect_name, table_name, key_column, unit in candidates:
        if (
            aspect_name not in report.aspects
            or table_name not in report.aspects[aspect_name].details
        ):
            continue
        table = report.aspects[aspect_name].details[table_name]
        if table.height and value_column in table.columns:
            table = table.head(int(options["top_k"]))
            if key_column not in table.columns:
                key_column = table.columns[0]
            return table, key_column, unit
    raise spec.NotApplicableError(
        f"No entity breakdown table with '{value_column}' available for"
        " Top-K / tornado plot"
    )


def _traffic_matrix(volumes: pl.DataFrame, n_ranks: int) -> np.ndarray:
    """Fills an n_ranks x n_ranks source-by-destination matrix of MB sent."""
    matrix = np.zeros((n_ranks, n_ranks))
    for row in volumes.iter_rows(named=True):
        matrix[int(row["src_rank"]), int(row["dst_rank"])] = float(
            row["volume_mb"]
        )
    return matrix


def _highest_rank(volumes: pl.DataFrame) -> int:
    """Returns the highest source or destination rank in a traffic table."""
    return max(
        int(cast(Any, volumes["src_rank"].max())),
        int(cast(Any, volumes["dst_rank"].max())),
    )


@spec.register_plot(
    "traffic_delta",
    group="distributed",
    title="Collective traffic matrix",
    aliases=("heatmap", "traffic"),
    legacy=True,
    needs="aspect traffic_matrix",
)
def traffic(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots rank-to-rank collective traffic volume.

    Draws one heatmap per group, or for a pair the candidate minus the
    baseline.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel delta chart for two groups, else a grid of heatmaps.

    Raises:
        spec.NotApplicableError: The report lacks the traffic_matrix aspect.
    """
    labels = ctx.labels
    traffic_aspect = ctx.require_report().aspects.get("traffic_matrix")
    if traffic_aspect is None:
        raise spec.NotApplicableError(
            "Traffic heatmap requires 'traffic_matrix' aspect"
        )
    volumes_per_label = [
        traffic_aspect.details.get(f"matrix_{label}", pl.DataFrame())
        for label in labels
    ]
    n_ranks = max(
        (
            _highest_rank(volumes) + 1
            for volumes in volumes_per_label
            if volumes.height
        ),
        default=2,
    )
    ranks = list(range(n_ranks))
    if len(labels) == 2:
        baseline_matrix = _traffic_matrix(volumes_per_label[0], n_ranks)
        candidate_matrix = _traffic_matrix(volumes_per_label[1], n_ranks)
        delta = candidate_matrix - baseline_matrix
        color_limit = max(1e-6, float(np.abs(delta).max()))
        heatmap = ir.Trace(
            "heatmap",
            x=ranks,
            y=ranks,
            z=delta.tolist(),
            style={"cmap": "RdBu_r", "vmin": -color_limit, "vmax": color_limit},
        )
        return chart_helpers.single(
            f"Traffic delta ({labels[1]} - {labels[0]})",
            ir.Panel(
                traces=[heatmap],
                xlabel="Destination rank",
                ylabel="Source rank",
                colorbar="Delta volume (MB)",
                legend=False,
            ),
        )
    panels = []
    for label, volumes in zip(labels, volumes_per_label, strict=True):
        heatmap = ir.Trace(
            "heatmap",
            x=ranks,
            y=ranks,
            z=_traffic_matrix(volumes, n_ranks).tolist(),
            style={"cmap": "Blues"},
        )
        panels.append(
            ir.Panel(
                traces=[heatmap],
                title=label,
                xlabel="Destination rank",
                ylabel="Source rank",
                colorbar="Volume (MB)",
                legend=False,
            )
        )
    return chart_helpers.grid("Traffic matrix", panels)


@spec.register_plot(
    "skew",
    group="distributed",
    title="Per-rank load distribution",
    legacy=True,
    needs="aspect rank_skew",
)
def skew(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots a box of per-rank compute time for each group.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one box per group that has per-rank data.

    Raises:
        spec.NotApplicableError: The report lacks the rank_skew aspect or its
            per-rank data.
    """
    if not ctx.has_aspect("rank_skew"):
        raise spec.NotApplicableError("Skew plot requires 'rank_skew' aspect")
    per_rank_by_label = ctx.require_report().aspects["rank_skew"].details
    traces = []
    for i, label in enumerate(ctx.labels):
        if (
            label not in per_rank_by_label
            or not per_rank_by_label[label].height
        ):
            continue
        traces.append(
            ir.Trace(
                "box",
                y=per_rank_by_label[label]["compute_us"].to_list(),
                name=label,
                style={"ci": i},
                meta={"label": label},
            )
        )
    if not traces:
        raise spec.NotApplicableError(
            "No per-rank distribution data available for skew plot"
        )
    return chart_helpers.single(
        "Multi-rank load distribution",
        ir.Panel(
            traces=traces, ylabel="Per-rank compute time (us)", legend=False
        ),
    )


@spec.register_plot(
    "roofline",
    group="kernels",
    title="Kernel roofline",
    legacy=True,
    options={
        "peak_tflops": spec.Opt(
            "float",
            None,
            "Compute roof (TFLOP/s); draws the roofline with peak_bw_gbs.",
        ),
        "peak_bw_gbs": spec.Opt("float", None, "Memory roof (GB/s)."),
    },
    needs="aspect roofline_diff (Nsight Compute data)",
)
def roofline(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots arithmetic intensity against achieved TFLOP/s per kernel.

    For a pair, a line connects each kernel's baseline and candidate points.
    With both peaks given, the roofline itself is drawn.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel log-log chart.

    Raises:
        spec.NotApplicableError: The report lacks the roofline_diff aspect or
            kernels with FLOP/byte data.
    """
    if not ctx.has_aspect("roofline_diff"):
        raise spec.NotApplicableError(
            "Roofline plot requires 'roofline_diff' aspect"
        )
    roofline_aspect = ctx.require_report().aspects["roofline_diff"]
    labels = ctx.labels
    traces: list[ir.Trace] = []
    kernels_by_label = {
        label: roofline_aspect.details.get(f"roofline_{label}", pl.DataFrame())
        for label in labels
    }
    for i, label in enumerate(labels):
        kernels = kernels_by_label[label]
        if kernels.is_empty():
            continue
        intensity = kernels["arithmetic_intensity"].to_numpy()
        tflops = kernels["achieved_tflops"].to_numpy()
        traces.append(
            ir.Trace(
                "scatter",
                x=np.maximum(_LOG_AXIS_FLOOR, intensity).tolist(),
                y=np.maximum(_LOG_AXIS_FLOOR, tflops).tolist(),
                name=label,
                style={"ci": i},
                text=[
                    str(name) for name in kernels["canonical_name"].to_list()
                ],
                meta={"label": label},
            )
        )
    if (
        len(labels) == 2
        and kernels_by_label[labels[0]].height
        and kernels_by_label[labels[1]].height
    ):
        traces.extend(
            _shift_lines(
                kernels_by_label[labels[0]], kernels_by_label[labels[1]]
            )
        )
    if not traces:
        raise spec.NotApplicableError(
            "No kernels with FLOP/byte data (needs Nsight Compute counters)"
        )
    if options.get("peak_tflops") and options.get("peak_bw_gbs"):
        intensities = np.logspace(-2, 4, 80)
        # Attainable TFLOP/s: the compute roof, or the bandwidth in TB/s
        # times the arithmetic intensity.
        attainable_tflops = np.minimum(
            options["peak_tflops"], options["peak_bw_gbs"] / 1e3 * intensities
        )
        traces.append(
            ir.Trace(
                "line",
                x=intensities.tolist(),
                y=attainable_tflops.tolist(),
                name="roofline",
                color="#333333",
                style={"width": 1.6},
            )
        )
    return chart_helpers.single(
        "Kernel roofline" + (" shift" if len(labels) == 2 else ""),
        ir.Panel(
            traces=traces,
            xlabel="Arithmetic intensity (FLOP / byte)",
            ylabel="Performance (TFLOP/s)",
            xscale="log",
            yscale="log",
        ),
    )


def _shift_lines(
    baseline_kernels: pl.DataFrame, candidate_kernels: pl.DataFrame
) -> list[ir.Trace]:
    """Connects each kernel's baseline and candidate roofline points."""
    pairs = baseline_kernels.join(
        candidate_kernels, on="canonical_name", suffix="_c"
    )
    return [
        ir.Trace(
            "line",
            x=[
                max(_LOG_AXIS_FLOOR, row["arithmetic_intensity"]),
                max(_LOG_AXIS_FLOOR, row["arithmetic_intensity_c"]),
            ],
            y=[
                max(_LOG_AXIS_FLOOR, row["achieved_tflops"]),
                max(_LOG_AXIS_FLOOR, row["achieved_tflops_c"]),
            ],
            color=theme.NEUTRAL,
            style={"width": 0.8, "nolegend": True},
        )
        for row in pairs.iter_rows(named=True)
    ]


def _find_metric(
    ctx: data.DataContext, metric: str | None
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
        spec.NotApplicableError: The requested metric is in no summary, or the
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
        raise spec.NotApplicableError(
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
        raise spec.NotApplicableError("No summary metrics available")
    return (
        first_with_summary.name,
        str(first_with_summary.summary["metric"][0]),
        str(first_with_summary.summary["unit"][0]),
    )


@spec.register_plot(
    "scaling",
    group="sweep",
    title="Metric across groups",
    legacy=True,
    options={"metric": spec.Opt("str", None, "Summary metric name.")},
)
def scaling(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots one summary metric across the groups.

    A bar for one group, a line for several.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart.

    Raises:
        spec.NotApplicableError: The metric cannot be found (see
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


@spec.register_plot(
    "waterfall",
    group="compare",
    title="Delta waterfall",
    needs="2 groups; a per-entity table of kernel_diff",
    options={
        "top_k": spec.Opt(
            "int", 8, "Entities shown; the rest collapse into 'other'."
        ),
        "table": spec.Opt(
            "str",
            "by_family",
            "kernel_diff table: by_family, by_kernel or by_scope.",
        ),
    },
)
def waterfall(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots how per-entity time changes add up from baseline to candidate.

    The entities with the largest absolute change are shown; the rest are
    summed into ``other``.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of floating bars between the two totals.

    Raises:
        spec.NotApplicableError: The report does not have exactly two groups, or
            the table lacks per-group columns.
    """
    report = ctx.require_report()
    if len(report.groups) != 2:
        raise spec.NotApplicableError(
            "Waterfall needs exactly two groups (baseline and candidate)"
        )
    per_entity = ctx.report_table("kernel_diff", options["table"])
    baseline_label, candidate_label = ctx.labels
    key_column = per_entity.columns[0]
    if not {baseline_label, candidate_label, "delta"} <= set(
        per_entity.columns
    ):
        raise spec.NotApplicableError(
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


@spec.register_plot(
    "gates",
    group="compare",
    title="Regression gates",
    needs="2+ groups and --gate thresholds",
)
def gates(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each gated metric's observed change (%) against its threshold.

    Violated gates are drawn in red, with a vertical line per threshold.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart of horizontal bars.

    Raises:
        spec.NotApplicableError: No gates are configured, or no gated metric has
            a delta in the report.
    """
    report = ctx.require_report()
    config = ctx.config
    if not config.gates:
        raise spec.NotApplicableError(
            "no gates configured (use --gate 'metric:+5%' or a manifest"
            " 'gates:' block)"
        )
    gate_rows = _gate_rows(report, config.gates)
    if not gate_rows:
        raise spec.NotApplicableError(
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
    report: models.ComparisonReport, thresholds: dict[str, float]
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


@spec.register_plot(
    "scorecard",
    group="compare",
    title="Metric scorecard (% change vs first group)",
    needs="2+ groups",
    options={
        "max_metrics": spec.Opt("int", 40, "Cap on metrics shown."),
        "aspects": spec.Opt("list", None, "Only these aspects."),
    },
)
def scorecard(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots every summary metric's % change relative to the first group.

    Metrics whose first-group value is zero or missing for a group are left
    out; the largest changes come first.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel heatmap of metric x group.

    Raises:
        spec.NotApplicableError: The report has fewer than two groups or no
            comparable metric.
    """
    report = ctx.require_report()
    if len(report.groups) < 2:
        raise spec.NotApplicableError("Scorecard needs at least two groups")
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
        raise spec.NotApplicableError("no comparable metrics")
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


@spec.register_plot(
    "sweep",
    group="sweep",
    title="Metric vs configuration",
    needs="2+ groups",
    options={
        "metric": spec.Opt(
            "list", None, "Summary metric(s); default the first preferred one."
        ),
        "normalize": spec.Opt(
            "str",
            "none",
            "none, first (value / first group) or speedup (first / value).",
            ("none", "first", "speedup"),
        ),
        "x": spec.Opt(
            "str",
            "auto",
            "auto (a number in the label if every label has one, else the"
            " group index), index or label.",
            ("auto", "index", "label"),
        ),
        "logx": spec.Opt("bool", False, "Log x axis."),
        "logy": spec.Opt("bool", False, "Log y axis."),
    },
)
def sweep(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots one or more metrics across ordered groups, optionally normalised.

    Suits a sweep over batch size, GPUs, precision and the like; x is the
    number in each label when every label has a distinct one.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one line per metric.

    Raises:
        spec.NotApplicableError: A metric cannot be found (see `_find_metric`).
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


@spec.register_plot(
    "energy_pareto",
    group="edge",
    title="Latency vs energy",
    needs="a latency and an energy metric (edge/jtop bundles)",
    options={
        "x": spec.Opt("str", "mean_step_ms", "Latency-like metric."),
        "y": spec.Opt("str", "energy_per_query_mj", "Energy-like metric."),
    },
)
def energy_pareto(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots the groups in the latency/energy plane with the Pareto frontier.

    Lower is better on both axes; the frontier links the groups that no
    other group matches or beats on both (with a different point).

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one point per group.

    Raises:
        spec.NotApplicableError: A metric cannot be found, or no group has both.
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
        raise spec.NotApplicableError("no group has both metrics")
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
