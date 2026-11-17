"""Checks whether the data suits the plot kind asked for.

A violin of five samples, a line through unordered categories or a cross-rank
plot of a one-rank trace still draw, but they mislead. These checks look at
the built chart (sample size, cardinality, variance, ordering) and at the
trace (rank count, step drift, kernel durations), and warn without stopping
the plot. Each message ends with "Suggested alternatives:" and the names of
registered plot kinds.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any
import warnings

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import spec


class SuitabilityWarning(UserWarning):
    """A plot kind that is a poor fit for the data it was given."""


def _to_finite_float(val: Any) -> float | None:
    """Returns val as a float, or None if it is not a finite number."""
    if val is None or isinstance(val, (str, bytes)):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _is_numeric_str(val: Any) -> bool:
    """Returns whether val reads as a number."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _get_world_sizes(ctx: context.DataContext) -> set[int]:
    """Returns the distinct world sizes of the report's trace artifacts."""
    world_sizes: set[int] = set()
    if ctx.report and ctx.report.groups:
        for group in ctx.report.groups:
            for trial in group.trials:
                for art in trial.artifacts.values():
                    world_sizes.add(art.world_size)
    if not world_sizes:
        try:
            df = ctx.frame("kernels")
            if "rank" in df.columns:
                ranks = df["rank"].drop_nulls().unique().to_list()
                if ranks:
                    world_sizes.add(len(ranks))
        except Exception:
            pass
    return world_sizes


def _check_temporal_drift(ctx: context.DataContext) -> bool:
    """Returns whether the median step time moves over the steps.

    Drift means the Theil-Sen slope over the steps changes the median step
    time by more than 5% across the run.
    """
    try:
        steps_df = ctx.frame("steps")
        cols = steps_df.columns
        if "duration_ms" not in cols or "step" not in cols:
            return False
        grouped = (
            steps_df.group_by("step")
            .agg(pl.col("duration_ms").median().alias("dur_med"))
            .sort("step")
        )
        if grouped.height < 4:
            return False
        x = grouped["step"].to_numpy().astype(float)
        y = grouped["dur_med"].to_numpy().astype(float)
        diffs = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[j] != x[i]:
                    diffs.append((y[j] - y[i]) / (x[j] - x[i]))
        if not diffs:
            return False
        slope = float(np.median(diffs))
        y_med = float(np.median(y))
        total_change = abs(slope) * (x.max() - x.min())
        return total_change > 0.05 * y_med and abs(slope) > 1e-4
    except Exception:
        return False


def check_suitability(
    ctx: context.DataContext,
    kind: str,
    options: Mapping[str, Any],
    chart: ir.Chart | None = None,
) -> list[str]:
    """Checks the data against the plot kind and lists the problems found.

    Args:
        ctx: Data the plot draws from.
        kind: Plot kind name or alias.
        options: The kind's options, resolved or as given.
        chart: The built chart, if there is one; most checks read it. A
            chart whose ``meta["skip_suitability"]`` is set is not checked:
            its marks mean something other than what the checks assume.

    Returns:
        One message per distinct problem, each ending with suggested kinds.
    """
    if chart is not None and chart.meta.get("skip_suitability"):
        return []
    messages: list[str] = []
    seen: set[str] = set()

    def add_warning(msg: str) -> None:
        if msg not in seen:
            seen.add(msg)
            messages.append(msg)

    norm_kind = kind.lower().replace("-", "_")
    is_violin_kind = norm_kind in ("violin", "violinplot", "violin_plot")
    is_boxplot_kind = norm_kind in ("boxplot", "box", "box_plot")
    is_line_kind = norm_kind in ("line", "line_plot", "lineplot")

    has_violin_mark = False
    has_box_mark = False
    has_line_mark = False
    if chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "violin":
                    has_violin_mark = True
                elif trace.mark == "box":
                    has_box_mark = True
                elif trace.mark == "line":
                    has_line_mark = True

    # Sample size.
    if (is_violin_kind or has_violin_mark) and chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "violin":
                    raw_y = trace.y if trace.y is not None else []
                    vals = [
                        f
                        for v in raw_y
                        if (f := _to_finite_float(v)) is not None
                    ]
                    n = len(vals)
                    if n < 10:
                        add_warning(
                            f"Sample size (N={n}) is too small for reliable "
                            "kernel density estimation in violin plot. "
                            "Suggested alternatives: 'boxplot', "
                            "'kernel_scatter', 'category_compare'."
                        )

    if (is_boxplot_kind or has_box_mark) and chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "box":
                    raw_y = trace.y if trace.y is not None else []
                    vals = [
                        f
                        for v in raw_y
                        if (f := _to_finite_float(v)) is not None
                    ]
                    n = len(vals)
                    if n < 5:
                        add_warning(
                            f"Sample size (N={n}) is insufficient to calculate "
                            "meaningful quartiles for boxplot. "
                            "Suggested alternatives: 'kernel_scatter', "
                            "'category_compare'."
                        )

    if (is_line_kind or has_line_mark) and chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "line":
                    raw_seq = (
                        trace.x
                        if trace.x is not None
                        else (trace.y if trace.y is not None else [])
                    )
                    vals = [v for v in raw_seq if v is not None]
                    n = len(vals)
                    if n < 2:
                        add_warning(
                            f"Sample size (N={n}) is insufficient for line "
                            "plot. Suggested alternatives: 'kernel_scatter', "
                            "'category_compare'."
                        )

    # Distinct values and variance.
    if (is_violin_kind or has_violin_mark) and chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "violin":
                    raw_y = trace.y if trace.y is not None else []
                    arr = np.asarray(
                        [
                            f
                            for v in raw_y
                            if (f := _to_finite_float(v)) is not None
                        ],
                        dtype=float,
                    )
                    if len(arr) >= 10:
                        u_count = len(np.unique(arr))
                        variance = float(np.var(arr))
                        if u_count <= 3 or variance == 0.0:
                            add_warning(
                                f"Data has only {u_count} distinct values or "
                                "no variance, so a violin's density estimate "
                                "means little. Suggested alternatives: "
                                "'boxplot', 'metric_dist', 'category_compare'."
                            )

    if (is_boxplot_kind or has_box_mark) and chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "box":
                    raw_y = trace.y if trace.y is not None else []
                    arr = np.asarray(
                        [
                            f
                            for v in raw_y
                            if (f := _to_finite_float(v)) is not None
                        ],
                        dtype=float,
                    )
                    n = len(arr)
                    if n >= 30:
                        var = float(np.var(arr))
                        if var > 0:
                            m = float(np.mean(arr))
                            m3 = float(np.mean((arr - m) ** 3))
                            m4 = float(np.mean((arr - m) ** 4))
                            skew = m3 / (var**1.5)
                            kurt = m4 / (var**2)
                            bc = (skew**2 + 1.0) / kurt if kurt > 0 else 0.0
                            if bc > 0.555:
                                add_warning(
                                    "The data has several peaks and a boxplot "
                                    "hides them. "
                                    "Suggested alternatives: 'violin', "
                                    "'metric_dist'."
                                )

    # Ordering along the x axis.
    if (is_line_kind or has_line_mark) and chart is not None:
        for panel in chart.panels:
            for trace in panel.traces:
                if trace.mark == "line" and trace.x is not None:
                    x_list = list(trace.x)
                    if x_list:
                        if any(
                            isinstance(val, str) and not _is_numeric_str(val)
                            for val in x_list
                        ):
                            add_warning(
                                "The x axis holds categories, and a line "
                                "joins them as if they were ordered. "
                                "Suggested alternatives: 'category_compare', "
                                "'kernel_scatter'."
                            )
                        elif len(x_list) > 1 and trace.y is not None:
                            y_list = list(trace.y)
                            if len(y_list) == len(x_list):
                                seen_xy: dict[Any, Any] = {}
                                has_dup_varying_y = False
                                for xv, yv in zip(x_list, y_list, strict=True):
                                    if xv in seen_xy:
                                        if yv != seen_xy[xv]:
                                            has_dup_varying_y = True
                                            break
                                    else:
                                        seen_xy[xv] = yv
                                if has_dup_varying_y:
                                    add_warning(
                                        "Line plot contains duplicate X values "
                                        "with varying Y values without "
                                        "aggregation, causing vertical "
                                        "zigzagging. Suggested alternatives: "
                                        "'kernel_scatter', 'boxplot', "
                                        "'violin'."
                                    )

    # Step-time drift.
    if (
        is_violin_kind
        or is_boxplot_kind
        or norm_kind in ("step_compare", "step_dist")
    ) and _check_temporal_drift(ctx):
        add_warning(
            "Step time drifts over the run, and a distribution plot hides "
            "the trend. Suggested alternatives: 'step_trace', 'metric_trace'."
        )

    # Rank count.
    world_sizes = _get_world_sizes(ctx)
    max_world_size = max(world_sizes) if world_sizes else 1
    multi_rank_kinds = ("skew", "straggler", "rank_heatmap", "network_skew")

    if max_world_size == 1:
        if norm_kind in multi_rank_kinds or options.get("by") == "rank":
            add_warning(
                f"Multi-rank distribution plot '{kind}' requested, "
                "but trace has only a single rank (world_size=1). "
                "Cross-rank distribution is degenerate. "
                "Suggested alternatives: 'breakdown', 'timeline', 'step_dist'."
            )
    elif max_world_size == 2 and options.get("by") == "rank":
        is_dist_violin = (
            norm_kind == "metric_dist" and options.get("mode") == "violin"
        )
        if is_violin_kind or is_dist_violin or has_violin_mark:
            add_warning(
                "Cross-rank distribution contains only 2 ranks; "
                "violin density estimation is degenerate. "
                "Suggested alternatives: 'category_compare', 'boxplot', "
                "'step_compare'."
            )

    # Kernel durations.
    metric = str(options.get("metric", "")).lower()
    is_duration_metric = metric == "duration_ms" or "duration" in metric
    is_log_scale = bool(
        options.get("logx")
        or options.get("logy")
        or options.get("yscale") == "log"
    )
    if chart is not None:
        for p in chart.panels:
            if p.yscale == "log" or p.xscale == "log":
                is_log_scale = True

    if is_duration_metric:
        all_durations: list[float] = []
        if chart is not None:
            for p in chart.panels:
                for t in p.traces:
                    if (
                        t.mark
                        in ("box", "violin", "hist", "scatter", "bar", "line")
                        and t.y is not None
                    ):
                        for v in t.y:
                            num = _to_finite_float(v)
                            if num is not None and num > 0:
                                all_durations.append(num)
        if all_durations:
            min_dur = min(all_durations)
            max_dur = max(all_durations)
            if not is_log_scale and min_dur > 0 and (max_dur / min_dur) > 1000:
                ratio = max_dur / min_dur
                add_warning(
                    f"Kernel durations span multiple orders of magnitude "
                    f"(ratio {ratio:.1f}); linear scaling compresses "
                    "distribution details. Suggested alternatives: enable "
                    "log scale (logy=True) or use 'violin' with log scale."
                )
            is_deg_dur = (
                len(all_durations) < 5 or float(np.var(all_durations)) == 0.0
            )
            if (is_violin_kind or has_violin_mark) and is_deg_dur:
                add_warning(
                    "Kernel durations have zero variance or sample size "
                    f"(N={len(all_durations)}) is too small for violin "
                    "density estimation. Suggested alternatives: "
                    "'boxplot', 'metric_dist', 'category_compare'."
                )

    # Queue depth.
    is_queue_depth_kind = norm_kind in (
        "queue_depth",
        "concurrency",
        "stream_concurrency",
        "active_kernels",
    )
    if is_queue_depth_kind:
        kernel_count = None
        if chart is not None and "kernel_count" in chart.meta:
            kernel_count = int(chart.meta["kernel_count"])
        if kernel_count is None:
            try:
                k_df = ctx.frame("kernels", options)
                kernel_count = k_df.height
            except Exception:
                pass

        if kernel_count is not None and kernel_count < 5:
            add_warning(
                f"Sample size (N={kernel_count}) is insufficient for queue "
                "depth profiling. Suggested alternatives: 'timeline', "
                "'metric_trace'."
            )

        if chart is not None:
            max_concurrency = 0
            total_points = 0
            idle_points = 0
            for panel in chart.panels:
                for trace in panel.traces:
                    if trace.y is not None:
                        for val in trace.y:
                            v = _to_finite_float(val)
                            if v is not None:
                                total_points += 1
                                if v <= 0:
                                    idle_points += 1
                                if v > max_concurrency:
                                    max_concurrency = int(v)

            if (
                kernel_count is None or kernel_count >= 5
            ) and max_concurrency <= 1:
                add_warning(
                    "The queue depth never exceeds 1: no kernels overlap, "
                    "so execution is serial. Suggested alternatives: "
                    "'timeline', 'stream_util', 'breakdown'."
                )

            if total_points > 0 and (idle_points / total_points) > 0.8:
                add_warning(
                    "No kernel runs for over 80% of the window "
                    "(queue depth 0). Suggested alternatives: "
                    "'overlap_timeline', 'breakdown'."
                )

    return messages


def extract_suggested_alternatives(warning: str) -> list[str]:
    """Reads the suggested plot kinds out of a warning message.

    Args:
        warning: A message from ``check_suitability``.

    Returns:
        The quoted names after "Suggested alternatives:", up to the first
        full stop; empty if the message has none.
    """
    match = re.search(
        r"Suggested alternatives:\s*([^.]+)", warning, re.IGNORECASE
    )
    if not match:
        return []
    return re.findall(r"'([^']+)'", match.group(1))


def known_alternatives(warning: str) -> list[str]:
    """Returns the suggested kinds of a warning that are registered kinds.

    Args:
        warning: A message from ``check_suitability``.

    Returns:
        The names from ``extract_suggested_alternatives`` that
        ``spec.get_kind`` accepts, in order.
    """
    known = []
    for name in extract_suggested_alternatives(warning):
        try:
            spec.get_kind(name)
        except ValueError:
            continue
        known.append(name)
    return known


def validate_and_warn(
    ctx: context.DataContext,
    kind: str,
    options: Mapping[str, Any],
    chart: ir.Chart | None = None,
) -> None:
    """Warns about each problem ``check_suitability`` finds.

    Args:
        ctx: Data the plot draws from.
        kind: Plot kind name or alias.
        options: The kind's options, resolved or as given.
        chart: The built chart, if there is one.
    """
    for message in check_suitability(ctx, kind, options, chart=chart):
        warnings.warn(message, category=SuitabilityWarning, stacklevel=4)
