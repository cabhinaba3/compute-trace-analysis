"""Plots of how much a candidate differs from a baseline, and how surely."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

_PAIR_COLUMNS = {"metric", "baseline", "candidate", "delta_pct"}
# Kept apart from the palette so "significant" reads the same everywhere.
_SIGNIFICANT, _NOT_SIGNIFICANT = 0, 7


def _pair_summaries(
    ctx: context.DataContext, aspect: str, match: str
) -> pl.DataFrame:
    """Stacks the baseline-versus-candidate summaries of the report.

    Args:
        ctx: The report being plotted.
        aspect: One aspect to read, or "" for every aspect with a summary
            of two groups.
        match: Regular expression a metric name must contain; "" keeps all.

    Returns:
        One row per metric, with an ``aspect`` column added.

    Raises:
        spec.NotApplicableError: The report has no two-group summary.
    """
    report = ctx.require_report()
    if len(report.groups) != 2:
        raise spec.NotApplicableError(
            "needs exactly two groups (a baseline and a candidate)"
        )
    frames = []
    for name in [aspect] if aspect else list(report.aspects):
        table = ctx.report_table(name)
        if set(table.columns) >= _PAIR_COLUMNS:
            frames.append(table.with_columns(pl.lit(name).alias("aspect")))
    if not frames:
        raise spec.NotApplicableError(
            "no aspect has a baseline-versus-candidate summary"
        )
    rows = pl.concat(frames, how="diagonal_relaxed")
    if match:
        rows = rows.filter(pl.col("metric").str.contains(match))
    if rows.is_empty():
        raise spec.NotApplicableError(f"no metric matches '{match}'")
    return rows


def _optional(rows: pl.DataFrame, column: str, default: Any) -> pl.Expr:
    """Reads a summary column that some aspects do not have."""
    if column in rows.columns:
        return pl.col(column)
    return pl.lit(default)


@spec.register_plot(
    "forest",
    group="compare",
    title="Change of every metric, with confidence intervals",
    needs="a two-group report (baseline and candidate)",
    aliases=("forest_plot", "effect_sizes"),
    options={
        "aspect": spec.Opt("str", "", "Only this aspect (default: all)."),
        "match": spec.Opt(
            "str", "", "Only metrics whose name matches this expression."
        ),
        "top_k": spec.Opt(
            "int", 25, "Metrics with the largest change (0 = all)."
        ),
        "only_significant": spec.Opt(
            "bool", False, "Leave out changes that are not significant."
        ),
    },
)
def forest(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each metric's change in percent with a 95% interval.

    The interval combines the two groups' confidence intervals, so with one
    trial per group there is none and only the point is drawn. Metrics from
    different aspects share the axis because the change is relative.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with a lane per metric.

    Raises:
        spec.NotApplicableError: The report has no two-group summary, or
            nothing is left after the filters.
    """
    rows = _pair_summaries(ctx, options["aspect"], options["match"])
    baseline = pl.col("baseline").abs()
    half_width = (
        pl.when(baseline > 1e-12)
        .then(
            (
                _optional(rows, "ci95_baseline", 0.0).fill_null(0.0) ** 2
                + _optional(rows, "ci95_candidate", 0.0).fill_null(0.0) ** 2
            ).sqrt()
            / baseline
            * 100.0
        )
        .otherwise(0.0)
    )
    rows = (
        rows.with_columns(
            (pl.col("aspect") + ": " + pl.col("metric")).alias("lane"),
            half_width.alias("half"),
            _optional(rows, "significant", False)
            .fill_null(False)
            .alias("is_significant"),
        )
        .filter(pl.col("delta_pct").is_finite())
        .with_columns(
            (pl.col("delta_pct") - pl.col("half")).alias("low"),
            (pl.col("delta_pct") + pl.col("half")).alias("high"),
        )
    )
    if options["only_significant"]:
        rows = rows.filter(pl.col("is_significant"))
    if rows.is_empty():
        raise spec.NotApplicableError("no metric is left to draw")
    if options["top_k"] > 0:
        rows = rows.sort(pl.col("delta_pct").abs(), descending=True).head(
            options["top_k"]
        )
    rows = rows.sort("delta_pct", descending=True)
    lanes = rows["lane"].to_list()
    traces: list[ir.Trace] = [ir.Trace("vline", x=[0.0], style={"ci": 5})]
    for is_significant, name, color in (
        (True, "significant", _SIGNIFICANT),
        (False, "not significant", _NOT_SIGNIFICANT),
    ):
        part = rows.filter(pl.col("is_significant") == is_significant)
        if part.is_empty():
            continue
        style = {"lanes": lanes, "ci": color}
        traces.append(
            ir.Trace(
                "segment",
                x=part["low"].to_list(),
                x1=part["high"].to_list(),
                y=part["lane"].to_list(),
                style={**style, "width": 3},
                meta={},
            )
        )
        traces.append(
            ir.Trace(
                "scatter",
                x=part["delta_pct"].to_list(),
                y=part["lane"].to_list(),
                name=name,
                style=style,
            )
        )
    labels = ctx.labels
    panel = ir.Panel(
        traces=traces,
        xlabel=f"change of {labels[1]} against {labels[0]} (%)",
        invert_y=True,
    )
    notes = (
        "Whiskers combine the two groups' 95% confidence intervals; "
        "they need two or more trials per group."
    )
    chart = chart_helpers.single("Change of every metric", panel, notes=notes)
    chart.size = (6.5, max(3.0, 0.3 * len(lanes) + 1.4))
    return chart


@spec.register_plot(
    "volcano",
    group="compare",
    title="Size of the change against its significance",
    needs="a two-group report with two or more trials per group",
    aliases=("volcano_plot",),
    options={
        "aspect": spec.Opt("str", "", "Only this aspect (default: all)."),
        "match": spec.Opt(
            "str", "", "Only metrics whose name matches this expression."
        ),
        "x": spec.Opt(
            "str",
            "delta_pct",
            "Size of the change: percent, or Cohen's d.",
            ("delta_pct", "cohens_d"),
        ),
        "p": spec.Opt(
            "str",
            "p_welch",
            "Which test's p-value to use.",
            ("p_welch", "p_mannwhitney"),
        ),
        "labels": spec.Opt(
            "int", 8, "Points to name, strongest first (0 = none)."
        ),
    },
)
def volcano(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots every metric's change against -log10 of its p-value.

    Metrics high on the plot are unlikely to be noise; metrics far to either
    side changed a lot. The line is the report's significance level.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel scatter chart.

    Raises:
        spec.NotApplicableError: The report has no two-group summary, or no
            metric has a p-value (one trial per group gives none).
    """
    rows = _pair_summaries(ctx, options["aspect"], options["match"])
    x_column, p_column = options["x"], options["p"]
    if x_column not in rows.columns or p_column not in rows.columns:
        raise spec.NotApplicableError(f"the summaries have no {p_column}")
    rows = rows.filter(
        pl.col(x_column).is_finite(),
        pl.col(p_column).is_not_null(),
        pl.col(p_column).is_finite(),
        pl.col(p_column) > 0,
    )
    if rows.is_empty():
        raise spec.NotApplicableError(
            "no metric has a p-value; give each group two or more trials"
        )
    rows = rows.with_columns(
        (-pl.col(p_column).log10()).alias("strength"),
        (pl.col("aspect") + ": " + pl.col("metric")).alias("lane"),
        _optional(rows, "significant", False).fill_null(False).alias("sig"),
    )
    traces: list[ir.Trace] = [
        ir.Trace("vline", x=[0.0], style={"ci": 5}),
        ir.Trace(
            "hline",
            y=[float(-np.log10(ctx.config.alpha))],
            style={"ci": 5, "dash": "--"},
        ),
    ]
    for is_significant, name, color in (
        (True, "significant", _SIGNIFICANT),
        (False, "not significant", _NOT_SIGNIFICANT),
    ):
        part = rows.filter(pl.col("sig") == is_significant)
        if part.is_empty():
            continue
        traces.append(
            ir.Trace(
                "scatter",
                x=part[x_column].to_list(),
                y=part["strength"].to_list(),
                name=name,
                text=part["lane"].to_list(),
                style={"ci": color},
            )
        )
    if options["labels"] > 0:
        named = rows.with_columns(
            (pl.col(x_column).abs() * pl.col("strength")).alias("score")
        ).top_k(options["labels"], by="score")
        traces.append(
            ir.Trace(
                "text",
                x=named[x_column].to_list(),
                y=named["strength"].to_list(),
                text=named["metric"].to_list(),
                style={"ci": 5},
            )
        )
    x_label = (
        "change (%)" if x_column == "delta_pct" else "effect size (Cohen's d)"
    )
    return chart_helpers.single(
        "Change against significance",
        ir.Panel(
            traces=traces,
            xlabel=x_label,
            ylabel=f"-log10 {p_column.replace('_', ' ')}",
        ),
    )
