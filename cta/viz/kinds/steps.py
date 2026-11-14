"""Per-step plot kinds.

Covers the step-time trace with drift and outliers, step-time distributions,
group comparisons, and rank straggling (heatmap and spread).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz import theme
from cta.viz.kinds import chart_helpers

# Columns that identify one step of one rank.
_STEP_KEYS = ("label", "trial", "rank", "idx")
_RANK_METRICS = ("step_ms", "compute_ms", "comm_ms", "memcpy_ms", "idle_pct")
# Kernel categories summed by each per-step kernel-time metric.
_CATEGORIES_BY_METRIC = {
    "compute_ms": ["compute"],
    "comm_ms": ["comm"],
    "memcpy_ms": ["memcpy"],
}


@spec.register_plot(
    "step_trace",
    group="steps",
    title="Step time over steps (trend and outliers)",
    needs="steps table",
    options={
        "rolling": spec.Opt(
            "int", 5, "Rolling-median window in steps (0 = off)."
        ),
        "outliers": spec.Opt(
            "float", 3.5, "Mark steps above median + k robust sigmas (0 = off)."
        ),
        "trend": spec.Opt(
            "bool",
            True,
            "Fit and draw a Theil-Sen trend with its Kendall p-value.",
        ),
        "yscale": spec.Opt(
            "str",
            "auto",
            "linear, log, or auto (log when the slowest step is >30x the"
            " median).",
            ("auto", "linear", "log"),
        ),
    },
)
def step_trace(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots step duration against step index, one line per trial and rank.

    Each line can carry a rolling median, a Theil-Sen trend (drift) and its
    outliers.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart; its notes list the lines that drift significantly.
    """
    steps = ctx.frame("steps", options)
    labels = chart_helpers.unique_in_order(steps, "label")
    traces: list[ir.Trace] = []
    notes = []
    for color_index, label in enumerate(labels):
        group_steps = steps.filter(pl.col("label") == label)
        has_trials = group_steps["trial"].n_unique() > 1
        has_ranks = group_steps["rank"].n_unique() > 1
        # Rolling medians enter the legend only when lines need telling apart.
        is_crowded = len(labels) > 1 or has_ranks or has_trials
        trial_ranks = sorted(
            set(
                zip(
                    group_steps["trial"].to_list(),
                    group_steps["rank"].to_list(),
                    strict=True,
                )
            )
        )
        for trial, rank in trial_ranks:
            series_steps = group_steps.filter(
                (pl.col("rank") == rank) & (pl.col("trial") == trial)
            ).sort("start_ns")
            name = (
                label
                + (f" #{trial}" if has_trials else "")
                + (f" r{rank}" if has_ranks else "")
            )
            series_traces, note = _step_series_traces(
                series_steps,
                name,
                color_index,
                {"label": label, "rank": int(rank), "trial": int(trial)},
                options,
                has_rolling_legend=is_crowded,
            )
            traces.extend(series_traces)
            if note:
                notes.append(note)
    durations_ms = steps["duration_ms"].to_numpy()
    slowest_to_median = float(durations_ms.max()) / max(
        float(np.median(durations_ms)), 1e-12
    )
    is_log = options["yscale"] == "log" or (
        options["yscale"] == "auto" and slowest_to_median > 30
    )
    return chart_helpers.single(
        "Step time over steps",
        ir.Panel(
            traces=traces,
            xlabel="step (after warm-up slicing)",
            ylabel="step time (ms)",
            xkind="step",
            yscale="log" if is_log else "linear",
        ),
        notes="; ".join(notes),
    )


def _step_series_traces(
    series_steps: pl.DataFrame,
    name: str,
    color_index: int,
    meta: dict[str, Any],
    options: dict[str, Any],
    *,
    has_rolling_legend: bool,
) -> tuple[list[ir.Trace], str | None]:
    """Builds the traces of one trial and rank of the step-time trace.

    Args:
        series_steps: Steps of one trial and rank, in time order.
        name: Series name shown in the legend.
        color_index: Colour of the group.
        meta: Dashboard filter keys (label, rank, trial).
        options: Resolved options of `step_trace`.
        has_rolling_legend: Whether the rolling median gets a legend entry.

    Returns:
        The step line plus its rolling median, outliers and trend line as
        enabled, and a drift note when the trend is significant and moves by
        more than 2% of the median step over the run.
    """
    x = series_steps["idx"].to_numpy().astype(float)
    y = series_steps["duration_ms"].to_numpy()
    traces = [
        ir.Trace(
            "line",
            x=x.tolist(),
            y=y.tolist(),
            name=name,
            style={
                "ci": color_index,
                "alpha": 0.45 if options["rolling"] else 1.0,
                "width": 1.0,
                "marker": "o",
            },
            meta=meta,
        )
    ]
    if options["rolling"] > 1 and len(y) > 2:
        smoothed = chart_helpers.rolling_median(y, int(options["rolling"]))
        traces.append(
            ir.Trace(
                "line",
                x=x.tolist(),
                y=smoothed.tolist(),
                name=f"{name} rolling median",
                style={
                    "ci": color_index,
                    "width": 2.0,
                    "nolegend": not has_rolling_legend,
                },
                meta=meta,
            )
        )
    if options["outliers"]:
        sigma = chart_helpers.mad_sigma(y)
        is_outlier = (
            y > np.median(y) + options["outliers"] * sigma
            if sigma > 0
            else np.zeros(len(y), bool)
        )
        if is_outlier.any():
            traces.append(
                ir.Trace(
                    "scatter",
                    x=x[is_outlier].tolist(),
                    y=y[is_outlier].tolist(),
                    name=f"{name} outliers",
                    color=theme.BAD,
                    style={"s": 36, "nolegend": True},
                    meta=meta,
                )
            )
    note = None
    fit = chart_helpers.trend(x, y) if options["trend"] else None
    if fit:
        slope, intercept, p_value = fit
        traces.append(
            ir.Trace(
                "line",
                x=[float(x.min()), float(x.max())],
                y=[intercept + slope * x.min(), intercept + slope * x.max()],
                color=theme.NEUTRAL,
                name=f"{name} trend {slope:+.3g} ms/step (p={p_value:.2g})",
                style={"dash": "--", "width": 1.2},
                meta=meta,
            )
        )
        fitted_change_ms = abs(slope) * max(1.0, x.max() - x.min())
        if p_value < 0.05 and fitted_change_ms > 0.02 * np.median(y):
            note = (
                f"{name}: step time drifts {slope:+.3g} ms/step"
                f" (Kendall p={p_value:.2g})"
            )
    return traces, note


@spec.register_plot(
    "step_dist",
    group="steps",
    title="Step-time distribution",
    needs="steps table",
    options={
        "mode": spec.Opt("str", "ecdf", "ecdf or hist.", ("ecdf", "hist")),
        "bins": spec.Opt("int", 30, "Histogram bins."),
    },
)
def step_dist(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots the empirical CDF (or histogram) of step times per group.

    Each group's legend entry carries its p50, p90 and p99.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one series per group.
    """
    steps = ctx.frame("steps", options)
    traces: list[ir.Trace] = []
    labels = chart_helpers.unique_in_order(steps, "label")
    for color_index, label in enumerate(labels):
        durations_ms = steps.filter(pl.col("label") == label)[
            "duration_ms"
        ].to_numpy()
        ms_by_percentile = {
            q: chart_helpers.percentile(durations_ms, q) for q in (50, 90, 99)
        }
        name = (
            f"{label} (p50 {ms_by_percentile[50]:.3g},"
            f" p90 {ms_by_percentile[90]:.3g},"
            f" p99 {ms_by_percentile[99]:.3g} ms)"
        )
        if options["mode"] == "hist":
            traces.append(
                ir.Trace(
                    "hist",
                    y=durations_ms.tolist(),
                    name=name,
                    style={
                        "ci": color_index,
                        "bins": options["bins"],
                        "density": True,
                    },
                    meta={"label": label},
                )
            )
        else:
            sorted_ms = np.sort(durations_ms)
            cumulative_fraction = (np.arange(len(sorted_ms)) + 1) / len(
                sorted_ms
            )
            traces.append(
                ir.Trace(
                    "step",
                    x=sorted_ms.tolist(),
                    y=cumulative_fraction.tolist(),
                    name=name,
                    style={"ci": color_index},
                    meta={"label": label},
                )
            )
    ylabel = "cumulative fraction" if options["mode"] == "ecdf" else "density"
    return chart_helpers.single(
        "Step-time distribution",
        ir.Panel(traces=traces, xlabel="step time (ms)", ylabel=ylabel),
    )


@spec.register_plot(
    "step_compare",
    group="compare",
    title="Paired step-time distributions",
    needs="steps table; best with 2+ groups",
)
def step_compare(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots a box of step times per group.

    Every group after the first is named with the relative change of its
    median against the first group's. The interactive renderer also marks
    the mean.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one box per group.
    """
    steps = ctx.frame("steps", options)
    labels = chart_helpers.unique_in_order(steps, "label")
    baseline_ms = steps.filter(pl.col("label") == labels[0])["duration_ms"]
    baseline_median_ms = float(np.median(baseline_ms.to_numpy()))
    traces = []
    for i, label in enumerate(labels):
        durations_ms = steps.filter(pl.col("label") == label)[
            "duration_ms"
        ].to_numpy()
        change = ""
        if i and baseline_median_ms:
            change_pct = (
                100
                * (np.median(durations_ms) - baseline_median_ms)
                / baseline_median_ms
            )
            change = f" ({change_pct:+.1f}%)"
        traces.append(
            ir.Trace(
                "box",
                y=durations_ms.tolist(),
                name=f"{label}{change}",
                style={"ci": i},
                meta={"label": label},
            )
        )
    return chart_helpers.single(
        "Step-time distributions",
        ir.Panel(traces=traces, ylabel="step time (ms)", legend=False),
    )


def _per_rank_step_values(
    ctx: data.DataContext, options: dict[str, Any], metric: str
) -> pl.DataFrame:
    """Computes a per-step metric for every rank of every group.

    Neither table is filtered by the `where` option, so every step keeps
    its row.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of the calling kind.
        metric: One of `_RANK_METRICS`.

    Returns:
        A frame with columns label, trial, rank, idx and value.
    """
    steps = ctx.frame("steps", {**options, "where": None})
    if metric == "step_ms":
        return steps.select(*_STEP_KEYS, pl.col("duration_ms").alias("value"))
    kernels = ctx.frame("kernels", {**options, "where": None})
    categories = _CATEGORIES_BY_METRIC.get(metric)
    if categories is None:  # idle_pct
        busy_per_step = kernels.group_by(*_STEP_KEYS).agg(
            pl.col("duration_ms").sum().alias("busy")
        )
        steps_with_busy = steps.join(
            busy_per_step, on=list(_STEP_KEYS), how="left"
        ).with_columns(pl.col("busy").fill_null(0.0))
        busy_fraction = (pl.col("busy") / pl.col("duration_ms")).clip(0, 1)
        return steps_with_busy.select(
            *_STEP_KEYS, (100 * (1 - busy_fraction)).alias("value")
        )
    category_ms = (
        kernels.filter(pl.col("category").is_in(categories))
        .group_by(*_STEP_KEYS)
        .agg(pl.col("duration_ms").sum().alias("value"))
    )
    return (
        steps.select(*_STEP_KEYS)
        .join(category_ms, on=list(_STEP_KEYS), how="left")
        .with_columns(pl.col("value").fill_null(0.0))
    )


@spec.register_plot(
    "rank_heatmap",
    group="distributed",
    title="Per-rank, per-step heatmap",
    needs="steps (and kernels) with 2+ ranks",
    options={
        "metric": spec.Opt(
            "str", "step_ms", "Per-step quantity.", _RANK_METRICS
        )
    },
)
def rank_heatmap(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots a per-step quantity as a heatmap over rank x step.

    Stragglers show as hot rows, slow steps as hot columns.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one heatmap per group that has 2+ ranks.

    Raises:
        spec.NotApplicableError: No group has at least two ranks.
    """
    per_step = _per_rank_step_values(ctx, options, options["metric"])
    panels = []
    for label in chart_helpers.unique_in_order(per_step, "label"):
        mean_per_step = (
            per_step.filter(pl.col("label") == label)
            .group_by("rank", "idx")
            .agg(pl.col("value").mean())
        )
        if mean_per_step["rank"].n_unique() < 2:
            continue
        steps_wide = mean_per_step.pivot(
            on="idx", index="rank", values="value"
        ).sort("rank")
        step_columns = sorted(
            (column for column in steps_wide.columns if column != "rank"),
            key=int,
        )
        cells = steps_wide.select(step_columns).to_numpy()
        # Cap the colour scale at p98: a cold first step would otherwise
        # wash out every other cell.
        color_max = float(np.nanpercentile(cells, 98))
        is_clipped = color_max < float(np.nanmax(cells))
        heatmap = ir.Trace(
            "heatmap",
            x=[int(column) for column in step_columns],
            y=[f"rank {rank}" for rank in steps_wide["rank"].to_list()],
            z=cells.tolist(),
            style={"cmap": "magma", "vmax": color_max},
            meta={"label": label},
        )
        colorbar = options["metric"] + (
            " (clipped at p98)" if is_clipped else ""
        )
        panels.append(
            ir.Panel(
                traces=[heatmap],
                title=label,
                xlabel="step",
                colorbar=colorbar,
                legend=False,
                xkind="step",
            )
        )
    if not panels:
        raise spec.NotApplicableError("needs a trace with at least 2 ranks")
    return chart_helpers.grid(
        "Per-rank, per-step " + options["metric"], panels, max_cols=1
    )


@spec.register_plot(
    "straggler",
    group="distributed",
    title="Cross-rank spread per step",
    needs="steps table with 2+ ranks",
    options={
        "metric": spec.Opt(
            "str", "step_ms", "Per-step quantity.", _RANK_METRICS
        )
    },
)
def straggler(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each rank's value per step inside the min-max band of all ranks.

    A wide band is a straggler or a load imbalance.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one panel per group that has 2+ ranks; each
        panel title gives the mean spread.

    Raises:
        spec.NotApplicableError: No group has at least two ranks.
    """
    per_step = _per_rank_step_values(ctx, options, options["metric"])
    panels = []
    for label in chart_helpers.unique_in_order(per_step, "label"):
        group_values = per_step.filter(pl.col("label") == label)
        if group_values["rank"].n_unique() < 2:
            continue
        band = (
            group_values.group_by("idx")
            .agg(
                pl.col("value").min().alias("min_value"),
                pl.col("value").max().alias("max_value"),
            )
            .sort("idx")
        )
        traces = [
            ir.Trace(
                "band",
                x=band["idx"].to_list(),
                y=band["min_value"].to_list(),
                y1=band["max_value"].to_list(),
                name="min-max across ranks",
                color=theme.NEUTRAL,
                meta={"label": label},
            )
        ]
        ranks = sorted(group_values["rank"].unique().to_list())
        for color_index, rank in enumerate(ranks):
            rank_means = (
                group_values.filter(pl.col("rank") == rank)
                .group_by("idx")
                .agg(pl.col("value").mean())
                .sort("idx")
            )
            traces.append(
                ir.Trace(
                    "line",
                    x=rank_means["idx"].to_list(),
                    y=rank_means["value"].to_list(),
                    name=f"rank {rank}",
                    style={"ci": color_index, "width": 1.0},
                    meta={"label": label, "rank": int(rank)},
                )
            )
        spread = band["max_value"] - band["min_value"]
        mean_spread = float(cast(float, spread.mean()))
        panels.append(
            ir.Panel(
                traces=traces,
                title=f"{label} (mean spread {mean_spread:.3g})",
                xlabel="step",
                ylabel=options["metric"],
                xkind="step",
            )
        )
    if not panels:
        raise spec.NotApplicableError("needs a trace with at least 2 ranks")
    return chart_helpers.grid("Cross-rank spread", panels, max_cols=1)
