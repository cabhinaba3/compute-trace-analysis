"""Kernel-level plot kinds: Pareto, family share, top-k, scatter and drift."""

from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz import theme
from cta.viz.kinds import chart_helpers

if TYPE_CHECKING:
    from cta import models

_GROUP_BY_COLUMNS = ("canonical_name", "name", "op_family", "scope", "category")


def _group_by_option(default: str) -> registry.Opt:
    """Declares the `by` option that picks the column to group kernels by."""
    return registry.Opt(
        "str", default, "Column to group kernels by.", _GROUP_BY_COLUMNS
    )


@registry.register_plot(
    "pareto",
    group="timeline",
    title="Kernel Pareto (share and cumulative share)",
    needs="kernels table",
    options={
        "by": _group_by_option("canonical_name"),
        "top_k": registry.Opt("int", 15, "Entities shown."),
    },
)
def pareto(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each entity's share of GPU time with the cumulative share line.

    Entities (kernels, families or scopes) are sorted by time; each group gets
    its own panel.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one panel per group.
    """
    kernels = ctx.frame("kernels", options)
    entity_column = options["by"]
    panels = []
    for label in chart_helpers.unique_in_order(kernels, "label"):
        per_entity = (
            kernels.filter(pl.col("label") == label)
            .group_by(entity_column)
            .agg(pl.col("duration_ms").sum())
            .sort("duration_ms", descending=True)
        )
        total_ms = float(per_entity["duration_ms"].sum())
        top = per_entity.head(int(options["top_k"]))
        names = [str(name)[:36] for name in top[entity_column].to_list()]
        share_pct = (100 * top["duration_ms"] / total_ms).to_list()
        cumulative_pct = np.cumsum(share_pct).tolist()
        share_bars = ir.Trace(
            "bar",
            x=names,
            y=share_pct,
            name="share of GPU time",
            style={"ci": 0},
            meta={"label": label},
        )
        cumulative_line = ir.Trace(
            "line",
            x=names,
            y=cumulative_pct,
            name="cumulative",
            color=theme.NEUTRAL,
            style={"marker": "o"},
            meta={"label": label},
        )
        panels.append(
            ir.Panel(
                traces=[share_bars, cumulative_line],
                title=f"{label} ({total_ms:.3g} ms total)",
                ylabel="% of GPU time",
                xlabel=entity_column,
            )
        )
    return chart_helpers.grid("Kernel Pareto", panels, max_cols=1)


@registry.register_plot(
    "family_share",
    group="overview",
    title="Time by kernel family / category / scope",
    needs="kernels table",
    options={
        "by": _group_by_option("op_family"),
        "normalize": registry.Opt(
            "bool", False, "Show 100% stacked shares instead of ms."
        ),
        "top_k": registry.Opt(
            "int", 8, "Entities kept; the rest become 'other'."
        ),
    },
)
def family_share(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots stacked bars of GPU time per step for each group.

    Bars are split by the chosen column; entities outside the top K are
    merged into ``other``.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel chart with one bar per group.
    """
    kernels = ctx.frame("kernels", options)
    entity_column = options["by"]
    # Only the kernels are filtered by `where`; every step of a group counts.
    steps_per_label = (
        ctx.frame("steps", {**options, "where": None})
        .group_by("label", "trial", "rank")
        .len()
        .group_by("label")
        .agg(pl.col("len").sum().alias("n_steps"))
    )
    per_entity = (
        kernels.group_by("label", entity_column)
        .agg(pl.col("duration_ms").sum())
        .join(steps_per_label, on="label")
        .with_columns(
            (pl.col("duration_ms") / pl.col("n_steps")).alias("ms_per_step")
        )
    )
    top_entities = (
        per_entity.group_by(entity_column)
        .agg(pl.col("ms_per_step").sum())
        .sort("ms_per_step", descending=True)
        .head(int(options["top_k"]))[entity_column]
        .to_list()
    )
    entity_or_other = (
        pl.when(pl.col(entity_column).is_in(top_entities))
        .then(pl.col(entity_column))
        .otherwise(pl.lit("other"))
        .alias(entity_column)
    )
    per_shown_entity = (
        per_entity.with_columns(entity_or_other)
        .group_by("label", entity_column)
        .agg(pl.col("ms_per_step").sum())
    )
    labels = chart_helpers.unique_in_order(kernels, "label")
    if options["normalize"]:
        ms_per_label = [
            per_shown_entity.filter(pl.col("label") == label)["ms_per_step"]
            for label in labels
        ]
        total_per_label = [label_ms.sum() or 1.0 for label_ms in ms_per_label]
    traces = []
    for i, entity in enumerate([*top_entities, "other"]):
        entity_rows = per_shown_entity.filter(pl.col(entity_column) == entity)
        if entity_rows.is_empty():
            continue
        ms_by_label = {
            row["label"]: row["ms_per_step"]
            for row in entity_rows.iter_rows(named=True)
        }
        heights = [ms_by_label.get(label, 0.0) for label in labels]
        if options["normalize"]:
            heights = [
                100 * height / total
                for height, total in zip(heights, total_per_label, strict=True)
            ]
        traces.append(
            ir.Trace(
                "bar", x=labels, y=heights, name=str(entity), style={"ci": i}
            )
        )
    if options["normalize"]:
        ylabel = "% of GPU time"
    else:
        ylabel = "GPU time per step (ms)"
    return chart_helpers.single(
        "GPU time by " + entity_column,
        ir.Panel(traces=traces, ylabel=ylabel, stacked=True),
    )


@registry.register_plot(
    "kernel_scatter",
    group="compare",
    title="Kernel time: baseline vs candidate",
    needs="kernels table, exactly 2 groups",
    options={
        "by": _group_by_option("canonical_name"),
        "label_top": registry.Opt(
            "int", 6, "Annotate this many biggest movers."
        ),
    },
)
def kernel_scatter(
    ctx: context.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots each kernel's total time in the baseline against the candidate.

    One point per kernel, baseline on x and candidate on y; points on the
    diagonal are unchanged. The biggest movers are annotated.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel log-log chart.

    Raises:
        NotApplicableError: The report does not have exactly two groups.
    """
    kernels = ctx.frame("kernels", options)
    labels = chart_helpers.unique_in_order(kernels, "label")
    if len(labels) != 2:
        raise registry.NotApplicableError("needs exactly two groups")
    entity_column = options["by"]
    baseline_ms = _total_ms_per_entity(
        kernels, labels[0], entity_column, "baseline_ms"
    )
    candidate_ms = _total_ms_per_entity(
        kernels, labels[1], entity_column, "candidate_ms"
    )
    per_entity = (
        baseline_ms.join(
            candidate_ms, on=entity_column, how="full", coalesce=True
        )
        .fill_null(0.0)
        .with_columns(
            (pl.col("candidate_ms") - pl.col("baseline_ms"))
            .abs()
            .alias("abs_change_ms")
        )
    )
    # Log axes cannot show zero: a kernel missing from one side is drawn at
    # half the smallest positive time.
    smallest_ms = min(
        _smallest_positive(per_entity["baseline_ms"]),
        _smallest_positive(per_entity["candidate_ms"]),
    )
    floor_ms = max(1e-6, float(smallest_ms) / 2)
    x = np.maximum(per_entity["baseline_ms"].to_numpy(), floor_ms)
    y = np.maximum(per_entity["candidate_ms"].to_numpy(), floor_ms)
    axis_min = float(min(x.min(), y.min()))
    axis_max = float(max(x.max(), y.max()))
    biggest_movers = per_entity.sort("abs_change_ms", descending=True).head(
        int(options["label_top"])
    )
    traces = [
        ir.Trace(
            "line",
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            color=theme.NEUTRAL,
            name="no change",
            style={"dash": "--"},
        ),
        ir.Trace(
            "scatter",
            x=x.tolist(),
            y=y.tolist(),
            name="kernels",
            style={"ci": 0},
            text=[str(name) for name in per_entity[entity_column].to_list()],
        ),
        ir.Trace(
            "text",
            x=np.maximum(
                biggest_movers["baseline_ms"].to_numpy(), floor_ms
            ).tolist(),
            y=np.maximum(
                biggest_movers["candidate_ms"].to_numpy(), floor_ms
            ).tolist(),
            text=[
                str(name)[:24]
                for name in biggest_movers[entity_column].to_list()
            ],
        ),
    ]
    return chart_helpers.single(
        "Kernel time (ms): " + f"{labels[0]} vs {labels[1]}",
        ir.Panel(
            traces=traces,
            xlabel=f"{labels[0]} (ms, floor {floor_ms:.2g})",
            ylabel=f"{labels[1]} (ms)",
            xscale="log",
            yscale="log",
        ),
    )


def _smallest_positive(values: pl.Series) -> Any:
    """Returns the smallest positive value, or 1e-3 when there is none."""
    return cast(Any, values.filter(values > 0).min()) or 1e-3


def _total_ms_per_entity(
    kernels: pl.DataFrame, label: str, entity_column: str, alias: str
) -> pl.DataFrame:
    """Sums one group's kernel time per entity into a column named `alias`."""
    return (
        kernels.filter(pl.col("label") == label)
        .group_by(entity_column)
        .agg(pl.col("duration_ms").sum().alias(alias))
    )


@registry.register_plot(
    "kernel_drift",
    group="drift",
    title="Per-kernel duration drift across steps",
    needs="kernels table with >= 4 steps",
    options={
        "top_k": registry.Opt("int", 20, "Kernels shown (by total time)."),
        "by": _group_by_option("canonical_name"),
    },
)
def kernel_drift(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each kernel's mean duration per step relative to its own median.

    A kernel drifting up or down shows as a gradient along its row; rows with
    a significant trend are flagged with their drift per step.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-column chart with one heatmap per group; its notes list up to
        six drifting kernels.

    Raises:
        NotApplicableError: No group has kernels in at least four steps.
    """
    kernels = ctx.frame("kernels", options).filter(pl.col("idx").is_not_null())
    entity_column = options["by"]
    panels, notes = [], []
    for label in chart_helpers.unique_in_order(kernels, "label"):
        drift = _drift_heatmap(
            kernels.filter(pl.col("label") == label),
            label,
            entity_column,
            int(options["top_k"]),
        )
        if drift is None:
            continue
        panel, drift_notes = drift
        panels.append(panel)
        notes.extend(drift_notes)
    if not panels:
        raise registry.NotApplicableError("needs at least 4 steps")
    return chart_helpers.grid(
        "Per-kernel duration drift",
        panels,
        max_cols=1,
        notes="; ".join(notes[:6]),
    )


def _drift_heatmap(
    group_kernels: pl.DataFrame, label: str, entity_column: str, top_k: int
) -> tuple[ir.Panel, list[str]] | None:
    """Builds one group's per-kernel drift heatmap.

    Args:
        group_kernels: Kernels of one group that belong to a step.
        label: The group label.
        entity_column: Column that names a row (kernel, family, ...).
        top_k: Number of entities kept, by total time.

    Returns:
        The heatmap panel and a note per drifting entity, or None when the
        kept entities span fewer than four steps.
    """
    top_entities = (
        group_kernels.group_by(entity_column)
        .agg(pl.col("duration_ms").sum())
        .sort("duration_ms", descending=True)
        .head(top_k)[entity_column]
        .to_list()
    )
    mean_per_step = (
        group_kernels.filter(pl.col(entity_column).is_in(top_entities))
        .group_by(entity_column, "idx")
        .agg(pl.col("duration_ms").mean().alias("mean_ms"))
    )
    if mean_per_step["idx"].n_unique() < 4:
        return None
    steps_wide = mean_per_step.pivot(
        on="idx", index=entity_column, values="mean_ms"
    )
    step_columns = sorted(
        (column for column in steps_wide.columns if column != entity_column),
        key=int,
    )
    step_indices = [int(column) for column in step_columns]
    relative_rows, row_labels, notes = [], [], []
    for row in steps_wide.sort(entity_column).iter_rows(named=True):
        means_ms = np.array(
            [
                row[column] if row[column] is not None else np.nan
                for column in step_columns
            ],
            dtype=float,
        )
        has_value = ~np.isnan(means_ms)
        median_ms = float(np.nanmedian(means_ms))
        if median_ms > 0:
            relative_rows.append((means_ms / median_ms).tolist())
        else:
            relative_rows.append([1.0] * len(means_ms))
        fit = None
        if has_value.sum() >= 4:
            fit = chart_helpers.trend(
                np.array(step_indices, float)[has_value], means_ms[has_value]
            )
        flag = _drift_flag(fit, median_ms, len(step_columns))
        if flag:
            notes.append(f"{label}: {str(row[entity_column])[:30]}{flag}")
        row_labels.append(str(row[entity_column])[:30] + flag)
    heatmap = ir.Trace(
        "heatmap",
        x=step_indices,
        y=row_labels,
        z=relative_rows,
        style={"cmap": "RdBu_r", "vmin": 0.5, "vmax": 1.5},
        meta={"label": label},
    )
    panel = ir.Panel(
        traces=[heatmap],
        title=label,
        xlabel="step",
        colorbar="duration / kernel median",
        legend=False,
        xkind="step",
    )
    return panel, notes


def _drift_flag(
    fit: tuple[float, float, float] | None, median_ms: float, n_steps: int
) -> str:
    """Formats a significant drift as `` +x.x%/step``, else returns ``""``.

    Args:
        fit: Theil-Sen slope, intercept and Kendall p-value, if fitted.
        median_ms: The entity's median duration per step.
        n_steps: Width of the step window.

    Returns:
        The flag when p < 0.05 and the fitted change over the window exceeds
        5% of the median.
    """
    if not (fit and median_ms > 0):
        return ""
    slope, _, p_value = fit
    if p_value < 0.05 and abs(slope) * n_steps / median_ms > 0.05:
        return f" {100 * slope / median_ms:+.1f}%/step"
    return ""


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


@registry.register_plot(
    "topk",
    group="compare",
    title="Top contributors / hotspots",
    aliases=("tornado", "top_k", "hotspots"),
    exported=True,
    options={
        "top_k": registry.Opt("int", 10, "How many entities."),
        "table": registry.Opt(
            "str", None, "Table of kernel_diff, e.g. by_kernel or by_scope."
        ),
    },
    needs="a per-entity report table (kernel_diff, layer_fusion, ...)",
)
def topk(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
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
        NotApplicableError: The report has more than two groups, or no
            per-entity table has the needed column.
    """
    report = ctx.require_report()
    if len(report.groups) not in (1, 2):
        raise registry.NotApplicableError(
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
    report: models.Report,
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
        NotApplicableError: No candidate table qualifies.
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
    raise registry.NotApplicableError(
        f"No entity breakdown table with '{value_column}' available for"
        " Top-K / tornado plot"
    )
