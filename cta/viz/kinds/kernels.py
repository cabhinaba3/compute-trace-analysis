"""Kernel-level plot kinds.

Covers the kernel Pareto, family share, baseline-vs-candidate scatter,
per-kernel drift and the Nsight Compute views (occupancy, throughput, stalls).
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

_GROUP_BY_COLUMNS = ("canonical_name", "name", "op_family", "scope", "category")
# Nsight Compute throughput columns and their legend names, in colour order.
_THROUGHPUT_SERIES = (
    ("sm_throughput_pct", "SM"),
    ("dram_throughput_pct", "DRAM"),
    ("mem_throughput_pct", "memory (max unit)"),
    ("tensor_pipe_pct", "tensor pipe"),
)


def _group_by_option(default: str) -> spec.Opt:
    """Declares the `by` option that picks the column to group kernels by."""
    return spec.Opt(
        "str", default, "Column to group kernels by.", _GROUP_BY_COLUMNS
    )


@spec.register_plot(
    "pareto",
    group="timeline",
    title="Kernel Pareto (share and cumulative share)",
    needs="kernels table",
    options={
        "by": _group_by_option("canonical_name"),
        "top_k": spec.Opt("int", 15, "Entities shown."),
    },
)
def pareto(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
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


@spec.register_plot(
    "family_share",
    group="overview",
    title="Time by kernel family / category / scope",
    needs="kernels table",
    options={
        "by": _group_by_option("op_family"),
        "normalize": spec.Opt(
            "bool", False, "Show 100% stacked shares instead of ms."
        ),
        "top_k": spec.Opt("int", 8, "Entities kept; the rest become 'other'."),
    },
)
def family_share(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
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


@spec.register_plot(
    "kernel_scatter",
    group="compare",
    title="Kernel time: baseline vs candidate",
    needs="kernels table, exactly 2 groups",
    options={
        "by": _group_by_option("canonical_name"),
        "label_top": spec.Opt("int", 6, "Annotate this many biggest movers."),
    },
)
def kernel_scatter(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each kernel's total time in the baseline against the candidate.

    One point per kernel, baseline on x and candidate on y; points on the
    diagonal are unchanged. The biggest movers are annotated.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel log-log chart.

    Raises:
        spec.NotApplicableError: The report does not have exactly two groups.
    """
    kernels = ctx.frame("kernels", options)
    labels = chart_helpers.unique_in_order(kernels, "label")
    if len(labels) != 2:
        raise spec.NotApplicableError("needs exactly two groups")
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


@spec.register_plot(
    "kernel_drift",
    group="drift",
    title="Per-kernel duration drift across steps",
    needs="kernels table with >= 4 steps",
    options={
        "top_k": spec.Opt("int", 20, "Kernels shown (by total time)."),
        "by": _group_by_option("canonical_name"),
    },
)
def kernel_drift(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
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
        spec.NotApplicableError: No group has kernels in at least four steps.
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
        raise spec.NotApplicableError("needs at least 4 steps")
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


def _longest_kernels(
    kernels: pl.DataFrame, label: str, count: int
) -> pl.DataFrame:
    """Returns the `count` longest profiled kernels of one group.

    Kernel ids restart at 0 in every report, so groups are never pooled.
    """
    return (
        kernels.filter(pl.col("label") == label)
        .sort("duration_ns", descending=True)
        .unique(["trial", "kernel_id"], keep="first", maintain_order=True)
        .head(count)
    )


def _ranked_kernel_names(kernels: pl.DataFrame) -> list[str]:
    """Labels bars by truncated kernel name and rank, so names stay unique."""
    return [
        f"{str(name)[:26]} #{rank}"
        for rank, name in enumerate(kernels["canonical_name"].to_list())
    ]


def _marker_sizes(
    kernels: pl.DataFrame, size_column: str
) -> list[float] | None:
    """Scales marker sizes to 6..36 by a column.

    Returns None when sizing is off (``none``), or the column is missing or
    all null.
    """
    if (
        size_column == "none"
        or size_column not in kernels.columns
        or not kernels[size_column].drop_nulls().len()
    ):
        return None
    magnitudes = kernels[size_column].fill_null(0).to_numpy().astype(float)
    return (6 + 30 * magnitudes / max(magnitudes.max(), 1e-9)).tolist()


@spec.register_plot(
    "occupancy",
    group="kernels",
    title="Occupancy vs duration (Nsight Compute)",
    needs="Nsight Compute data",
    options={
        "x": spec.Opt("str", "occupancy_pct", "ncu column on x."),
        "y": spec.Opt("str", "duration_ns", "ncu column on y."),
        "size": spec.Opt(
            "str",
            "dram_bytes",
            "Column that scales the marker (none to disable).",
        ),
        "logy": spec.Opt("bool", True, "Log y axis."),
    },
)
def occupancy(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots achieved occupancy against duration per profiled kernel.

    Points are coloured by kernel family and sized by DRAM traffic (or the
    `size` column); each group gets its own panel.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart of at most two columns with one panel per group.

    Raises:
        spec.NotApplicableError: The x or y column is missing, or no kernel has
            values in both.
    """
    kernels = ctx.frame("ncu_kernels", options)
    x_column, y_column = options["x"], options["y"]
    for column in (x_column, y_column):
        if column not in kernels.columns:
            raise spec.NotApplicableError(
                f"ncu table has no column '{column}':"
                f" {', '.join(kernels.columns)}"
            )
    families = sorted(kernels["op_family"].unique().to_list())
    panels = []
    labels = chart_helpers.unique_in_order(kernels, "label")
    for panel_index, label in enumerate(labels):
        traces = []
        for family_index, family in enumerate(families):
            family_kernels = kernels.filter(
                (pl.col("label") == label) & (pl.col("op_family") == family)
            ).drop_nulls([x_column, y_column])
            if family_kernels.is_empty():
                continue
            kernel_names = [
                str(name)[:40]
                for name in family_kernels["canonical_name"].to_list()
            ]
            traces.append(
                ir.Trace(
                    "scatter",
                    x=family_kernels[x_column].to_list(),
                    y=family_kernels[y_column].to_list(),
                    name=family,
                    size=_marker_sizes(family_kernels, options["size"]),
                    style={"ci": family_index, "nolegend": panel_index > 0},
                    text=kernel_names,
                    meta={"label": label},
                )
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=label,
                xlabel=x_column,
                ylabel=y_column,
                yscale="log" if options["logy"] else "linear",
            )
        )
    if not any(panel.traces for panel in panels):
        raise spec.NotApplicableError("no kernels with those columns")
    return chart_helpers.grid("Kernel occupancy", panels, max_cols=2)


@spec.register_plot(
    "ncu_util",
    group="kernels",
    title="Throughput utilisation per kernel (Nsight Compute)",
    needs="Nsight Compute data",
    options={
        "top_k": spec.Opt("int", 12, "Kernels shown per group (by duration).")
    },
)
def ncu_util(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots throughput (% of peak) of each group's longest kernels.

    Shows SM, DRAM, memory and tensor-pipe throughput side by side, which
    tells what limits each kernel.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart of at most two columns with one panel per group.

    Raises:
        spec.NotApplicableError: No throughput column has data.
    """
    kernels = ctx.frame("ncu_kernels", options)
    panels = []
    labels = chart_helpers.unique_in_order(kernels, "label")
    for panel_index, label in enumerate(labels):
        longest = _longest_kernels(kernels, label, int(options["top_k"]))
        kernel_names = _ranked_kernel_names(longest)
        traces = []
        for series_index, (column, series_name) in enumerate(
            _THROUGHPUT_SERIES
        ):
            if (
                column not in longest.columns
                or not longest[column].drop_nulls().len()
            ):
                continue
            traces.append(
                ir.Trace(
                    "barh",
                    x=longest[column].fill_null(0).to_list(),
                    y=kernel_names,
                    name=series_name,
                    style={"ci": series_index, "nolegend": panel_index > 0},
                    meta={"label": label},
                )
            )
        panels.append(
            ir.Panel(
                traces=traces, title=label, xlabel="% of peak", invert_y=True
            )
        )
    if not any(panel.traces for panel in panels):
        raise spec.NotApplicableError("ncu table has no throughput columns")
    return chart_helpers.grid(
        "Throughput utilisation (% of peak)", panels, max_cols=2
    )


@spec.register_plot(
    "stalls",
    group="kernels",
    title="Warp stall split (Nsight Compute)",
    needs="Nsight Compute data",
    options={
        "top_k": spec.Opt("int", 12, "Kernels shown per group (by duration).")
    },
)
def stalls(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots the memory vs compute share of warp stalls.

    Covers each group's longest kernels as stacked horizontal bars.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart of at most two columns with one panel per group.

    Raises:
        spec.NotApplicableError: Either stall column is missing or all null.
    """
    kernels = ctx.frame("ncu_kernels", options)
    for column in ("stall_mem_pct", "stall_compute_pct"):
        if (
            column not in kernels.columns
            or not kernels[column].drop_nulls().len()
        ):
            raise spec.NotApplicableError("ncu table has no stall data")
    panels = []
    for label in chart_helpers.unique_in_order(kernels, "label"):
        longest = _longest_kernels(kernels, label, int(options["top_k"]))
        kernel_names = _ranked_kernel_names(longest)
        memory_stalls = ir.Trace(
            "barh",
            x=longest["stall_mem_pct"].fill_null(0).to_list(),
            y=kernel_names,
            name="memory stalls",
            style={"ci": 1},
            meta={"label": label},
        )
        compute_stalls = ir.Trace(
            "barh",
            x=longest["stall_compute_pct"].fill_null(0).to_list(),
            y=kernel_names,
            name="compute stalls",
            style={"ci": 0},
            meta={"label": label},
        )
        panels.append(
            ir.Panel(
                traces=[memory_stalls, compute_stalls],
                title=label,
                xlabel="% of stall cycles",
                invert_y=True,
                stacked=True,
            )
        )
    return chart_helpers.grid("Warp stalls", panels, max_cols=2)
