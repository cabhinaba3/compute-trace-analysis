"""Nsight Compute plot kinds: roofline, occupancy, throughput and stalls."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz import context
from cta.viz import ir
from cta.viz import registry
from cta.viz import theme
from cta.viz.kinds import chart_helpers

# Nsight Compute throughput columns and their legend names, in colour order.
_THROUGHPUT_SERIES = (
    ("sm_throughput_pct", "SM"),
    ("dram_throughput_pct", "DRAM"),
    ("mem_throughput_pct", "memory (max unit)"),
    ("tensor_pipe_pct", "tensor pipe"),
)


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


@registry.register_plot(
    "occupancy",
    group="kernels",
    title="Occupancy vs duration (Nsight Compute)",
    needs="Nsight Compute data",
    options={
        "x": registry.Opt("str", "occupancy_pct", "ncu column on x."),
        "y": registry.Opt("str", "duration_ns", "ncu column on y."),
        "size": registry.Opt(
            "str",
            "dram_bytes",
            "Column that scales the marker (none to disable).",
        ),
        "logy": registry.Opt("bool", True, "Log y axis."),
    },
)
def occupancy(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots achieved occupancy against duration per profiled kernel.

    Points are coloured by kernel family and sized by DRAM traffic (or the
    `size` column); each group gets its own panel.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart of at most two columns with one panel per group.

    Raises:
        NotApplicableError: The x or y column is missing, or no kernel has
            values in both.
    """
    kernels = ctx.frame("ncu_kernels", options)
    x_column, y_column = options["x"], options["y"]
    for column in (x_column, y_column):
        if column not in kernels.columns:
            raise registry.NotApplicableError(
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
        raise registry.NotApplicableError("no kernels with those columns")
    return chart_helpers.grid("Kernel occupancy", panels, max_cols=2)


@registry.register_plot(
    "ncu_util",
    group="kernels",
    title="Throughput utilisation per kernel (Nsight Compute)",
    needs="Nsight Compute data",
    options={
        "top_k": registry.Opt(
            "int", 12, "Kernels shown per group (by duration)."
        )
    },
)
def ncu_util(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots throughput (% of peak) of each group's longest kernels.

    Shows SM, DRAM, memory and tensor-pipe throughput side by side, which
    tells what limits each kernel.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart of at most two columns with one panel per group.

    Raises:
        NotApplicableError: No throughput column has data.
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
        raise registry.NotApplicableError("ncu table has no throughput columns")
    return chart_helpers.grid(
        "Throughput utilisation (% of peak)", panels, max_cols=2
    )


@registry.register_plot(
    "stalls",
    group="kernels",
    title="Warp stall split (Nsight Compute)",
    needs="Nsight Compute data",
    options={
        "top_k": registry.Opt(
            "int", 12, "Kernels shown per group (by duration)."
        )
    },
)
def stalls(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots the memory vs compute share of warp stalls.

    Covers each group's longest kernels as stacked horizontal bars.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart of at most two columns with one panel per group.

    Raises:
        NotApplicableError: Either stall column is missing or all null.
    """
    kernels = ctx.frame("ncu_kernels", options)
    for column in ("stall_mem_pct", "stall_compute_pct"):
        if (
            column not in kernels.columns
            or not kernels[column].drop_nulls().len()
        ):
            raise registry.NotApplicableError("ncu table has no stall data")
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


# Smallest value drawn on the roofline's log axes.
_LOG_AXIS_FLOOR = 1e-3


@registry.register_plot(
    "roofline",
    group="kernels",
    title="Kernel roofline",
    exported=True,
    options={
        "peak_tflops": registry.Opt(
            "float",
            None,
            "Compute roof (TFLOP/s); draws the roofline with peak_bw_gbs.",
        ),
        "peak_bw_gbs": registry.Opt("float", None, "Memory roof (GB/s)."),
    },
    needs="aspect roofline (Nsight Compute data)",
)
def roofline(ctx: context.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots arithmetic intensity against achieved TFLOP/s per kernel.

    For a pair, a line connects each kernel's baseline and candidate points.
    With both peaks given, the roofline itself is drawn.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A one-panel log-log chart.

    Raises:
        NotApplicableError: The report lacks the roofline aspect or
            kernels with FLOP/byte data.
    """
    if not ctx.has_aspect("roofline"):
        raise registry.NotApplicableError(
            "Roofline plot requires 'roofline' aspect"
        )
    roofline_aspect = ctx.require_report().aspects["roofline"]
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
        raise registry.NotApplicableError(
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
