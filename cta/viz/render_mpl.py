"""Matplotlib renderer for :class:`~cta.viz.ir.Chart` (static PDF/SVG/PNG).

Figures use the theme's rcParams, which select Type-42 (TrueType) fonts.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import axes as matplotlib_axes
from matplotlib import figure
import matplotlib.pyplot as plt

from cta.viz import ir
from cta.viz import theme as viz_theme


def render(
    chart: ir.Chart,
    *,
    style: str = "paper_single_col",
    theme: str = "light",
    palette: str | None = None,
    width: float | None = None,
    height: float | None = None,
) -> figure.Figure:
    """Draws a chart as a Matplotlib figure.

    Args:
        chart: The chart to draw.
        style: Per-panel size preset from ``theme.STYLE_SIZES``; unknown
            names use ``paper_single_col``. Ignored when the chart has a size.
        theme: ``"light"`` or ``"dark"``.
        palette: Palette name or colour list (see ``theme.palette``).
        width: Figure width in inches; overrides the computed width.
        height: Figure height in inches; overrides the computed height.

    Returns:
        The figure; the caller saves and closes it.

    Raises:
        ValueError: If the theme or palette is unknown.
    """
    palette_colors = viz_theme.palette(palette)
    panel_width, panel_height = viz_theme.STYLE_SIZES.get(
        style, viz_theme.STYLE_SIZES["paper_single_col"]
    )
    fig_width, fig_height = chart.size or (
        panel_width * chart.cols,
        panel_height * chart.rows,
    )
    fig_width, fig_height = width or fig_width, height or fig_height
    with plt.rc_context(cast(Any, viz_theme.rc_for(theme))):
        fig = plt.figure(
            figsize=(fig_width, fig_height), constrained_layout=True
        )
        axes_grid = (
            fig.subplots(
                chart.rows, chart.cols, squeeze=False, sharey=chart.sharey
            )
            if len(chart.panels) > 1
            else [[fig.add_subplot(111)]]
        )
        flat_axes = [ax for row in axes_grid for ax in row]
        for ax, panel in zip(flat_axes, chart.panels, strict=False):
            _draw_panel(ax, panel, palette_colors, fig)
        for ax in flat_axes[len(chart.panels) :]:
            ax.set_visible(False)
        if len(chart.panels) > 1 or not chart.panels[0].title:
            fig.suptitle(chart.title, fontsize=9)
        if chart.notes:
            fig.text(
                0.01, -0.02, chart.notes, fontsize=5.5, va="top", wrap=True
            )
        return fig


def _color(trace: ir.Trace, index: int, palette_colors: tuple[str, ...]) -> Any:
    """Returns the trace's own colour, else a palette colour.

    The palette index is ``trace.style["ci"]`` if set, else ``index``.
    """
    if trace.color is not None:
        return trace.color
    palette_index = int(trace.style.get("ci", index))
    return palette_colors[palette_index % len(palette_colors)]


def _draw_panel(
    ax: matplotlib_axes.Axes,
    panel: ir.Panel,
    palette_colors: tuple[str, ...],
    fig: figure.Figure,
) -> None:
    """Draws one panel's traces, axis settings and legend on ``ax``."""
    is_labelled = False
    box_traces = [trace for trace in panel.traces if trace.mark == "box"]
    bar_traces = [trace for trace in panel.traces if trace.mark == "bar"]
    barh_traces = [trace for trace in panel.traces if trace.mark == "barh"]
    if box_traces:
        _draw_boxes(ax, box_traces, palette_colors)
    if bar_traces:
        _draw_bars(
            ax,
            bar_traces,
            palette_colors,
            panel.stacked,
            vertical=True,
            overlay=panel.overlay,
        )
        is_labelled = True
    if barh_traces:
        _draw_bars(
            ax,
            barh_traces,
            palette_colors,
            panel.stacked,
            vertical=False,
            overlay=panel.overlay,
        )
        is_labelled = True
    # Traces without their own colour or palette index take the next colour.
    color_index = 0
    for trace in panel.traces:
        if trace.mark in ("box", "bar", "barh"):
            continue
        color = _color(trace, color_index, palette_colors)
        if trace.color is None and "ci" not in trace.style:
            color_index += 1
        has_legend_entry = bool(trace.name) and not trace.style.get("nolegend")
        label = trace.name if has_legend_entry else None
        _draw_trace(ax, trace, color, label, panel, fig)
        is_labelled = is_labelled or label is not None
    _finish_axes(ax, panel, is_labelled)


def _draw_trace(
    ax: matplotlib_axes.Axes,
    trace: ir.Trace,
    color: Any,
    label: str | None,
    panel: ir.Panel,
    fig: figure.Figure,
) -> None:
    """Draws one non-bar, non-box trace.

    Args:
        ax: The panel's axes.
        trace: The trace.
        color: Resolved trace colour.
        label: Legend label; None draws no legend entry.
        panel: The panel (for its colour bar label).
        fig: The figure (for colour bars).
    """
    style = trace.style
    x = (
        np.asarray(trace.x)
        if trace.x is not None and trace.mark != "heatmap"
        else None
    )
    y = (
        np.asarray(trace.y)
        if trace.y is not None and trace.mark != "heatmap"
        else None
    )
    if trace.mark == "line":
        ax.plot(
            x,
            y,
            color=color,
            lw=style.get("width", 1.2),
            ls=style.get("dash", "-"),
            marker=style.get("marker"),
            ms=3,
            label=label,
            alpha=style.get("alpha", 1.0),
        )
    elif trace.mark == "step":
        ax.step(
            x,
            y,
            where="post",
            color=color,
            lw=style.get("width", 1.2),
            label=label,
            alpha=style.get("alpha", 1.0),
        )
    elif trace.mark == "scatter":
        _draw_scatter(ax, trace, x, y, color, label, panel, fig)
    elif trace.mark == "hist":
        ax.hist(
            np.asarray(trace.y, dtype=float),
            bins=style.get("bins", 30),
            density=style.get("density", False),
            color=color,
            alpha=style.get("alpha", 0.55),
            label=label,
        )
    elif trace.mark == "band":
        ax.fill_between(
            x,
            np.asarray(trace.y, dtype=float),
            np.asarray(trace.y1, dtype=float),
            color=color,
            alpha=style.get("alpha", 0.2),
            lw=0,
            label=label,
        )
    elif trace.mark == "segment":
        _draw_segments(ax, trace, color)
    elif trace.mark == "heatmap":
        _draw_heatmap(ax, trace, panel, fig)
    elif trace.mark == "hline":
        ax.axhline(
            float(trace.y[0]),  # type: ignore[index]
            color=color,
            lw=style.get("width", 0.9),
            ls=style.get("dash", "--"),
            label=label,
        )
    elif trace.mark == "vline":
        ax.axvline(
            float(trace.x[0]),  # type: ignore[index]
            color=color,
            lw=style.get("width", 0.9),
            ls=style.get("dash", "--"),
            label=label,
        )
    elif trace.mark == "text":
        for x_value, y_value, text in zip(
            trace.x or [], trace.y or [], trace.text or [], strict=False
        ):
            ax.annotate(
                text,
                (x_value, y_value),
                fontsize=6,
                xytext=(2, 2),
                textcoords="offset points",
                color=color,
            )


def _draw_scatter(
    ax: matplotlib_axes.Axes,
    trace: ir.Trace,
    x: Any,
    y: Any,
    color: Any,
    label: str | None,
    panel: ir.Panel,
    fig: figure.Figure,
) -> None:
    """Draws a scatter trace, with a colour bar for per-point colour values.

    Args:
        ax: The panel's axes.
        trace: The scatter trace.
        x: X values as an array.
        y: Y values as an array.
        color: Resolved trace colour, used unless the trace has one colour
            value per point.
        label: Legend label; None draws no legend entry.
        panel: The panel; its ``colorbar`` label enables the colour bar.
        fig: The figure the colour bar is added to.
    """
    style = trace.style
    sizes = (
        np.asarray(trace.size) if trace.size is not None else style.get("s", 28)
    )
    has_point_colors = trace.color is not None and not isinstance(
        trace.color, str
    )
    points = ax.scatter(
        x,
        y,
        s=sizes,
        c=trace.color if has_point_colors else color,
        cmap=style.get("cmap"),
        alpha=style.get("alpha", 0.9),
        label=label,
        zorder=3,
        marker=style.get("marker", "o"),
    )
    if style.get("cmap") and panel.colorbar:
        fig.colorbar(points, ax=ax, label=panel.colorbar)


def _draw_heatmap(
    ax: matplotlib_axes.Axes,
    trace: ir.Trace,
    panel: ir.Panel,
    fig: figure.Figure,
) -> None:
    """Draws a heatmap trace with a colour bar and thinned tick labels."""
    style = trace.style
    image = ax.imshow(
        np.asarray(trace.z, dtype=float),
        aspect="auto",
        cmap=style.get("cmap", "viridis"),
        origin="upper",
        vmin=style.get("vmin"),
        vmax=style.get("vmax"),
    )
    fig.colorbar(image, ax=ax, label=panel.colorbar)
    # Label at most about 12 columns and 25 rows.
    if trace.x is not None:
        tick_step = max(1, len(trace.x) // 12)
        ax.set_xticks(
            range(0, len(trace.x), tick_step),
            [str(value) for value in list(trace.x)[::tick_step]],
        )
    if trace.y is not None:
        tick_step = max(1, len(trace.y) // 25)
        ax.set_yticks(
            range(0, len(trace.y), tick_step),
            [str(value) for value in list(trace.y)[::tick_step]],
        )
    ax.grid(False)


def _draw_boxes(
    ax: matplotlib_axes.Axes,
    box_traces: list[ir.Trace],
    palette_colors: tuple[str, ...],
) -> None:
    """Draws all box traces of a panel as one box plot."""
    box_values = [np.asarray(trace.y, dtype=float) for trace in box_traces]
    box_artists = ax.boxplot(
        box_values,
        tick_labels=[trace.name or "" for trace in box_traces],
        patch_artist=True,
    )
    for k, patch in enumerate(box_artists["boxes"]):
        patch.set_facecolor(_color(box_traces[k], k, palette_colors))
        patch.set_alpha(0.75)


def _finish_axes(
    ax: matplotlib_axes.Axes, panel: ir.Panel, is_labelled: bool
) -> None:
    """Applies the panel's scales, titles, limits and legend to ``ax``."""
    # set_xscale resets tick labels, so only touch it when needed.
    if panel.xscale != "linear":
        ax.set_xscale(panel.xscale)
    if panel.yscale != "linear":
        ax.set_yscale(panel.yscale)
    ax.set_title(panel.title)
    ax.set_xlabel(panel.xlabel)
    ax.set_ylabel(panel.ylabel)
    if panel.xrange:
        ax.set_xlim(*panel.xrange)
    if panel.invert_y:
        ax.invert_yaxis()
    handles, legend_labels = ax.get_legend_handles_labels()
    if panel.legend and handles and (is_labelled or legend_labels):
        ax.legend(frameon=False, loc="best")


def _draw_bars(
    ax: matplotlib_axes.Axes,
    bars: list[ir.Trace],
    palette_colors: tuple[str, ...],
    stacked: bool,
    *,
    vertical: bool,
    overlay: bool = False,
) -> None:
    """Draws bar traces over shared categories.

    Bars of different traces sit side by side, or share one position when
    ``stacked`` (each trace on top of the previous ones) or ``overlay``. A
    trace's ``y1`` sets its bars' base explicitly.

    Args:
        ax: The panel's axes.
        bars: The bar traces, all vertical or all horizontal.
        palette_colors: Palette for traces without their own colour.
        stacked: If True, stack the traces.
        vertical: True for ``bar`` traces (categories on x), False for
            ``barh`` traces (categories on y).
        overlay: If True, draw all traces at the same position.
    """
    categories = list(
        dict.fromkeys(
            str(value)
            for trace in bars
            for value in (trace.x if vertical else trace.y) or []
        )
    )
    position_by_category = {
        category: i for i, category in enumerate(categories)
    }
    n = len(bars)
    width = 0.6 if (stacked or overlay) else 0.8 / max(1, n)
    stack_base = np.zeros(len(categories))
    for k, trace in enumerate(bars):
        trace_categories = (trace.x if vertical else trace.y) or []
        positions = np.array(
            [position_by_category[str(value)] for value in trace_categories],
            dtype=float,
        )
        values = np.asarray(trace.y if vertical else trace.x, dtype=float)
        color = _color(trace, k, palette_colors)
        offset = 0.0 if (stacked or overlay) else (k - (n - 1) / 2) * width
        base = (
            np.asarray(trace.y1, dtype=float)
            if trace.y1 is not None
            else (stack_base[positions.astype(int)] if stacked else None)
        )
        if vertical:
            ax.bar(
                positions + offset,
                values,
                width=width,
                bottom=base,
                color=color,
                label=trace.name,
            )
        else:
            ax.barh(
                positions + offset,
                values,
                height=width,
                left=base,
                color=color,
                label=trace.name,
            )
        if stacked:
            np.add.at(stack_base, positions.astype(int), values)
    longest_category = max(
        (len(category) for category in categories), default=0
    )
    if vertical and (len(categories) > 4 or longest_category > 9):
        ax.set_xticks(
            range(len(categories)), categories, rotation=35, ha="right"
        )
    elif vertical:
        ax.set_xticks(range(len(categories)), categories)
    else:
        ax.set_yticks(range(len(categories)), categories)


def _draw_segments(
    ax: matplotlib_axes.Axes, trace: ir.Trace, color: Any
) -> None:
    """Draws a segment trace as horizontal lines, one lane per y value.

    Lanes listed in ``trace.style["lanes"]`` map to rows 0, 1, ...; other y
    values are used as numeric positions.
    """
    lanes = trace.style.get("lanes") or []
    position_by_lane = {lane: i for i, lane in enumerate(lanes)}
    starts = np.asarray(trace.x, dtype=float)
    ends = np.asarray(trace.x1, dtype=float)
    y_positions = np.array(
        [position_by_lane.get(value, value) for value in trace.y or []],
        dtype=float,
    )
    ax.hlines(
        y_positions,
        starts,
        ends,
        colors=color,
        lw=trace.style.get("width", 4),
        label=trace.name or "_nolegend_",
    )
    if lanes:
        ax.set_yticks(range(len(lanes)), lanes)
