"""Plotly renderer for :class:`~cta.viz.ir.Chart` (interactive HTML).

Needs ``pip install cta[viz]``.
"""

from __future__ import annotations

from typing import Any

from cta.viz import ir
from cta.viz import theme as viz_theme

_INSTALL_HINT = "interactive HTML plots need plotly: pip install 'cta[viz]'"


def _graph_objects() -> Any:
    """Imports ``plotly.graph_objects``.

    Raises:
        ImportError: If Plotly is not installed; the message says how to
            install it.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return go


def render(
    chart: ir.Chart,
    *,
    theme: str = "light",
    palette: str | None = None,
    width: float | None = None,
    height: float | None = None,
) -> Any:
    """Draws a chart as a Plotly figure.

    Args:
        chart: The chart to draw.
        theme: ``"dark"`` uses the ``plotly_dark`` template, anything else
            ``plotly_white``.
        palette: Palette name or colour list (see ``theme.palette``).
        width: Figure width in inches (100 px per inch); None uses the chart
            size, else lets Plotly fill the page width.
        height: Figure height in inches (100 px per inch); None uses the
            chart size, else 380 px per grid row plus 60 px.

    Returns:
        The ``plotly.graph_objects.Figure``. Its layout ``meta`` holds the
        chart's meta plus ``step_axes``, the x axes that the dashboard's
        step-range filter drives.

    Raises:
        ImportError: If Plotly is not installed.
        ValueError: If the palette is unknown.
    """
    go = _graph_objects()
    from plotly import subplots

    palette_colors = viz_theme.palette(palette)
    n_panels = len(chart.panels)
    is_grid = n_panels > 1
    titles = [panel.title for panel in chart.panels] if is_grid else None
    fig = (
        subplots.make_subplots(
            rows=chart.rows,
            cols=chart.cols,
            subplot_titles=titles,
            shared_yaxes=chart.sharey,
        )
        if is_grid
        else go.Figure()
    )
    step_axes: list[str] = []
    barmode = "group"
    for i, panel in enumerate(chart.panels):
        row, col = i // chart.cols + 1, i % chart.cols + 1
        cell: dict[str, Any] = {"row": row, "col": col} if is_grid else {}
        color_index = 0
        for trace in panel.traces:
            color_index += _add_trace(
                fig, go, trace, panel, color_index, palette_colors, cell
            )
        axis_suffix = "" if i == 0 else str(i + 1)
        if panel.xkind == "step":
            step_axes.append(f"xaxis{axis_suffix}")
        # Plotly has one bar mode per figure: the last stacked or overlay
        # panel sets it.
        if panel.stacked:
            barmode = "stack"
        elif panel.overlay:
            barmode = "overlay"
        fig.update_xaxes(
            title_text=panel.xlabel,
            type="log" if panel.xscale == "log" else None,
            **cell,
        )
        fig.update_yaxes(
            title_text=panel.ylabel,
            type="log" if panel.yscale == "log" else None,
            autorange="reversed" if panel.invert_y else None,
            **cell,
        )
        if panel.xrange:
            fig.update_xaxes(range=list(panel.xrange), **cell)
    width_px = None
    if width or chart.size:
        width_in = width or (chart.size[0] if chart.size else 6.0 * chart.cols)
        width_px = int(width_in * 100)
    height_px = 380 * chart.rows + 60
    if height or chart.size:
        height_in = height or (
            chart.size[1] if chart.size else 4.2 * chart.rows
        )
        height_px = int(height_in * 100)
    if n_panels == 1 and chart.panels[0].title:
        title = chart.panels[0].title
    else:
        title = chart.title
    fig.update_layout(
        title=title,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        barmode=barmode,
        width=width_px,
        height=height_px,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        meta={**chart.meta, "step_axes": step_axes},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def _colorscale(cmap: str | None) -> str | None:
    """Returns the Plotly colorscale for a Matplotlib colormap name."""
    return viz_theme.PLOTLY_SCALES.get(cmap, cmap) if cmap else None


def _add_trace(
    fig: Any,
    go: Any,
    trace: ir.Trace,
    panel: ir.Panel,
    color_index: int,
    palette_colors: tuple[str, ...],
    cell: dict[str, Any],
) -> int:
    """Adds one trace to the figure.

    Args:
        fig: The Plotly figure.
        go: The ``plotly.graph_objects`` module.
        trace: The trace to add.
        panel: The panel the trace belongs to.
        color_index: Palette index for a trace without its own colour.
        palette_colors: The palette.
        cell: ``row``/``col`` of the panel in a subplot grid; empty for a
            single panel.

    Returns:
        The number of palette colours the trace used (0 or 1).
    """
    used_palette_colors = (
        0 if (trace.color is not None or "ci" in trace.style) else 1
    )
    if trace.color is not None:
        color = trace.color
    else:
        palette_index = int(trace.style.get("ci", color_index))
        color = palette_colors[palette_index % len(palette_colors)]
    style = trace.style
    common: dict[str, Any] = {
        "name": trace.name,
        "meta": trace.meta,
        "showlegend": (
            bool(trace.name) and not style.get("nolegend") and panel.legend
        ),
    }
    if trace.name:
        common["legendgroup"] = trace.name
    mark = trace.mark
    if mark == "bar":
        base = list(trace.y1) if trace.y1 is not None else None
        fig.add_trace(
            go.Bar(
                x=list(trace.x or []),
                y=list(trace.y or []),
                base=base,
                marker_color=color,
                **common,
            ),
            **cell,
        )
    elif mark == "barh":
        fig.add_trace(
            go.Bar(
                x=list(trace.x or []),
                y=list(trace.y or []),
                orientation="h",
                marker_color=color,
                **common,
            ),
            **cell,
        )
    elif mark in ("line", "step"):
        line = {
            "color": color,
            "width": style.get("width", 1.4),
            "dash": _dash(style.get("dash")),
            "shape": "hv" if mark == "step" else "linear",
        }
        fig.add_trace(
            go.Scatter(
                x=list(trace.x or []),
                y=list(trace.y or []),
                mode="lines+markers" if style.get("marker") else "lines",
                line=line,
                opacity=style.get("alpha", 1.0),
                **common,
            ),
            **cell,
        )
    elif mark == "scatter":
        fig.add_trace(
            go.Scatter(
                x=list(trace.x or []),
                y=list(trace.y or []),
                mode="markers",
                marker=_scatter_marker(trace, panel, color),
                text=list(trace.text) if trace.text else None,
                **common,
            ),
            **cell,
        )
    elif mark == "box":
        fig.add_trace(
            go.Box(
                y=list(trace.y or []),
                marker_color=color,
                boxmean=True,
                **common,
            ),
            **cell,
        )
    elif mark == "hist":
        fig.add_trace(
            go.Histogram(
                x=list(trace.y or []),
                nbinsx=style.get("bins", 30),
                opacity=style.get("alpha", 0.55),
                histnorm="probability density" if style.get("density") else "",
                marker_color=color,
                **common,
            ),
            **cell,
        )
    elif mark == "heatmap":
        fig.add_trace(
            go.Heatmap(
                z=[list(row) for row in trace.z or []],
                x=list(trace.x) if trace.x is not None else None,
                y=list(trace.y) if trace.y is not None else None,
                colorscale=_colorscale(style.get("cmap")) or "Viridis",
                zmin=style.get("vmin"),
                zmax=style.get("vmax"),
                colorbar={"title": panel.colorbar},
                meta=trace.meta,
            ),
            **cell,
        )
    elif mark == "segment":
        starts, ends = list(trace.x or []), list(trace.x1 or [])
        lengths = [
            end - start for start, end in zip(starts, ends, strict=False)
        ]
        fig.add_trace(
            go.Bar(
                x=lengths,
                base=starts,
                y=list(trace.y or []),
                orientation="h",
                marker_color=color,
                text=list(trace.text) if trace.text else None,
                hoverinfo="text" if trace.text else "x+y",
                **common,
            ),
            **cell,
        )
    elif mark == "band":
        # An invisible upper edge, then the lower edge filled up to it.
        x = list(trace.x or [])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=list(trace.y1 or []),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                meta=trace.meta,
                hoverinfo="skip",
            ),
            **cell,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=list(trace.y or []),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_rgba(color, style.get("alpha", 0.2)),
                **common,
            ),
            **cell,
        )
    elif mark in ("hline", "vline"):
        if mark == "hline":
            coord = (trace.y or [0])[0]
            add_line = fig.add_hline
        else:
            coord = (trace.x or [0])[0]
            add_line = fig.add_vline
        add_line(
            coord,
            line_dash=_dash(style.get("dash", "--")),
            line_color=color,
            line_width=style.get("width", 1.0),
            annotation_text=trace.name or None,
            **cell,
        )
    elif mark == "text":
        fig.add_trace(
            go.Scatter(
                x=list(trace.x or []),
                y=list(trace.y or []),
                mode="text",
                text=list(trace.text or []),
                textposition="top right",
                showlegend=False,
                meta=trace.meta,
            ),
            **cell,
        )
    return used_palette_colors


def _scatter_marker(
    trace: ir.Trace, panel: ir.Panel, color: Any
) -> dict[str, Any]:
    """Returns the Plotly marker of a scatter trace.

    A trace with one colour value per point gets a colour scale (with a
    colour bar if the panel has a colour bar label); otherwise every point
    has ``color``.
    """
    style = trace.style
    marker: dict[str, Any] = {
        "size": list(trace.size) if trace.size is not None else 7,
        "opacity": style.get("alpha", 0.9),
    }
    if trace.color is not None and not isinstance(trace.color, str):
        marker.update(
            color=list(trace.color),
            colorscale=_colorscale(style.get("cmap")) or "Viridis",
            showscale=bool(panel.colorbar),
            colorbar={"title": panel.colorbar},
        )
    else:
        marker["color"] = color
    return marker


def _dash(dash: Any) -> str:
    """Maps a Matplotlib line style to a Plotly dash; other strings pass."""
    return {
        "-": "solid",
        "--": "dash",
        ":": "dot",
        "-.": "dashdot",
        None: "solid",
    }.get(dash, dash if isinstance(dash, str) else "solid")


def _rgba(hex_color: Any, alpha: float) -> str:
    """Returns ``rgba(r,g,b,alpha)`` for a ``#rrggbb`` colour.

    Any other colour (a name, a short hex form) is returned unchanged,
    without the alpha.
    """
    digits = str(hex_color).lstrip("#")
    if len(digits) != 6 or any(
        char not in "0123456789abcdefABCDEF" for char in digits
    ):
        return str(hex_color)
    red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"
