"""Entry points: build the chart of a plot kind or spec, then render it.

Matplotlib renders pdf/svg/png files, Plotly renders html files.
"""

from __future__ import annotations

from collections.abc import Mapping
import pathlib
from typing import Any, TYPE_CHECKING

from cta.viz import data
from cta.viz import spec as viz_spec
from cta.viz import theme  # noqa: F401  (re-exported for backwards compatibility)

if TYPE_CHECKING:
    from matplotlib import figure

    from cta.core import models
    from cta.viz import ir

_RENDER_KEYS = ("theme", "palette", "width", "height")


def build_chart(
    ctx: data.DataContext, kind: str, options: Mapping[str, Any] | None = None
) -> tuple[ir.Chart, dict[str, Any]]:
    """Builds the backend-neutral chart of a plot kind.

    Args:
        ctx: Data the plot draws from.
        kind: Plot kind name or alias.
        options: The kind's options as given; defaults fill the rest.

    Returns:
        The chart and the resolved options. A ``title`` option replaces the
        chart title (and the panel title of a single-panel chart).

    Raises:
        ValueError: If the kind or an option is unknown, or an option value
            cannot be coerced.
        viz_spec.NotApplicableError: If the kind cannot be drawn from this data.
    """
    plot_kind = viz_spec.get_kind(kind)
    resolved = viz_spec.resolve_options(plot_kind, options or {})
    chart = plot_kind.build(ctx, resolved)
    if resolved.get("title"):
        chart.title = resolved["title"]
        if len(chart.panels) == 1:
            chart.panels[0].title = resolved["title"]
    return chart, resolved


def to_matplotlib(
    chart: ir.Chart, options: Mapping[str, Any], style: str = "paper_single_col"
) -> figure.Figure:
    """Draws a chart as a Matplotlib figure.

    Args:
        chart: The chart to draw.
        options: Resolved options; ``theme``, ``palette``, ``width`` and
            ``height`` are used.
        style: Size preset from ``theme.STYLE_SIZES``.

    Returns:
        The figure.
    """
    from cta.viz import render_mpl

    return render_mpl.render(
        chart,
        style=style,
        theme=options.get("theme") or "light",
        palette=options.get("palette"),
        width=options.get("width"),
        height=options.get("height"),
    )


def to_plotly(chart: ir.Chart, options: Mapping[str, Any]) -> Any:
    """Draws a chart as a Plotly figure.

    Args:
        chart: The chart to draw.
        options: Resolved options; ``theme``, ``palette``, ``width`` and
            ``height`` are used.

    Returns:
        The ``plotly.graph_objects.Figure``.
    """
    from cta.viz import render_plotly

    return render_plotly.render(
        chart,
        theme=options.get("theme") or "light",
        palette=options.get("palette"),
        width=options.get("width"),
        height=options.get("height"),
    )


def write_html(fig: Any, path: pathlib.Path, *, cdn: bool = False) -> None:
    """Writes a Plotly figure as a standalone HTML file.

    Args:
        fig: The Plotly figure.
        path: Output file; missing parent directories are created.
        cdn: If True, the page loads plotly.js from the CDN instead of
            embedding it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path), include_plotlyjs="cdn" if cdn else True, full_html=True
    )


def render_chart(
    chart: ir.Chart,
    options: Mapping[str, Any],
    path: pathlib.Path | None,
    *,
    style: str = "paper_single_col",
    dpi: int = 300,
    cdn: bool = False,
) -> Any:
    """Draws a built chart and writes it to ``path``, if given.

    A ``.html`` path gives an interactive Plotly file; any other path a
    Matplotlib file in the format of its suffix.

    Args:
        chart: The chart to draw.
        options: Resolved options (theme, palette, width, height).
        path: Output file; None only returns the Matplotlib figure.
        style: Size preset for Matplotlib output.
        dpi: Resolution of raster Matplotlib output.
        cdn: For HTML output, load plotly.js from the CDN.

    Returns:
        The Plotly figure for HTML output, else the Matplotlib figure
        (already closed when it was written to a file).
    """
    if path is not None and path.suffix.lower() == ".html":
        plotly_figure = to_plotly(chart, options)
        write_html(plotly_figure, path, cdn=cdn)
        return plotly_figure
    import matplotlib.pyplot as plt

    fig = to_matplotlib(chart, options, style)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def render_plot(
    report: models.ComparisonReport,
    *,
    kind: str = "breakdown",
    path: pathlib.Path | None = None,
    style: str = "paper_single_col",
    dpi: int = 300,
    **options: Any,
) -> Any:
    """Renders one plot kind of a report (see ``cta plot --list-kinds``).

    Args:
        report: The report to plot.
        kind: Plot kind name or alias.
        path: Output file (``.html`` for Plotly); None returns the figure
            only.
        style: Size preset for Matplotlib output.
        dpi: Resolution of raster Matplotlib output.
        **options: The kind's declared options.

    Returns:
        The rendered figure (see :func:`render_chart`).

    Raises:
        ValueError: If the kind or an option is invalid.
        viz_spec.NotApplicableError: If the kind cannot be drawn for this
            report.
    """
    chart, resolved = build_chart(data.DataContext(report), kind, options)
    return render_chart(chart, resolved, path, style=style, dpi=dpi)


def render_spec(
    ctx: data.DataContext,
    spec: viz_spec.PlotSpec,
    path: pathlib.Path | None,
    *,
    style: str = "paper_single_col",
    dpi: int = 300,
) -> Any:
    """Renders one declarative plot spec.

    Args:
        ctx: Data the plot draws from.
        spec: The plot spec.
        path: Output file (``.html`` for Plotly); None returns the figure
            only.
        style: Size preset for Matplotlib output.
        dpi: Resolution of raster Matplotlib output.

    Returns:
        The rendered figure (see :func:`render_chart`).

    Raises:
        ValueError: If the kind or an option is invalid.
        viz_spec.NotApplicableError: If the kind cannot be drawn from this data.
    """
    chart, resolved = build_chart(ctx, spec.kind, spec.options)
    return render_chart(chart, resolved, path, style=style, dpi=dpi)
