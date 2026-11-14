"""Backend-neutral chart description.

Plot kinds build a :class:`Chart` from data; ``render_mpl`` and
``render_plotly`` draw it. Keeping the marks to a handful (bar, barh, line,
step, scatter, box, hist, heatmap, segment, band, hline, vline, text) is what
lets every kind, including user SQL plots, come out of both backends.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any

MARKS = (
    "bar",
    "barh",
    "line",
    "step",
    "scatter",
    "box",
    "hist",
    "heatmap",
    "segment",
    "band",
    "hline",
    "vline",
    "text",
)


@dataclasses.dataclass(slots=True)
class Trace:
    """One drawable series.

    ``x``/``y`` carry the data; some marks read them differently: ``segment``
    uses x=start, x1=end, y=lane; ``band`` uses y=lower, y1=upper;
    ``heatmap`` uses ``z`` plus x/y tick labels; ``box`` and ``hist`` hold
    the raw values in ``y``.

    Attributes:
        mark: The mark type, one of ``MARKS``.
        x: X values (or category labels).
        y: Y values (or category labels).
        name: Legend entry; None or empty draws no legend entry.
        color: One colour, or one colour value per point (scatter colour
            scale). None takes the next palette colour.
        x1: Segment end positions.
        y1: Band upper bounds; for bars, the base each bar floats on.
        z: Heatmap cell values, one row per y label.
        size: Scatter marker sizes.
        text: Per-point text (annotations, hover text).
        meta: Filter tags (``label``, ``rank``, ``device``) the dashboard uses
            to hide or show the trace.
        style: Mark options (alpha, width, dash, stack, bins, cmap, density,
            marker, ...); ``ci`` pins the palette index and ``nolegend``
            hides the legend entry.
    """

    mark: str
    x: Sequence[Any] | None = None
    y: Sequence[Any] | None = None
    name: str | None = None
    color: str | Sequence[str] | None = None
    x1: Sequence[Any] | None = None
    y1: Sequence[Any] | None = None
    z: Sequence[Sequence[float]] | None = None
    size: Sequence[float] | None = None
    text: Sequence[str] | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    style: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Rejects marks the renderers do not know."""
        if self.mark not in MARKS:
            raise ValueError(
                f"Unknown mark '{self.mark}'. Supported: {', '.join(MARKS)}"
            )


@dataclasses.dataclass(slots=True)
class Panel:
    """One set of axes and the traces drawn on it.

    Attributes:
        traces: The series to draw, in drawing order.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        xscale: X-axis scale, ``"linear"`` or ``"log"``.
        yscale: Y-axis scale, ``"linear"`` or ``"log"``.
        invert_y: If True, the y axis runs top to bottom.
        legend: If False, no legend is drawn.
        xrange: Fixed x-axis limits; None lets the backend choose.
        colorbar: Label of the heatmap/scatter colour bar.
        xkind: ``"step"`` marks a step axis that the dashboard's step-range
            filter may drive.
        stacked: If True, bar traces stack instead of grouping.
        overlay: If True, bar traces share one position (waterfalls: floating
            bars with ``Trace.y1`` as the base).
    """

    traces: list[Trace] = dataclasses.field(default_factory=list)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xscale: str = "linear"
    yscale: str = "linear"
    invert_y: bool = False
    legend: bool = True
    xrange: tuple[float, float] | None = None
    colorbar: str = ""
    xkind: str = ""
    stacked: bool = False
    overlay: bool = False


@dataclasses.dataclass(slots=True)
class Chart:
    """A figure: a grid of panels with a title.

    Attributes:
        title: Figure title.
        panels: The panels, filled into the grid row by row.
        cols: Number of grid columns.
        notes: One-line caption. The dashboard shows it under the chart and
            Matplotlib output draws it below the axes; Plotly figures leave
            it out.
        size: Figure size in inches; None uses the theme default for the
            panel count.
        sharey: If True, panels share the y axis.
        meta: Extra figure metadata, passed on to the Plotly layout.
    """

    title: str
    panels: list[Panel]
    cols: int = 1
    notes: str = ""
    size: tuple[float, float] | None = None
    sharey: bool = False
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def rows(self) -> int:
        """The number of grid rows needed for all panels."""
        return -(-len(self.panels) // max(1, self.cols))
