"""Backend-neutral chart description.

Plot kinds build a :class:`Chart` from data; ``render_mpl`` and ``render_plotly`` draw it. Keeping the marks to a
handful (bar, barh, line, step, scatter, box, hist, heatmap, segment, band, hline, vline, text) is what lets every
kind, including user SQL plots, come out of both backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

MARKS = ("bar", "barh", "line", "step", "scatter", "box", "hist", "heatmap", "segment", "band", "hline", "vline", "text")


@dataclass(slots=True)
class Trace:
    """One drawable series.

    x/y carry the data (``segment``: x=start, x1=end, y=lane; ``band``: y=lower, y1=upper; ``heatmap``: z plus
    x/y tick labels; ``box``/``hist``: y holds the raw values). ``meta`` ({label, rank, device}) lets the dashboard
    filter traces; ``style`` holds mark options (alpha, width, dash, stack, bins, cmap, density, marker...).
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
    meta: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mark not in MARKS:
            raise ValueError(f"Unknown mark '{self.mark}'. Supported: {', '.join(MARKS)}")


@dataclass(slots=True)
class Panel:
    traces: list[Trace] = field(default_factory=list)
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xscale: str = "linear"  # linear | log
    yscale: str = "linear"
    invert_y: bool = False
    legend: bool = True
    xrange: tuple[float, float] | None = None
    colorbar: str = ""  # label of the heatmap/scatter colour bar
    xkind: str = ""  # "step" marks a step axis the dashboard's step-range filter may drive
    stacked: bool = False  # bar traces stack instead of grouping
    overlay: bool = False  # bar traces share one position (waterfalls: floating bars via Trace.y1 as the base)


@dataclass(slots=True)
class Chart:
    title: str
    panels: list[Panel]
    cols: int = 1
    notes: str = ""  # one-line caption shown under the chart (dashboard) / dropped in static files
    size: tuple[float, float] | None = None  # inches; None = the theme default for the panel count
    sharey: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return -(-len(self.panels) // max(1, self.cols))
