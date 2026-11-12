"""Entry points: build a chart for a plot kind/spec and render it (Matplotlib for pdf/svg/png, Plotly for html)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cta.viz.data import DataContext
from cta.viz.spec import PlotSpec, get_kind, resolve_options
from cta.viz.theme import OKABE_ITO, PAPER_RC, STYLE_SIZES  # noqa: F401  (re-exported for backwards compatibility)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from cta.core.models import ComparisonReport
    from cta.viz.ir import Chart

_RENDER_KEYS = ("theme", "palette", "width", "height")


def build_chart(ctx: DataContext, kind: str, options: Mapping[str, Any] | None = None) -> tuple[Chart, dict[str, Any]]:
    """Resolve options (defaults, coercion, unknown-key errors) and build the backend-neutral chart."""
    k = get_kind(kind)
    opts = resolve_options(k, options or {})
    chart = k.build(ctx, opts)
    if opts.get("title"):
        chart.title = opts["title"]
        if len(chart.panels) == 1:
            chart.panels[0].title = opts["title"]
    return chart, opts


def to_matplotlib(chart: Chart, opts: Mapping[str, Any], style: str = "paper_single_col") -> Figure:
    from cta.viz import render_mpl

    return render_mpl.render(chart, style=style, theme=opts.get("theme") or "light", colors=opts.get("palette"),
                             width=opts.get("width"), height=opts.get("height"))  # fmt: skip


def to_plotly(chart: Chart, opts: Mapping[str, Any]) -> Any:
    from cta.viz import render_plotly

    return render_plotly.render(chart, theme=opts.get("theme") or "light", colors=opts.get("palette"), width=opts.get("width"), height=opts.get("height"))


def write_html(fig: Any, path: Path, *, cdn: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn" if cdn else True, full_html=True)


def render_chart(chart: Chart, opts: Mapping[str, Any], path: Path | None, *, style: str = "paper_single_col", dpi: int = 300, cdn: bool = False) -> Any:
    """Draw a built chart. A ``.html`` path gives an interactive Plotly file, anything else a Matplotlib figure."""
    if path is not None and path.suffix.lower() == ".html":
        fig = to_plotly(chart, opts)
        write_html(fig, path, cdn=cdn)
        return fig
    import matplotlib.pyplot as plt

    fig = to_matplotlib(chart, opts, style)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def render_plot(
    report: ComparisonReport,
    *,
    kind: str = "breakdown",
    path: Path | None = None,
    style: str = "paper_single_col",
    dpi: int = 300,
    **options: Any,
) -> Any:
    """Render one plot kind (see ``cta plot --list-kinds``). ``options`` are the kind's declared options."""
    chart, opts = build_chart(DataContext(report), kind, options)
    return render_chart(chart, opts, path, style=style, dpi=dpi)


def render_spec(ctx: DataContext, spec: PlotSpec, path: Path | None, *, style: str = "paper_single_col", dpi: int = 300) -> Any:
    chart, opts = build_chart(ctx, spec.kind, spec.options)
    return render_chart(chart, opts, path, style=style, dpi=dpi)
