"""Draw a :class:`~cta.viz.ir.Chart` with Plotly (interactive HTML; needs ``pip install cta[viz]``)."""

from __future__ import annotations

from typing import Any

from cta.viz.ir import Chart, Panel, Trace
from cta.viz.theme import PLOTLY_SCALES, palette

_INSTALL = "interactive HTML plots need plotly: pip install 'cta[viz]'"


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc
    return go


def render(chart: Chart, *, theme: str = "light", colors: str | None = None, width: float | None = None, height: float | None = None) -> Any:
    go = _go()
    from plotly.subplots import make_subplots

    cols = palette(colors)
    n = len(chart.panels)
    titles = [p.title for p in chart.panels] if n > 1 else None
    fig = make_subplots(rows=chart.rows, cols=chart.cols, subplot_titles=titles, shared_yaxes=chart.sharey) if n > 1 else go.Figure()
    step_axes: list[str] = []
    barmode = "group"
    for i, panel in enumerate(chart.panels):
        row, col = i // chart.cols + 1, i % chart.cols + 1
        kw: dict[str, Any] = {"row": row, "col": col} if n > 1 else {}
        ci = 0
        for tr in panel.traces:
            ci += _add_trace(fig, go, tr, panel, ci, cols, kw)
        ax = "" if i == 0 else str(i + 1)
        if panel.xkind == "step":
            step_axes.append(f"xaxis{ax}")
        barmode = "stack" if panel.stacked else "overlay" if panel.overlay else barmode
        fig.update_xaxes(title_text=panel.xlabel, type="log" if panel.xscale == "log" else None, **kw)
        fig.update_yaxes(title_text=panel.ylabel, type="log" if panel.yscale == "log" else None,
                         autorange="reversed" if panel.invert_y else None, **kw)  # fmt: skip
        if panel.xrange:
            fig.update_xaxes(range=list(panel.xrange), **kw)
    w = int((width or (chart.size[0] if chart.size else 6.0 * chart.cols)) * 100) if (width or chart.size) else None
    h = int((height or (chart.size[1] if chart.size else 4.2 * chart.rows)) * 100) if (height or chart.size) else None
    fig.update_layout(
        title=chart.title if n != 1 or not chart.panels[0].title else chart.panels[0].title,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        barmode=barmode,
        width=w if width or chart.size else None,
        height=h if height or chart.size else 380 * chart.rows + 60,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        meta={**chart.meta, "step_axes": step_axes},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def _scale(name: str | None) -> str | None:
    return PLOTLY_SCALES.get(name, name) if name else None


def _add_trace(fig: Any, go: Any, tr: Trace, p: Panel, ci: int, cols: tuple[str, ...], kw: dict[str, Any]) -> int:
    used = 0 if (tr.color is not None or "ci" in tr.style) else 1
    color = tr.color if tr.color is not None else cols[int(tr.style.get("ci", ci)) % len(cols)]
    st = tr.style
    common: dict[str, Any] = {"name": tr.name, "meta": tr.meta, "showlegend": bool(tr.name) and not st.get("nolegend") and p.legend}
    if tr.name:
        common["legendgroup"] = tr.name
    m = tr.mark
    if m == "bar":
        fig.add_trace(go.Bar(x=list(tr.x or []), y=list(tr.y or []), base=list(tr.y1) if tr.y1 is not None else None, marker_color=color, **common), **kw)
    elif m == "barh":
        fig.add_trace(go.Bar(x=list(tr.x or []), y=list(tr.y or []), orientation="h", marker_color=color, **common), **kw)
    elif m in ("line", "step"):
        fig.add_trace(
            go.Scatter(x=list(tr.x or []), y=list(tr.y or []), mode="lines+markers" if st.get("marker") else "lines",
                       line={"color": color, "width": st.get("width", 1.4), "dash": _dash(st.get("dash")),
                             "shape": "hv" if m == "step" else "linear"}, opacity=st.get("alpha", 1.0), **common),
            **kw,
        )  # fmt: skip
    elif m == "scatter":
        marker: dict[str, Any] = {"size": list(tr.size) if tr.size is not None else 7, "opacity": st.get("alpha", 0.9)}
        if tr.color is not None and not isinstance(tr.color, str):
            marker.update(color=list(tr.color), colorscale=_scale(st.get("cmap")) or "Viridis", showscale=bool(p.colorbar),
                          colorbar={"title": p.colorbar})  # fmt: skip
        else:
            marker["color"] = color
        fig.add_trace(go.Scatter(x=list(tr.x or []), y=list(tr.y or []), mode="markers", marker=marker,
                                 text=list(tr.text) if tr.text else None, **common), **kw)  # fmt: skip
    elif m == "box":
        fig.add_trace(go.Box(y=list(tr.y or []), marker_color=color, boxmean=True, **common), **kw)
    elif m == "hist":
        fig.add_trace(go.Histogram(x=list(tr.y or []), nbinsx=st.get("bins", 30), opacity=st.get("alpha", 0.55),
                                   histnorm="probability density" if st.get("density") else "", marker_color=color, **common), **kw)  # fmt: skip
    elif m == "heatmap":
        fig.add_trace(go.Heatmap(z=[list(r) for r in tr.z or []], x=list(tr.x) if tr.x is not None else None,
                                 y=list(tr.y) if tr.y is not None else None, colorscale=_scale(st.get("cmap")) or "Viridis",
                                 zmin=st.get("vmin"), zmax=st.get("vmax"), colorbar={"title": p.colorbar}, meta=tr.meta), **kw)  # fmt: skip
    elif m == "segment":
        xs, xe = list(tr.x or []), list(tr.x1 or [])
        fig.add_trace(go.Bar(x=[b - a for a, b in zip(xs, xe, strict=False)], base=xs, y=list(tr.y or []), orientation="h",
                             marker_color=color, text=list(tr.text) if tr.text else None, hoverinfo="text" if tr.text else "x+y", **common), **kw)  # fmt: skip
    elif m == "band":
        x = list(tr.x or [])
        fig.add_trace(go.Scatter(x=x, y=list(tr.y1 or []), mode="lines", line={"width": 0}, showlegend=False, meta=tr.meta, hoverinfo="skip"), **kw)
        fig.add_trace(go.Scatter(x=x, y=list(tr.y or []), mode="lines", line={"width": 0}, fill="tonexty", fillcolor=_rgba(color, st.get("alpha", 0.2)),
                                 **common), **kw)  # fmt: skip
    elif m in ("hline", "vline"):
        coord = (tr.y or [0])[0] if m == "hline" else (tr.x or [0])[0]
        add = fig.add_hline if m == "hline" else fig.add_vline
        add(coord, line_dash=_dash(st.get("dash", "--")), line_color=color, line_width=st.get("width", 1.0), annotation_text=tr.name or None, **kw)
    elif m == "text":
        fig.add_trace(go.Scatter(x=list(tr.x or []), y=list(tr.y or []), mode="text", text=list(tr.text or []), textposition="top right",
                                 showlegend=False, meta=tr.meta), **kw)  # fmt: skip
    return used


def _dash(d: Any) -> str:
    return {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot", None: "solid"}.get(d, d if isinstance(d, str) else "solid")


def _rgba(hex_color: Any, alpha: float) -> str:
    h = str(hex_color).lstrip("#")
    if len(h) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
        return str(hex_color)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
