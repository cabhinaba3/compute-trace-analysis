"""Draw a :class:`~cta.viz.ir.Chart` with Matplotlib (static PDF/SVG/PNG, Type-42 fonts)."""

from __future__ import annotations

from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from cta.viz.ir import Chart, Panel, Trace  # noqa: E402
from cta.viz.theme import STYLE_SIZES, palette, rc_for  # noqa: E402


def render(chart: Chart, *, style: str = "paper_single_col", theme: str = "light", colors: str | None = None,
           width: float | None = None, height: float | None = None) -> Figure:  # fmt: skip
    cols = palette(colors)
    base_w, base_h = STYLE_SIZES.get(style, STYLE_SIZES["paper_single_col"])
    w, h = chart.size or (base_w * chart.cols, base_h * chart.rows)
    w, h = width or w, height or h
    with plt.rc_context(cast(Any, rc_for(theme))):
        fig = plt.figure(figsize=(w, h), constrained_layout=True)
        axes = fig.subplots(chart.rows, chart.cols, squeeze=False, sharey=chart.sharey) if len(chart.panels) > 1 else [[fig.add_subplot(111)]]
        flat = [ax for row in axes for ax in row]
        for ax, panel in zip(flat, chart.panels, strict=False):
            _draw_panel(ax, panel, cols, fig)
        for ax in flat[len(chart.panels):]:
            ax.set_visible(False)
        if len(chart.panels) > 1 or not chart.panels[0].title:
            fig.suptitle(chart.title, fontsize=9)
        if chart.notes:
            fig.text(0.01, -0.02, chart.notes, fontsize=5.5, va="top", wrap=True)
        return fig


def _color(tr: Trace, i: int, cols: tuple[str, ...]) -> Any:
    if tr.color is not None:
        return tr.color
    return cols[int(tr.style.get("ci", i)) % len(cols)]


def _draw_panel(ax: Axes, p: Panel, cols: tuple[str, ...], fig: Figure) -> None:
    labelled = False
    boxes = [t for t in p.traces if t.mark == "box"]
    bars = [t for t in p.traces if t.mark == "bar"]
    barhs = [t for t in p.traces if t.mark == "barh"]
    ci = 0
    if boxes:
        data = [np.asarray(t.y, dtype=float) for t in boxes]
        bp = ax.boxplot(data, tick_labels=[t.name or "" for t in boxes], patch_artist=True)
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_color(boxes[k], k, cols))
            patch.set_alpha(0.75)
    if bars:
        _draw_bars(ax, bars, cols, p.stacked, vertical=True, overlay=p.overlay)
        labelled = True
    if barhs:
        _draw_bars(ax, barhs, cols, p.stacked, vertical=False, overlay=p.overlay)
        labelled = True
    for tr in p.traces:
        if tr.mark in ("box", "bar", "barh"):
            continue
        c = _color(tr, ci, cols)
        ci += 1 if (tr.color is None and "ci" not in tr.style) else 0
        st = tr.style
        label = tr.name if tr.name and not st.get("nolegend") else None
        x = np.asarray(tr.x) if tr.x is not None and tr.mark != "heatmap" else None
        y = np.asarray(tr.y) if tr.y is not None and tr.mark != "heatmap" else None
        if tr.mark == "line":
            ax.plot(x, y, color=c, lw=st.get("width", 1.2), ls=st.get("dash", "-"), marker=st.get("marker"), ms=3, label=label, alpha=st.get("alpha", 1.0))
        elif tr.mark == "step":
            ax.step(x, y, where="post", color=c, lw=st.get("width", 1.2), label=label, alpha=st.get("alpha", 1.0))
        elif tr.mark == "scatter":
            sizes = np.asarray(tr.size) if tr.size is not None else st.get("s", 28)
            sc = ax.scatter(x, y, s=sizes, c=tr.color if (tr.color is not None and not isinstance(tr.color, str)) else c,
                            cmap=st.get("cmap"), alpha=st.get("alpha", 0.9), label=label, zorder=3, marker=st.get("marker", "o"))  # fmt: skip
            if st.get("cmap") and p.colorbar:
                fig.colorbar(sc, ax=ax, label=p.colorbar)
        elif tr.mark == "hist":
            vals = np.asarray(tr.y, dtype=float)
            ax.hist(vals, bins=st.get("bins", 30), density=st.get("density", False), color=c, alpha=st.get("alpha", 0.55), label=label)
        elif tr.mark == "band":
            ax.fill_between(x, np.asarray(tr.y, dtype=float), np.asarray(tr.y1, dtype=float), color=c, alpha=st.get("alpha", 0.2), lw=0, label=label)
        elif tr.mark == "segment":
            _draw_segments(ax, tr, c)
        elif tr.mark == "heatmap":
            z = np.asarray(tr.z, dtype=float)
            im = ax.imshow(z, aspect="auto", cmap=st.get("cmap", "viridis"), origin="upper", vmin=st.get("vmin"), vmax=st.get("vmax"))
            fig.colorbar(im, ax=ax, label=p.colorbar)
            if tr.x is not None:
                step = max(1, len(tr.x) // 12)
                ax.set_xticks(range(0, len(tr.x), step), [str(v) for v in list(tr.x)[::step]])
            if tr.y is not None:
                step = max(1, len(tr.y) // 25)
                ax.set_yticks(range(0, len(tr.y), step), [str(v) for v in list(tr.y)[::step]])
            ax.grid(False)
        elif tr.mark == "hline":
            ax.axhline(float(tr.y[0]), color=c, lw=st.get("width", 0.9), ls=st.get("dash", "--"), label=label)  # type: ignore[index]
        elif tr.mark == "vline":
            ax.axvline(float(tr.x[0]), color=c, lw=st.get("width", 0.9), ls=st.get("dash", "--"), label=label)  # type: ignore[index]
        elif tr.mark == "text":
            for xi, yi, s in zip(tr.x or [], tr.y or [], tr.text or [], strict=False):
                ax.annotate(s, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points", color=c)
        labelled = labelled or label is not None
    if p.xscale != "linear":  # set_xscale resets tick labels, so only touch it when needed
        ax.set_xscale(p.xscale)
    if p.yscale != "linear":
        ax.set_yscale(p.yscale)
    ax.set_title(p.title)
    ax.set_xlabel(p.xlabel)
    ax.set_ylabel(p.ylabel)
    if p.xrange:
        ax.set_xlim(*p.xrange)
    if p.invert_y:
        ax.invert_yaxis()
    handles, names = ax.get_legend_handles_labels()
    if p.legend and handles and (labelled or names):
        ax.legend(frameon=False, loc="best")


def _draw_bars(ax: Axes, bars: list[Trace], cols: tuple[str, ...], stacked: bool, *, vertical: bool, overlay: bool = False) -> None:
    cats = list(dict.fromkeys(str(v) for t in bars for v in (t.x if vertical else t.y) or []))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(bars)
    width = 0.6 if (stacked or overlay) else 0.8 / max(1, n)
    base = np.zeros(len(cats))
    for k, t in enumerate(bars):
        pos = np.array([idx[str(v)] for v in (t.x if vertical else t.y) or []], dtype=float)
        vals = np.asarray(t.y if vertical else t.x, dtype=float)
        c = _color(t, k, cols)
        off = 0.0 if (stacked or overlay) else (k - (n - 1) / 2) * width
        label = t.name
        under = np.asarray(t.y1, dtype=float) if t.y1 is not None else (base[pos.astype(int)] if stacked else None)
        if vertical:
            ax.bar(pos + off, vals, width=width, bottom=under, color=c, label=label)
        else:
            ax.barh(pos + off, vals, height=width * (1 if stacked else 1), left=under, color=c, label=label)
        if stacked:
            np.add.at(base, pos.astype(int), vals)
    if vertical and (len(cats) > 4 or max((len(c) for c in cats), default=0) > 9):
        ax.set_xticks(range(len(cats)), cats, rotation=35, ha="right")
    else:
        (ax.set_xticks if vertical else ax.set_yticks)(range(len(cats)), cats)


def _draw_segments(ax: Axes, tr: Trace, c: Any) -> None:
    lanes = tr.style.get("lanes") or []
    ypos = {ln: i for i, ln in enumerate(lanes)}
    xs = np.asarray(tr.x, dtype=float)
    xe = np.asarray(tr.x1, dtype=float)
    ys = np.array([ypos.get(v, v) for v in tr.y or []], dtype=float)
    ax.hlines(ys, xs, xe, colors=c, lw=tr.style.get("width", 4), label=tr.name or "_nolegend_")
    if lanes:
        ax.set_yticks(range(len(lanes)), lanes)
