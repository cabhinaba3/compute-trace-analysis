"""``custom``: draw any raw table, report table or DuckDB SQL result with a small grammar of graphics."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import grid
from cta.viz.spec import NotApplicable, Opt, register_plot

_AGGS = ("sum", "mean", "median", "min", "max", "count", "p50", "p90", "p99")
_MARKS = ("line", "step", "bar", "barh", "scatter", "box", "hist", "heatmap", "area")


def _agg_expr(col: str, how: str) -> pl.Expr:
    c = pl.col(col)
    return {"sum": c.sum(), "mean": c.mean(), "median": c.median(), "min": c.min(), "max": c.max(), "count": c.count(),
            "p50": c.quantile(0.5), "p90": c.quantile(0.9), "p99": c.quantile(0.99)}[how].alias(col)  # fmt: skip


@register_plot(
    "custom", group="custom", title="Any table or SQL query", needs="a source (raw table, report:<aspect>[.<table>]) or sql",
    options={
        "source": Opt("str", None, "kernels, steps, gpu_metrics, runtime, ncu_kernels, ncu_findings, jtop, osrt, or report:<aspect>[.<table>]."),
        "sql": Opt("str", None, "DuckDB SQL over the named tables (cta plot --list-sources); overrides source."),
        "mark": Opt("str", "line", "How to draw it.", _MARKS),
        "x": Opt("str", None, "Column on x."), "y": Opt("str", None, "Column on y (values)."),
        "color": Opt("str", None, "Column whose values become separate coloured series."),
        "facet": Opt("str", None, "Column whose values become separate panels."),
        "size": Opt("str", None, "Column that scales scatter markers."),
        "agg": Opt("str", None, "Aggregate y per (x, color, facet).", _AGGS),
        "sort": Opt("str", None, "Sort bars by y: asc or desc.", ("asc", "desc")),
        "top_k": Opt("int", None, "Keep the first K x values after sorting."),
        "xscale": Opt("str", "linear", "linear or log.", ("linear", "log")), "yscale": Opt("str", "linear", "linear or log.", ("linear", "log")),
        "stack": Opt("bool", False, "Stack bars."), "bins": Opt("int", 30, "Histogram bins."),
        "xlabel": Opt("str", None, "x axis label."), "ylabel": Opt("str", None, "y axis label."),
    },
)  # fmt: skip
def custom(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """A user-defined plot: pick a source or write SQL, then map columns to x, y, colour, facet and size."""
    if o.get("sql"):
        df = ctx.sql(str(o["sql"]), o)
    elif o.get("source"):
        src = str(o["source"])
        if src.startswith("report:"):
            asp, _, tbl = src[len("report:"):].partition(".")
            df = ctx.report_table(asp, tbl or "summary")
        else:
            df = ctx.frame(src, o)
    else:
        raise NotApplicable("custom needs `source` or `sql`")
    mark, x, y = o["mark"], o.get("x"), o.get("y")
    for name, col in (("x", x), ("y", y), ("color", o.get("color")), ("facet", o.get("facet")), ("size", o.get("size"))):
        if col and col not in df.columns:
            raise ValueError(f"{name}='{col}' is not a column. Columns: {', '.join(df.columns)}")
    if mark != "hist" and mark != "box" and not (x and y) and mark != "heatmap":
        raise ValueError(f"mark '{mark}' needs both x and y")
    if mark == "hist" and not (y or x):
        raise ValueError("hist needs y (or x) for the values")
    keys = [c for c in (x, o.get("color"), o.get("facet")) if c]
    if o.get("agg") and y:
        df = df.group_by(list(dict.fromkeys(keys))).agg(_agg_expr(y, o["agg"])) if keys else df.select(_agg_expr(y, o["agg"]))
    facets = df[o["facet"]].unique(maintain_order=True).to_list() if o.get("facet") else [None]
    panels = []
    for fv in facets:
        sub = df if fv is None else df.filter(pl.col(o["facet"]) == fv)
        if o.get("sort") and y and mark in ("bar", "barh"):
            sub = sub.sort(y, descending=o["sort"] == "desc")
        if o.get("top_k"):
            sub = sub.head(int(o["top_k"]))
        panels.append(Panel(traces=_traces(sub, o, mark, x, y), title="" if fv is None else str(fv), xlabel=o.get("xlabel") or (x or ""), ylabel=o.get("ylabel") or (y or ""),
                            xscale=o["xscale"], yscale=o["yscale"], stacked=bool(o["stack"]), colorbar=o.get("size") or "" if mark == "heatmap" else ""))  # fmt: skip
    title = o.get("title") or (f"{y} by {x}" if x and y else "custom plot")
    return grid(title, panels, max_cols=3)


def _traces(df: pl.DataFrame, o: dict[str, Any], mark: str, x: str | None, y: str | None) -> list[Trace]:
    color = o.get("color")
    if mark == "heatmap":
        if not (x and y and o.get("color")):
            raise ValueError("heatmap needs x, y and color (the cell value)")
        pv = df.pivot(on=x, index=y, values=color, aggregate_function="mean").sort(y)
        cols = [c for c in pv.columns if c != y]
        return [Trace("heatmap", x=cols, y=pv[y].to_list(), z=pv.select(cols).to_numpy().tolist(), style={"cmap": "viridis"})]
    groups = df.partition_by(color, maintain_order=True, as_dict=True) if color else {(None,): df}
    out = []
    for i, (key, g) in enumerate(groups.items()):
        name = None if key[0] is None else str(key[0])
        style = {"ci": i, "marker": "o" if mark == "line" and g.height < 60 else None}
        xs = g[x].to_list() if x else None
        if mark in ("line", "step", "area"):
            gs = g.sort(x) if x else g
            tr = Trace("step" if mark == "step" else "line", x=gs[x].to_list(), y=gs[y].to_list(), name=name, style=style)  # type: ignore[index]
            out.append(tr)
        elif mark == "scatter":
            size = None
            if o.get("size"):
                v = g[o["size"]].fill_null(0).to_numpy().astype(float)
                size = (6 + 30 * v / max(float(np.max(v)), 1e-9)).tolist()
            out.append(Trace("scatter", x=xs, y=g[y].to_list(), name=name, size=size, style=style))  # type: ignore[index]
        elif mark in ("bar", "barh"):
            cats = [str(v) for v in xs or []]
            out.append(Trace(mark, x=cats if mark == "bar" else g[y].to_list(), y=g[y].to_list() if mark == "bar" else cats, name=name, style=style))  # type: ignore[index]
        elif mark == "box":
            if x:  # one box per x category (per colour)
                for cat in g[x].unique(maintain_order=True).to_list():
                    out.append(Trace("box", y=g.filter(pl.col(x) == cat)[y or x].to_list(), name=f"{cat}" if name is None else f"{name}/{cat}", style={"ci": len(out)}))  # type: ignore[index]
            else:
                out.append(Trace("box", y=g[y].to_list(), name=name or y, style=style))  # type: ignore[index]
        elif mark == "hist":
            out.append(Trace("hist", y=g[y or x].to_list(), name=name, style={**style, "bins": o["bins"]}))  # type: ignore[index]
    return out
