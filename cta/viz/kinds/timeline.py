"""Timeline plot kinds: GPU Gantt, compute/comm busy fraction over time, per-stream utilisation."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import grid, ordered, single
from cta.viz.spec import Opt, register_plot


def _lane(df: pl.DataFrame) -> pl.Expr:
    """Lane name; trials of one group get a prefix so they never share a lane (their clocks are unrelated)."""
    base = pl.format("r{}/gpu{}/s{}", pl.col("rank"), pl.col("device"), pl.col("stream"))
    return (pl.format("t{}/", pl.col("trial")) + base).alias("lane") if df["trial"].n_unique() > 1 else base.alias("lane")


@register_plot(
    "timeline", group="timeline", title="GPU timeline (Gantt)", needs="kernels table (nsys, kineto or jax)",
    options={
        "by": Opt("str", "category", "What colours the bars.", ("category", "op_family", "name", "scope", "step")),
        "max_events": Opt("int", 20000, "Cap on drawn kernels; the earliest steps are kept and the chart says so."),
        "max_lanes": Opt("int", 40, "Cap on rank/device/stream rows (the busiest are kept)."),
    },
)  # fmt: skip
def timeline(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Kernels as bars on rank/device/stream lanes, times relative to each rank's first step."""
    k = ctx.frame("kernels", o)
    notes = []
    if not o.get("steps"):  # step 0 is usually a cold start that dwarfs the rest: show three steady steps instead
        n_steps = int(k["idx"].drop_nulls().max() or 0) + 1  # type: ignore[arg-type]
        if n_steps >= 5:
            k = ctx.frame("kernels", {**o, "steps": (1, 4)})
            notes.append(f"showing steps 1-3 of {n_steps} (pass steps=a:b for another window)")
    by = o["by"]
    values = sorted(str(v) for v in k[by].unique().to_list())
    ci = {v: i for i, v in enumerate(values)}
    panels = []
    for pi, label in enumerate(ordered(k, "label")):
        df = k.filter(pl.col("label") == label).with_columns(_lane(k))
        if df.height > o["max_events"]:
            df = df.sort("t_ms").head(o["max_events"])
            notes.append(f"{label}: showing the first {df.height} kernels (max_events={o['max_events']})")
        busiest = df.group_by("lane").agg(pl.col("duration_ms").sum().alias("_b")).sort("_b", descending=True).head(int(o["max_lanes"]))["lane"].to_list()
        lanes = sorted(busiest)
        df = df.filter(pl.col("lane").is_in(lanes))
        traces = []
        for v in values:
            sub = df.filter(pl.col(by).cast(pl.String) == v)
            if sub.is_empty():
                continue
            traces.append(Trace("segment", x=sub["t_ms"].to_list(), x1=sub["end_ms"].to_list(), y=sub["lane"].to_list(), name=v,
                                text=[f"{n} ({d:.3f} ms)" for n, d in zip(sub["name"].to_list(), sub["duration_ms"].to_list(), strict=True)],
                                style={"ci": ci[v], "lanes": lanes, "nolegend": pi > 0, "width": 5}, meta={"label": label}))  # fmt: skip
        panels.append(Panel(traces=traces, title=label, xlabel="time since rank's first step (ms)", ylabel="", xkind="", legend=pi == 0))
    return grid("GPU timeline", panels, max_cols=1, notes="; ".join(notes))


@register_plot(
    "overlap_timeline", group="timeline", title="Compute vs communication busy fraction over time", needs="kernels table",
    options={"bins": Opt("int", 120, "Time bins.")},
)  # fmt: skip
def overlap_timeline(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Fraction of each time bin that compute and communication/memcpy kernels keep the GPU(s) busy."""
    k = ctx.frame("kernels", o)
    panels = []
    for label in ordered(k, "label"):
        df = k.filter(pl.col("label") == label)
        t1 = float(df["end_ms"].max())  # type: ignore[arg-type]
        nb = int(o["bins"])
        edges = np.linspace(0.0, max(t1, 1e-9), nb + 1)
        width = edges[1] - edges[0]
        n_dev = max(1, df.select(pl.struct("rank", "device").n_unique()).item())
        traces = []
        for ci, (name, sel) in enumerate((("compute", df.filter(pl.col("category") == "compute")), ("comm / memcpy", df.filter(pl.col("category") != "compute")))):
            busy = _binned_busy(sel["t_ms"].to_numpy(), sel["end_ms"].to_numpy(), edges) / (width * n_dev)
            traces.append(Trace("step", x=edges[:-1].tolist(), y=np.minimum(busy, 1.0).tolist(), name=name, style={"ci": ci * 2}, meta={"label": label}))
        panels.append(Panel(traces=traces, title=label, xlabel="time since first step (ms)", ylabel="busy fraction"))
    return grid("Compute / communication busy fraction", panels, max_cols=1)


def _binned_busy(s: np.ndarray, e: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Total kernel time (ms) falling into each bin (kernels may span several bins)."""
    nb = len(edges) - 1
    out = np.zeros(nb)
    if len(s) == 0:
        return out
    b0 = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, nb - 1)
    b1 = np.clip(np.searchsorted(edges, e, side="left") - 1, 0, nb - 1)
    same = b0 == b1
    np.add.at(out, b0[same], (e - s)[same])
    for i in np.flatnonzero(~same):
        for b in range(b0[i], b1[i] + 1):
            out[b] += max(0.0, min(e[i], edges[b + 1]) - max(s[i], edges[b]))
    return out


@register_plot(
    "stream_util", group="timeline", title="Per-stream GPU busy time", needs="kernels table",
    options={"top_k": Opt("int", 16, "Busiest lanes shown.")},
)  # fmt: skip
def stream_util(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Busy time of each rank/device/stream as a share of the profiled step window."""
    k = ctx.frame("kernels", o)
    s = ctx.frame("steps", {**o, "where": None})
    win = s.group_by("label", "trial", "rank").agg((pl.col("end_ms").max() - pl.col("t_ms").min()).alias("window_ms"))
    busy = (
        k.with_columns(_lane(k)).group_by("label", "trial", "rank", "lane").agg(pl.col("duration_ms").sum().alias("busy_ms")).join(win, on=["label", "trial", "rank"])
        .with_columns((100 * pl.col("busy_ms") / pl.col("window_ms")).clip(0, 100).alias("busy_pct"))
    )  # fmt: skip
    top = busy.group_by("lane").agg(pl.col("busy_pct").max()).sort("busy_pct", descending=True).head(int(o["top_k"]))["lane"].to_list()
    traces = []
    for i, label in enumerate(ordered(k, "label")):
        sub = busy.filter((pl.col("label") == label) & pl.col("lane").is_in(top)).group_by("lane").agg(pl.col("busy_pct").mean()).sort("lane")
        traces.append(Trace("barh", x=sub["busy_pct"].to_list(), y=sub["lane"].to_list(), name=label, style={"ci": i}, meta={"label": label}))
    return single("Per-stream busy time", Panel(traces=traces, xlabel="busy time / step window (%)", invert_y=True))
