"""Per-step plot kinds: step-time trace and drift, distributions, comparisons, rank straggling."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import grid, mad_sigma, ordered, pct, rolling_median, single, trend
from cta.viz.spec import NotApplicable, Opt, register_plot
from cta.viz.theme import BAD, NEUTRAL


@register_plot(
    "step_trace", group="steps", title="Step time over steps (trend and outliers)", needs="steps table",
    options={
        "rolling": Opt("int", 5, "Rolling-median window in steps (0 = off)."),
        "outliers": Opt("float", 3.5, "Mark steps above median + k robust sigmas (0 = off)."),
        "trend": Opt("bool", True, "Fit and draw a Theil-Sen trend with its Kendall p-value."),
        "yscale": Opt("str", "auto", "linear, log, or auto (log when the slowest step is >30x the median).", ("auto", "linear", "log")),
    },
)  # fmt: skip
def step_trace(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Step duration against step index with rolling median, trend slope (drift) and outliers, one line per rank."""
    s = ctx.frame("steps", o)
    labels = ordered(s, "label")
    traces: list[Trace] = []
    notes = []
    for li, label in enumerate(labels):
        sub = s.filter(pl.col("label") == label)
        multi_trial = sub["trial"].n_unique() > 1
        for trial, rank in sorted({(t, r) for t, r in zip(sub["trial"].to_list(), sub["rank"].to_list(), strict=True)}):
            r = sub.filter((pl.col("rank") == rank) & (pl.col("trial") == trial)).sort("start_ns")
            x = r["idx"].to_numpy().astype(float)
            y = r["duration_ms"].to_numpy()
            meta = {"label": label, "rank": int(rank), "trial": int(trial)}
            multi = len(labels) > 1 or sub["rank"].n_unique() > 1 or multi_trial
            nm = label + (f" #{trial}" if multi_trial else "") + (f" r{rank}" if sub["rank"].n_unique() > 1 else "")
            traces.append(Trace("line", x=x.tolist(), y=y.tolist(), name=nm, style={"ci": li, "alpha": 0.45 if o["rolling"] else 1.0, "width": 1.0, "marker": "o"}, meta=meta))
            if o["rolling"] > 1 and len(y) > 2:
                traces.append(Trace("line", x=x.tolist(), y=rolling_median(y, int(o["rolling"])).tolist(), name=f"{nm} rolling median", style={"ci": li, "width": 2.0, "nolegend": not multi}, meta=meta))
            if o["outliers"]:
                sig = mad_sigma(y)
                hi = y > np.median(y) + o["outliers"] * sig if sig > 0 else np.zeros(len(y), bool)
                if hi.any():
                    traces.append(Trace("scatter", x=x[hi].tolist(), y=y[hi].tolist(), name=f"{nm} outliers", color=BAD, style={"s": 36, "nolegend": True}, meta=meta))
            if o["trend"]:
                tr = trend(x, y)
                if tr:
                    slope, icpt, p = tr
                    traces.append(Trace("line", x=[float(x.min()), float(x.max())], y=[icpt + slope * x.min(), icpt + slope * x.max()], color=NEUTRAL,
                                        name=f"{nm} trend {slope:+.3g} ms/step (p={p:.2g})", style={"dash": "--", "width": 1.2}, meta=meta))  # fmt: skip
                    if p < 0.05 and abs(slope) * max(1.0, x.max() - x.min()) > 0.02 * np.median(y):
                        notes.append(f"{nm}: step time drifts {slope:+.3g} ms/step (Kendall p={p:.2g})")
    ratio = float(s["duration_ms"].max()) / max(float(s["duration_ms"].median()), 1e-12)  # type: ignore[arg-type]
    log = o["yscale"] == "log" or (o["yscale"] == "auto" and ratio > 30)
    return single("Step time over steps", Panel(traces=traces, xlabel="step (after warm-up slicing)", ylabel="step time (ms)", xkind="step", yscale="log" if log else "linear"),
                  notes="; ".join(notes))


@register_plot(
    "step_dist", group="steps", title="Step-time distribution", needs="steps table",
    options={"mode": Opt("str", "ecdf", "ecdf or hist.", ("ecdf", "hist")), "bins": Opt("int", 30, "Histogram bins.")},
)  # fmt: skip
def step_dist(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Empirical CDF (or histogram) of step times with p50/p90/p99 per group."""
    s = ctx.frame("steps", o)
    traces: list[Trace] = []
    for li, label in enumerate(ordered(s, "label")):
        y = s.filter(pl.col("label") == label)["duration_ms"].to_numpy()
        q = {p: pct(y, p) for p in (50, 90, 99)}
        name = f"{label} (p50 {q[50]:.3g}, p90 {q[90]:.3g}, p99 {q[99]:.3g} ms)"
        if o["mode"] == "hist":
            traces.append(Trace("hist", y=y.tolist(), name=name, style={"ci": li, "bins": o["bins"], "density": True}, meta={"label": label}))
        else:
            ys = np.sort(y)
            traces.append(Trace("step", x=ys.tolist(), y=((np.arange(len(ys)) + 1) / len(ys)).tolist(), name=name, style={"ci": li}, meta={"label": label}))
    return single("Step-time distribution", Panel(traces=traces, xlabel="step time (ms)", ylabel="cumulative fraction" if o["mode"] == "ecdf" else "density"))


@register_plot("step_compare", group="compare", title="Paired step-time distributions", needs="steps table; best with 2+ groups")
def step_compare(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Box plot of step times per group (mean marked), with the relative change of the median vs the first group."""
    s = ctx.frame("steps", o)
    labels = ordered(s, "label")
    med0 = float(np.median(s.filter(pl.col("label") == labels[0])["duration_ms"].to_numpy()))
    traces = []
    for i, label in enumerate(labels):
        y = s.filter(pl.col("label") == label)["duration_ms"].to_numpy()
        change = f" ({100 * (np.median(y) - med0) / med0:+.1f}%)" if i and med0 else ""
        traces.append(Trace("box", y=y.tolist(), name=f"{label}{change}", style={"ci": i}, meta={"label": label}))
    return single("Step-time distributions", Panel(traces=traces, ylabel="step time (ms)", legend=False))


def _per_rank_metric(ctx: DataContext, o: dict[str, Any], metric: str) -> pl.DataFrame:
    """label, trial, rank, idx, value for a per-step metric."""
    s = ctx.frame("steps", {**o, "where": None})
    if metric == "step_ms":
        return s.select("label", "trial", "rank", "idx", pl.col("duration_ms").alias("value"))
    k = ctx.frame("kernels", {**o, "where": None})
    cat = {"compute_ms": ["compute"], "comm_ms": ["comm"], "memcpy_ms": ["memcpy"]}.get(metric)
    if cat is None:  # idle_pct
        busy = k.group_by("label", "trial", "rank", "idx").agg(pl.col("duration_ms").sum().alias("busy"))
        j = s.join(busy, on=["label", "trial", "rank", "idx"], how="left").with_columns(pl.col("busy").fill_null(0.0))
        return j.select("label", "trial", "rank", "idx", (100 * (1 - (pl.col("busy") / pl.col("duration_ms")).clip(0, 1))).alias("value"))
    agg = k.filter(pl.col("category").is_in(cat)).group_by("label", "trial", "rank", "idx").agg(pl.col("duration_ms").sum().alias("value"))
    return s.select("label", "trial", "rank", "idx").join(agg, on=["label", "trial", "rank", "idx"], how="left").with_columns(pl.col("value").fill_null(0.0))


_RANK_METRICS = ("step_ms", "compute_ms", "comm_ms", "memcpy_ms", "idle_pct")


@register_plot(
    "rank_heatmap", group="distributed", title="Per-rank, per-step heatmap", needs="steps (and kernels) with 2+ ranks",
    options={"metric": Opt("str", "step_ms", "Per-step quantity.", _RANK_METRICS)},
)  # fmt: skip
def rank_heatmap(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Heatmap of a per-step quantity over rank x step: stragglers show as hot rows, slow steps as hot columns."""
    d = _per_rank_metric(ctx, o, o["metric"])
    panels = []
    for label in ordered(d, "label"):
        sub = d.filter(pl.col("label") == label).group_by("rank", "idx").agg(pl.col("value").mean())
        if sub["rank"].n_unique() < 2:
            continue
        pv = sub.pivot(on="idx", index="rank", values="value").sort("rank")
        steps = sorted((c for c in pv.columns if c != "rank"), key=int)
        z = pv.select(steps).to_numpy()
        vmax = float(np.nanpercentile(z, 98))  # a cold first step would otherwise wash out every other cell
        clipped = vmax < float(np.nanmax(z))
        panels.append(Panel(traces=[Trace("heatmap", x=[int(c) for c in steps], y=[f"rank {r}" for r in pv["rank"].to_list()], z=z.tolist(),
                                          style={"cmap": "magma", "vmax": vmax}, meta={"label": label})], title=label, xlabel="step",
                            colorbar=o["metric"] + (" (clipped at p98)" if clipped else ""), legend=False, xkind="step"))  # fmt: skip
    if not panels:
        raise NotApplicable("needs a trace with at least 2 ranks")
    return grid("Per-rank, per-step " + o["metric"], panels, max_cols=1)


@register_plot(
    "straggler", group="distributed", title="Cross-rank spread per step", needs="steps table with 2+ ranks",
    options={"metric": Opt("str", "step_ms", "Per-step quantity.", _RANK_METRICS)},
)  # fmt: skip
def straggler(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Each rank's value per step with the min-max band; a wide band is a straggler or load imbalance."""
    d = _per_rank_metric(ctx, o, o["metric"])
    panels = []
    for label in ordered(d, "label"):
        sub = d.filter(pl.col("label") == label)
        if sub["rank"].n_unique() < 2:
            continue
        band = sub.group_by("idx").agg(pl.col("value").min().alias("lo"), pl.col("value").max().alias("hi")).sort("idx")
        traces = [Trace("band", x=band["idx"].to_list(), y=band["lo"].to_list(), y1=band["hi"].to_list(), name="min-max across ranks", color=NEUTRAL, meta={"label": label})]
        for ri, rank in enumerate(sorted(sub["rank"].unique().to_list())):
            r = sub.filter(pl.col("rank") == rank).group_by("idx").agg(pl.col("value").mean()).sort("idx")
            traces.append(Trace("line", x=r["idx"].to_list(), y=r["value"].to_list(), name=f"rank {rank}", style={"ci": ri, "width": 1.0}, meta={"label": label, "rank": int(rank)}))
        spread = float((band["hi"] - band["lo"]).mean())  # type: ignore[arg-type]
        panels.append(Panel(traces=traces, title=f"{label} (mean spread {spread:.3g})", xlabel="step", ylabel=o["metric"], xkind="step"))
    if not panels:
        raise NotApplicable("needs a trace with at least 2 ranks")
    return grid("Cross-rank spread", panels, max_cols=1)
