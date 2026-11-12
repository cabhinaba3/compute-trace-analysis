"""Kernel-level plot kinds: Pareto, family share, A/B scatter, per-kernel drift, Nsight Compute views."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import grid, ordered, single, trend
from cta.viz.spec import NotApplicable, Opt, register_plot
from cta.viz.theme import NEUTRAL

_BY = ("canonical_name", "name", "op_family", "scope", "category")


def _by_opt(default: str) -> Opt:
    return Opt("str", default, "Column to group kernels by.", _BY)


@register_plot("pareto", group="timeline", title="Kernel Pareto (share and cumulative share)", needs="kernels table",
               options={"by": _by_opt("canonical_name"), "top_k": Opt("int", 15, "Entities shown.")})  # fmt: skip
def pareto(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Share of GPU time by kernel/family/scope, sorted, with the cumulative share line; one panel per group."""
    k = ctx.frame("kernels", o)
    by = o["by"]
    panels = []
    for label in ordered(k, "label"):
        g = k.filter(pl.col("label") == label).group_by(by).agg(pl.col("duration_ms").sum()).sort("duration_ms", descending=True)
        total = float(g["duration_ms"].sum())
        top = g.head(int(o["top_k"]))
        names = [str(n)[:36] for n in top[by].to_list()]
        share = (100 * top["duration_ms"] / total).to_list()
        cum = np.cumsum(share).tolist()
        panels.append(Panel(traces=[Trace("bar", x=names, y=share, name="share of GPU time", style={"ci": 0}, meta={"label": label}),
                                    Trace("line", x=names, y=cum, name="cumulative", color=NEUTRAL, style={"marker": "o"}, meta={"label": label})],
                            title=f"{label} ({total:.3g} ms total)", ylabel="% of GPU time", xlabel=by))  # fmt: skip
    return grid("Kernel Pareto", panels, max_cols=1)


@register_plot("family_share", group="overview", title="Time by kernel family / category / scope", needs="kernels table",
               options={"by": _by_opt("op_family"), "normalize": Opt("bool", False, "Show 100% stacked shares instead of ms."), "top_k": Opt("int", 8, "Entities kept; the rest become 'other'.")})  # fmt: skip
def family_share(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Stacked bars of GPU time per group, split by the chosen column."""
    k = ctx.frame("kernels", o)
    by = o["by"]
    steps = ctx.frame("steps", {**o, "where": None}).group_by("label", "trial", "rank").len().group_by("label").agg(pl.col("len").sum().alias("n_steps"))
    agg = k.group_by("label", by).agg(pl.col("duration_ms").sum()).join(steps, on="label").with_columns((pl.col("duration_ms") / pl.col("n_steps")).alias("ms_per_step"))
    top = agg.group_by(by).agg(pl.col("ms_per_step").sum()).sort("ms_per_step", descending=True).head(int(o["top_k"]))[by].to_list()
    agg = agg.with_columns(pl.when(pl.col(by).is_in(top)).then(pl.col(by)).otherwise(pl.lit("other")).alias(by)).group_by("label", by).agg(pl.col("ms_per_step").sum())
    labels = ordered(k, "label")
    traces = []
    for i, v in enumerate([*top, "other"]):
        sub = agg.filter(pl.col(by) == v)
        if sub.is_empty():
            continue
        vals = {r["label"]: r["ms_per_step"] for r in sub.iter_rows(named=True)}
        ys = [vals.get(lb, 0.0) for lb in labels]
        if o["normalize"]:
            tot = [agg.filter(pl.col("label") == lb)["ms_per_step"].sum() or 1.0 for lb in labels]
            ys = [100 * y / t for y, t in zip(ys, tot, strict=True)]
        traces.append(Trace("bar", x=labels, y=ys, name=str(v), style={"ci": i}))
    return single("GPU time by " + by, Panel(traces=traces, ylabel="% of GPU time" if o["normalize"] else "GPU time per step (ms)", stacked=True))


@register_plot("kernel_scatter", group="compare", title="Kernel time: baseline vs candidate", needs="kernels table, exactly 2 groups",
               options={"by": _by_opt("canonical_name"), "label_top": Opt("int", 6, "Annotate this many biggest movers.")})  # fmt: skip
def kernel_scatter(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """One point per kernel: total time in the baseline (x) against the candidate (y); the diagonal means unchanged."""
    k = ctx.frame("kernels", o)
    labels = ordered(k, "label")
    if len(labels) != 2:
        raise NotApplicable("needs exactly two groups")
    by = o["by"]
    a = k.filter(pl.col("label") == labels[0]).group_by(by).agg(pl.col("duration_ms").sum().alias("a"))
    b = k.filter(pl.col("label") == labels[1]).group_by(by).agg(pl.col("duration_ms").sum().alias("b"))
    j = a.join(b, on=by, how="full", coalesce=True).fill_null(0.0).with_columns((pl.col("b") - pl.col("a")).abs().alias("_mv"))
    floor = max(1e-6, float(min(cast(Any, j.filter(pl.col("a") > 0)["a"].min()) or 1e-3, cast(Any, j.filter(pl.col("b") > 0)["b"].min()) or 1e-3)) / 2)
    x, y = np.maximum(j["a"].to_numpy(), floor), np.maximum(j["b"].to_numpy(), floor)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    top = j.sort("_mv", descending=True).head(int(o["label_top"]))
    traces = [Trace("line", x=[lo, hi], y=[lo, hi], color=NEUTRAL, name="no change", style={"dash": "--"}),
              Trace("scatter", x=x.tolist(), y=y.tolist(), name="kernels", style={"ci": 0}, text=[str(n) for n in j[by].to_list()]),
              Trace("text", x=np.maximum(top["a"].to_numpy(), floor).tolist(), y=np.maximum(top["b"].to_numpy(), floor).tolist(), text=[str(n)[:24] for n in top[by].to_list()])]  # fmt: skip
    return single("Kernel time (ms): " + f"{labels[0]} vs {labels[1]}", Panel(traces=traces, xlabel=f"{labels[0]} (ms, floor {floor:.2g})", ylabel=f"{labels[1]} (ms)", xscale="log", yscale="log"))


@register_plot("kernel_drift", group="drift", title="Per-kernel duration drift across steps", needs="kernels table with >= 4 steps",
               options={"top_k": Opt("int", 20, "Kernels shown (by total time)."), "by": _by_opt("canonical_name")})  # fmt: skip
def kernel_drift(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Mean duration of each kernel per step relative to its own median; a kernel drifting up or down shows as a gradient."""
    k = ctx.frame("kernels", o).filter(pl.col("idx").is_not_null())
    by = o["by"]
    panels, notes = [], []
    for label in ordered(k, "label"):
        sub = k.filter(pl.col("label") == label)
        top = sub.group_by(by).agg(pl.col("duration_ms").sum()).sort("duration_ms", descending=True).head(int(o["top_k"]))[by].to_list()
        per = sub.filter(pl.col(by).is_in(top)).group_by(by, "idx").agg(pl.col("duration_ms").mean().alias("v"))
        if per["idx"].n_unique() < 4:
            continue
        pv = per.pivot(on="idx", index=by, values="v")
        steps = sorted((c for c in pv.columns if c != by), key=int)
        rows, ylabels = [], []
        for r in pv.sort(by).iter_rows(named=True):
            y = np.array([r[c] if r[c] is not None else np.nan for c in steps], dtype=float)
            ok = ~np.isnan(y)
            med = float(np.nanmedian(y))
            rows.append((y / med).tolist() if med > 0 else [1.0] * len(y))
            tr = trend(np.array([int(c) for c in steps], float)[ok], y[ok]) if ok.sum() >= 4 else None
            flag = ""
            if tr and tr[2] < 0.05 and med > 0 and abs(tr[0]) * len(steps) / med > 0.05:
                flag = f" {100 * tr[0] / med:+.1f}%/step"
                notes.append(f"{label}: {str(r[by])[:30]}{flag}")
            ylabels.append(str(r[by])[:30] + flag)
        panels.append(Panel(traces=[Trace("heatmap", x=[int(c) for c in steps], y=ylabels, z=rows, style={"cmap": "RdBu_r", "vmin": 0.5, "vmax": 1.5}, meta={"label": label})],
                            title=label, xlabel="step", colorbar="duration / kernel median", legend=False, xkind="step"))  # fmt: skip
    if not panels:
        raise NotApplicable("needs at least 4 steps")
    return grid("Per-kernel duration drift", panels, max_cols=1, notes="; ".join(notes[:6]))


def _ncu(ctx: DataContext, o: dict[str, Any]) -> pl.DataFrame:
    return ctx.frame("ncu_kernels", o)


def _ncu_top(df: pl.DataFrame, label: str, k: int) -> pl.DataFrame:
    """Longest k kernels of one group (kernel ids restart at 0 in every report, so never pool groups)."""
    return df.filter(pl.col("label") == label).sort("duration_ns", descending=True).unique(["trial", "kernel_id"], keep="first", maintain_order=True).head(k)


@register_plot("occupancy", group="kernels", title="Occupancy vs duration (Nsight Compute)", needs="Nsight Compute data",
               options={"x": Opt("str", "occupancy_pct", "ncu column on x."), "y": Opt("str", "duration_ns", "ncu column on y."),
                        "size": Opt("str", "dram_bytes", "Column that scales the marker (none to disable)."), "logy": Opt("bool", True, "Log y axis.")})  # fmt: skip
def occupancy(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Achieved occupancy against duration per profiled kernel, coloured by family, sized by DRAM traffic; one panel per group."""
    df = _ncu(ctx, o)
    for col in (o["x"], o["y"]):
        if col not in df.columns:
            raise NotApplicable(f"ncu table has no column '{col}': {', '.join(df.columns)}")
    fams = sorted(df["op_family"].unique().to_list())
    panels = []
    for pi, label in enumerate(ordered(df, "label")):
        traces = []
        for i, fam in enumerate(fams):
            sub = df.filter((pl.col("label") == label) & (pl.col("op_family") == fam)).drop_nulls([o["x"], o["y"]])
            if sub.is_empty():
                continue
            size = None
            if o["size"] != "none" and o["size"] in sub.columns and sub[o["size"]].drop_nulls().len():
                v = sub[o["size"]].fill_null(0).to_numpy().astype(float)
                size = (6 + 30 * v / max(v.max(), 1e-9)).tolist()
            traces.append(Trace("scatter", x=sub[o["x"]].to_list(), y=sub[o["y"]].to_list(), name=fam, size=size,
                                style={"ci": i, "nolegend": pi > 0}, text=[str(n)[:40] for n in sub["canonical_name"].to_list()], meta={"label": label}))  # fmt: skip
        panels.append(Panel(traces=traces, title=label, xlabel=o["x"], ylabel=o["y"], yscale="log" if o["logy"] else "linear"))
    if not any(p.traces for p in panels):
        raise NotApplicable("no kernels with those columns")
    return grid("Kernel occupancy", panels, max_cols=2)


@register_plot("ncu_util", group="kernels", title="Throughput utilisation per kernel (Nsight Compute)", needs="Nsight Compute data",
               options={"top_k": Opt("int", 12, "Kernels shown per group (by duration).")})  # fmt: skip
def ncu_util(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """SM, DRAM, memory and tensor-pipe throughput (% of peak) for each group's longest kernels: shows what limits each."""
    df = _ncu(ctx, o)
    panels = []
    for pi, label in enumerate(ordered(df, "label")):
        top = _ncu_top(df, label, int(o["top_k"]))
        ys = [f"{str(n)[:26]} #{j}" for j, n in enumerate(top["canonical_name"].to_list())]
        traces = [Trace("barh", x=top[col].fill_null(0).to_list(), y=ys, name=nm, style={"ci": i, "nolegend": pi > 0}, meta={"label": label})
                  for i, (col, nm) in enumerate((("sm_throughput_pct", "SM"), ("dram_throughput_pct", "DRAM"), ("mem_throughput_pct", "memory (max unit)"), ("tensor_pipe_pct", "tensor pipe")))
                  if col in top.columns and top[col].drop_nulls().len()]  # fmt: skip
        panels.append(Panel(traces=traces, title=label, xlabel="% of peak", invert_y=True))
    if not any(p.traces for p in panels):
        raise NotApplicable("ncu table has no throughput columns")
    return grid("Throughput utilisation (% of peak)", panels, max_cols=2)


@register_plot("stalls", group="kernels", title="Warp stall split (Nsight Compute)", needs="Nsight Compute data",
               options={"top_k": Opt("int", 12, "Kernels shown per group (by duration).")})  # fmt: skip
def stalls(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Memory vs compute share of warp stalls for each group's longest kernels."""
    df = _ncu(ctx, o)
    for col in ("stall_mem_pct", "stall_compute_pct"):
        if col not in df.columns or not df[col].drop_nulls().len():
            raise NotApplicable("ncu table has no stall data")
    panels = []
    for label in ordered(df, "label"):
        top = _ncu_top(df, label, int(o["top_k"]))
        ys = [f"{str(n)[:26]} #{j}" for j, n in enumerate(top["canonical_name"].to_list())]
        panels.append(Panel(traces=[Trace("barh", x=top["stall_mem_pct"].fill_null(0).to_list(), y=ys, name="memory stalls", style={"ci": 1}, meta={"label": label}),
                                    Trace("barh", x=top["stall_compute_pct"].fill_null(0).to_list(), y=ys, name="compute stalls", style={"ci": 0}, meta={"label": label})],
                            title=label, xlabel="% of stall cycles", invert_y=True, stacked=True))  # fmt: skip
    return grid("Warp stalls", panels, max_cols=2)
