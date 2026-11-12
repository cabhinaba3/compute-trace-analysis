"""Plot kinds drawn from the aggregated report tables (the original six, plus waterfall, gates, scorecard, sweep, Pareto)."""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import polars as pl

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import grid, single
from cta.viz.spec import NotApplicable, Opt, register_plot
from cta.viz.theme import BAD, GOOD, NEUTRAL


@register_plot("breakdown", group="overview", title="Timeline execution breakdown", legacy=True,
               needs="aspect timeline_overlap or dag_critical_path")  # fmt: skip
def breakdown(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Stacked compute / overlapped comm / exposed comm / idle time per group."""
    labels = ctx.labels

    def series(aspect: str, items: list[tuple[str, str]]) -> list[tuple[str, list[float]]]:
        return [(nm, [ctx.metric(aspect, m).get(lb, 0.0) for lb in labels]) for nm, m in items]

    if ctx.has_aspect("timeline_overlap"):
        data = series("timeline_overlap", [("Compute only", "compute_only_ms"), ("Overlapped comm", "overlapped_comm_ms"),
                                           ("Exposed comm", "exposed_comm_ms"), ("GPU idle", "gpu_idle_ms")])  # fmt: skip
        ylabel, title = "Time (ms)", "Timeline execution breakdown"
    elif ctx.has_aspect("dag_critical_path"):
        data = series("dag_critical_path", [("CP compute", "cp_compute_us"), ("CP exposed comm", "cp_exposed_comm_us")])
        ylabel, title = "Critical-path time (us)", "Critical-path composition"
    else:
        raise NotApplicable("Breakdown plot requires 'timeline_overlap' or 'dag_critical_path' aspect")
    traces = [Trace("bar", x=labels, y=vals, name=nm, style={"ci": i}) for i, (nm, vals) in enumerate(data)]
    return single(title, Panel(traces=traces, ylabel=ylabel, stacked=True))


_TOPK_SOURCES = (
    ("kernel_diff", "by_family", "op_family", "ms"),
    ("layer_fusion", "by_layer", "name", "ms"),
    ("comm_collectives", "by_comm_type", "comm_type", "MB"),
    ("os_runtime", "by_syscall", "name", "ms"),
    ("roofline_diff", "by_kernel", "canonical_name", "us"),
)


@register_plot("topk", group="compare", title="Top contributors / hotspots", aliases=("tornado", "top_k", "hotspots"), legacy=True,
               options={"top_k": Opt("int", 10, "How many entities."), "table": Opt("str", None, "Table of kernel_diff, e.g. by_kernel or by_scope.")},
               needs="a per-entity report table (kernel_diff, layer_fusion, ...)")  # fmt: skip
def topk(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Top-K entities by time (one trace) or by delta (two traces)."""
    rep = ctx.require_report()
    if len(rep.groups) not in (1, 2):
        raise NotApplicable("Tornado / Top-K plot requires 1 trace (hotspots) or 2 traces (Baseline vs Candidate)")
    single_trace = len(rep.groups) == 1
    val_col = rep.groups[0].label if single_trace else "delta"
    cands = list(_TOPK_SOURCES)
    if o.get("table"):
        cands = [("kernel_diff", o["table"], {"by_family": "op_family", "by_scope": "scope"}.get(o["table"], "canonical_name"), "ms")] + cands
    for asp, tbl, kcol, unit in cands:  # noqa: B007
        if asp in rep.aspects and tbl in rep.aspects[asp].details:
            df = rep.aspects[asp].details[tbl]
            if df.height and val_col in df.columns:
                df = df.head(int(o["top_k"]))
                key = kcol if kcol in df.columns else df.columns[0]
                break
    else:
        raise NotApplicable(f"No entity breakdown table with '{val_col}' available for Top-K / tornado plot")
    names = [str(s)[:32] for s in df[key].to_list()]
    vals = df[val_col].to_list()
    if single_trace:
        return single(f"Top hotspots ({rep.groups[0].label})", Panel(
            traces=[Trace("barh", x=vals, y=names, name=rep.groups[0].label, style={"ci": 0, "nolegend": True})],
            xlabel=f"{rep.groups[0].label} [{unit}]", invert_y=True, legend=False))  # fmt: skip
    worse = [(n, v) for n, v in zip(names, vals, strict=True) if v > 0]
    better = [(n, v) for n, v in zip(names, vals, strict=True) if v <= 0]
    traces = [Trace("barh", x=[v for _, v in grp], y=[n for n, _ in grp], name=nm, color=col) for nm, col, grp in (("slower", BAD, worse), ("faster", GOOD, better)) if grp]
    return single("Top contributors to net change", Panel(
        traces=traces, xlabel=f"Delta ({rep.groups[1].label} - {rep.groups[0].label}) [{unit}]", invert_y=True))  # fmt: skip


def _dense(mdf: pl.DataFrame, n: int) -> np.ndarray:
    grid_ = np.zeros((n, n))
    for r in mdf.iter_rows(named=True):
        grid_[int(r["src_rank"]), int(r["dst_rank"])] = float(r["volume_mb"])
    return grid_


@register_plot("traffic_delta", group="distributed", title="Collective traffic matrix", aliases=("heatmap", "traffic"), legacy=True,
               needs="aspect traffic_matrix")  # fmt: skip
def traffic(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Rank-to-rank volume (one panel per group) or the delta for a pair."""
    labels = ctx.labels
    asp = ctx.require_report().aspects.get("traffic_matrix")
    if asp is None:
        raise NotApplicable("Traffic heatmap requires 'traffic_matrix' aspect")
    mats = [asp.details.get(f"matrix_{lb}", pl.DataFrame()) for lb in labels]
    n = max((max(int(cast(Any, m["src_rank"].max())), int(cast(Any, m["dst_rank"].max()))) + 1 for m in mats if m.height), default=2)
    ranks = list(range(n))
    if len(labels) == 2:
        mat = _dense(mats[1], n) - _dense(mats[0], n)
        vmax = max(1e-6, float(np.abs(mat).max()))
        tr = Trace("heatmap", x=ranks, y=ranks, z=mat.tolist(), style={"cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax})
        return single(f"Traffic delta ({labels[1]} - {labels[0]})", Panel(traces=[tr], xlabel="Destination rank", ylabel="Source rank", colorbar="Delta volume (MB)", legend=False))
    panels = [Panel(traces=[Trace("heatmap", x=ranks, y=ranks, z=_dense(m, n).tolist(), style={"cmap": "Blues"})], title=lb,
                    xlabel="Destination rank", ylabel="Source rank", colorbar="Volume (MB)", legend=False) for lb, m in zip(labels, mats, strict=True)]  # fmt: skip
    return grid("Traffic matrix", panels)


@register_plot("skew", group="distributed", title="Per-rank load distribution", legacy=True, needs="aspect rank_skew")
def skew(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Box plot of per-rank compute time for each group."""
    if not ctx.has_aspect("rank_skew"):
        raise NotApplicable("Skew plot requires 'rank_skew' aspect")
    asp = ctx.require_report().aspects["rank_skew"]
    traces = [Trace("box", y=asp.details[lb]["compute_us"].to_list(), name=lb, style={"ci": i}, meta={"label": lb})
              for i, lb in enumerate(ctx.labels) if lb in asp.details and asp.details[lb].height]  # fmt: skip
    if not traces:
        raise NotApplicable("No per-rank distribution data available for skew plot")
    return single("Multi-rank load distribution", Panel(traces=traces, ylabel="Per-rank compute time (us)", legend=False))


@register_plot("roofline", group="kernels", title="Kernel roofline", legacy=True,
               options={"peak_tflops": Opt("float", None, "Compute roof (TFLOP/s); draws the roofline with peak_bw_gbs."),
                        "peak_bw_gbs": Opt("float", None, "Memory roof (GB/s).")},
               needs="aspect roofline_diff (Nsight Compute data)")  # fmt: skip
def roofline(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Arithmetic intensity vs achieved TFLOP/s per kernel; arrows connect a pair's kernels."""
    if not ctx.has_aspect("roofline_diff"):
        raise NotApplicable("Roofline plot requires 'roofline_diff' aspect")
    asp = ctx.require_report().aspects["roofline_diff"]
    labels = ctx.labels
    traces: list[Trace] = []
    dfs = {lb: asp.details.get(f"roofline_{lb}", pl.DataFrame()) for lb in labels}
    for i, lb in enumerate(labels):
        df = dfs[lb]
        if df.is_empty():
            continue
        traces.append(Trace("scatter", x=np.maximum(1e-3, df["arith_intensity"].to_numpy()).tolist(), y=np.maximum(1e-3, df["achieved_tflops"].to_numpy()).tolist(),
                            name=lb, style={"ci": i}, text=[str(n) for n in df["canonical_name"].to_list()], meta={"label": lb}))  # fmt: skip
    if len(labels) == 2 and dfs[labels[0]].height and dfs[labels[1]].height:
        j = dfs[labels[0]].join(dfs[labels[1]], on="canonical_name", suffix="_c")
        for r in j.iter_rows(named=True):
            traces.append(Trace("line", x=[max(1e-3, r["arith_intensity"]), max(1e-3, r["arith_intensity_c"])],
                                y=[max(1e-3, r["achieved_tflops"]), max(1e-3, r["achieved_tflops_c"])], color=NEUTRAL, style={"width": 0.8, "nolegend": True}))  # fmt: skip
    if not traces:
        raise NotApplicable("No kernels with FLOP/byte data (needs Nsight Compute counters)")
    if o.get("peak_tflops") and o.get("peak_bw_gbs"):
        xs = np.logspace(-2, 4, 80)
        traces.append(Trace("line", x=xs.tolist(), y=np.minimum(o["peak_tflops"], o["peak_bw_gbs"] / 1e3 * xs).tolist(), name="roofline", color="#333333", style={"width": 1.6}))
    return single("Kernel roofline" + (" shift" if len(labels) == 2 else ""), Panel(
        traces=traces, xlabel="Arithmetic intensity (FLOP / byte)", ylabel="Performance (TFLOP/s)", xscale="log", yscale="log"))  # fmt: skip


_SCALING_PREF = ("mean_step_ms", "critical_path_us", "throughput_per_s", "energy_per_query_mj", "total_energy_j")


def _find_metric(ctx: DataContext, metric: str | None) -> tuple[str, str, str]:
    """(aspect, metric, unit) for the requested metric, else the first preferred one present."""
    rep = ctx.require_report()
    wanted = [metric] if metric else list(_SCALING_PREF)
    for m in wanted:
        for name, asp in rep.aspects.items():
            if not asp.summary.is_empty() and m in asp.summary["metric"].to_list():
                unit = asp.summary.filter(pl.col("metric") == m)["unit"][0]
                return name, m, str(unit)
    if metric:
        raise NotApplicable(f"metric '{metric}' not found in any aspect summary")
    asp0 = next((a for a in rep.aspects.values() if not a.summary.is_empty()), None)
    if asp0 is None:
        raise NotApplicable("No summary metrics available")
    return asp0.name, str(asp0.summary["metric"][0]), str(asp0.summary["unit"][0])


@register_plot("scaling", group="sweep", title="Metric across groups", legacy=True, options={"metric": Opt("str", None, "Summary metric name.")})
def scaling(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """One summary metric across the groups (bar for one group, line for several)."""
    aspect, metric, unit = _find_metric(ctx, o.get("metric"))
    vals = [ctx.metric(aspect, metric).get(lb, 0.0) for lb in ctx.labels]
    mark = "bar" if len(vals) == 1 else "line"
    tr = Trace(mark, x=ctx.labels, y=vals, name=metric, style={"ci": 0, "marker": "o", "nolegend": True})
    return single(f"Metric summary: {metric}" if len(vals) == 1 else f"Scaling comparison: {metric}",
                  Panel(traces=[tr], ylabel=f"{metric} ({unit})" if unit else metric, legend=False))  # fmt: skip


@register_plot("waterfall", group="compare", title="Delta waterfall", needs="2 groups; a per-entity table of kernel_diff",
               options={"top_k": Opt("int", 8, "Entities shown; the rest collapse into 'other'."),
                        "table": Opt("str", "by_family", "kernel_diff table: by_family, by_kernel or by_scope.")})  # fmt: skip
def waterfall(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """How per-entity time changes add up from the baseline total to the candidate total."""
    rep = ctx.require_report()
    if len(rep.groups) != 2:
        raise NotApplicable("Waterfall needs exactly two groups (baseline and candidate)")
    df = ctx.report_table("kernel_diff", o["table"])
    b, c = ctx.labels
    key = df.columns[0]
    if not {b, c, "delta"} <= set(df.columns):
        raise NotApplicable(f"table '{o['table']}' has no per-group columns")
    df = df.sort(pl.col("delta").abs(), descending=True)
    top, rest = df.head(int(o["top_k"])), df.tail(max(0, df.height - int(o["top_k"])))
    names = [str(n)[:28] for n in top[key].to_list()]
    deltas = top["delta"].to_list()
    if rest.height:
        names.append("other")
        deltas.append(float(rest["delta"].sum()))
    start = float(df[b].sum())
    run, base_y, cats, up, dn = start, [], [], [], []
    for n, d in zip(names, deltas, strict=True):
        cats.append(n)
        base_y.append(min(run, run + d))
        up.append(abs(d) if d > 0 else 0.0)
        dn.append(abs(d) if d <= 0 else 0.0)
        run += d
    xs = [b, *cats, c]
    traces = [
        Trace("bar", x=xs, y=[start, *[0.0] * len(cats), run], y1=[0.0] * len(xs), name="total", color=NEUTRAL),
        Trace("bar", x=cats, y=up, y1=base_y, name="slower", color=BAD),
        Trace("bar", x=cats, y=dn, y1=base_y, name="faster", color=GOOD),
    ]
    panel = Panel(traces=traces, ylabel=f"Kernel time (ms), {o['table']}", overlay=True)
    return single(f"Where the {run - start:+.3g} ms change comes from", panel)


@register_plot("gates", group="compare", title="Regression gates", needs="2+ groups and --gate thresholds")
def gates(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Observed change (%) of each gated metric against its threshold; red = violated."""
    rep = ctx.require_report()
    cfg = ctx.config
    if not cfg.gates:
        raise NotApplicable("no gates configured (use --gate 'metric:+5%' or a manifest 'gates:' block)")
    rows: list[tuple[str, float, float]] = []
    for metric, thr in cfg.gates.items():
        for asp in rep.aspects.values():
            s = asp.summary
            if "delta_pct" in s.columns and metric in s["metric"].to_list():
                rows.append((metric, float(s.filter(pl.col("metric") == metric)["delta_pct"][0]), float(thr)))
                break
    if not rows:
        raise NotApplicable("none of the gated metrics has a delta in this report")
    names = [r[0] for r in rows]
    bad = [r[1] > r[2] if r[2] >= 0 else r[1] < r[2] for r in rows]
    traces = [Trace("barh", x=[r[1] for r, b in zip(rows, bad, strict=True) if not b], y=[n for n, b in zip(names, bad, strict=True) if not b], name="within gate", color=GOOD),
              Trace("barh", x=[r[1] for r, b in zip(rows, bad, strict=True) if b], y=[n for n, b in zip(names, bad, strict=True) if b], name="violated", color=BAD)]  # fmt: skip
    traces = [t for t in traces if t.x]
    for n, _, thr in rows:
        traces.append(Trace("vline", x=[thr], color="#333333", name=f"{n} limit" if len(rows) == 1 else None, style={"nolegend": len(rows) > 1}))
    return single("Gate results", Panel(traces=traces, xlabel="Change vs baseline (%)", invert_y=True))


@register_plot("scorecard", group="compare", title="Metric scorecard (% change vs first group)", needs="2+ groups",
               options={"max_metrics": Opt("int", 40, "Cap on metrics shown."), "aspects": Opt("list", None, "Only these aspects.")})  # fmt: skip
def scorecard(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Heatmap of every summary metric's % change relative to the first group."""
    rep = ctx.require_report()
    if len(rep.groups) < 2:
        raise NotApplicable("Scorecard needs at least two groups")
    labels = ctx.labels
    rows: list[tuple[str, list[float]]] = []
    for name, asp in rep.aspects.items():
        if o.get("aspects") and name not in o["aspects"]:
            continue
        for m in asp.summary["metric"].to_list() if not asp.summary.is_empty() else []:
            vals = ctx.metric(name, m)
            if len(vals) == len(labels) and vals[labels[0]] not in (0.0,):
                base = vals[labels[0]]
                rows.append((f"{name}.{m}", [100.0 * (vals[lb] - base) / abs(base) for lb in labels[1:]]))
    if not rows:
        raise NotApplicable("no comparable metrics")
    rows = sorted(rows, key=lambda r: -max(abs(v) for v in r[1]))[: int(o["max_metrics"])]
    vmax = max(1e-6, max(abs(v) for _, r in rows for v in r))
    tr = Trace("heatmap", x=labels[1:], y=[r[0] for r in rows], z=[r[1] for r in rows], style={"cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax})
    return single(f"% change vs {labels[0]}", Panel(traces=[tr], colorbar="% change", legend=False, ylabel=""))


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


@register_plot("sweep", group="sweep", title="Metric vs configuration", needs="2+ groups",
               options={"metric": Opt("list", None, "Summary metric(s); default the first preferred one."),
                        "normalize": Opt("str", "none", "none, first (value / first group) or speedup (first / value).", ("none", "first", "speedup")),
                        "x": Opt("str", "auto", "auto (a number in the label if every label has one, else the group index), index or label.", ("auto", "index", "label")),
                        "logx": Opt("bool", False, "Log x axis."), "logy": Opt("bool", False, "Log y axis.")})  # fmt: skip
def sweep(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """One or more metrics across ordered groups (a sweep over batch size, GPUs, precision...), optionally normalised."""
    labels = ctx.labels
    metrics = o.get("metric") or [None]
    nums = [_NUM.search(lb) for lb in labels]
    use_num = o["x"] == "auto" and all(nums) and len({float(m.group()) for m in nums if m}) == len(labels)  # type: ignore[union-attr]
    xs: list[Any] = [float(m.group()) for m in nums if m] if use_num else (labels if o["x"] == "label" else list(range(len(labels))))
    traces = []
    for i, m in enumerate(metrics):
        aspect, name, unit = _find_metric(ctx, m)
        vals = ctx.metric(aspect, name)
        y = [vals.get(lb, float("nan")) for lb in labels]
        if o["normalize"] == "first" and y[0]:
            y = [v / y[0] for v in y]
        elif o["normalize"] == "speedup" and y[0]:
            y = [y[0] / v if v else float("nan") for v in y]
        traces.append(Trace("line", x=xs, y=y, name=name + (f" ({unit})" if unit and o["normalize"] == "none" else ""), style={"ci": i, "marker": "o"}))
    ylabel = {"none": "value", "first": "relative to first group", "speedup": "speedup vs first group"}[o["normalize"]]
    panel = Panel(traces=traces, xlabel="configuration" if not use_num else "value from group label", ylabel=ylabel,
                  xscale="log" if o["logx"] and use_num else "linear", yscale="log" if o["logy"] else "linear")  # fmt: skip
    if not use_num and o["x"] != "label":
        panel.xlabel = "group index (" + ", ".join(f"{i}={lb}" for i, lb in enumerate(labels)) + ")"
    if o["normalize"] != "none":
        traces.append(Trace("hline", y=[1.0], color=NEUTRAL, style={"nolegend": True}))
    return single("Metric vs configuration", panel)


@register_plot("energy_pareto", group="edge", title="Latency vs energy", needs="a latency and an energy metric (edge/jtop bundles)",
               options={"x": Opt("str", "mean_step_ms", "Latency-like metric."), "y": Opt("str", "energy_per_query_mj", "Energy-like metric.")})  # fmt: skip
def energy_pareto(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """Groups in the latency/energy plane with the Pareto frontier (lower is better on both axes)."""
    ax_, xm, xu = _find_metric(ctx, o["x"])
    ay_, ym, yu = _find_metric(ctx, o["y"])
    xv, yv = ctx.metric(ax_, xm), ctx.metric(ay_, ym)
    labels = [lb for lb in ctx.labels if lb in xv and lb in yv]
    if not labels:
        raise NotApplicable("no group has both metrics")
    pts = [(xv[lb], yv[lb], lb) for lb in labels]
    front = sorted(p for p in pts if not any(q[0] <= p[0] and q[1] <= p[1] and q[:2] != p[:2] for q in pts))
    traces = [Trace("scatter", x=[p[0]], y=[p[1]], name=p[2], style={"ci": i, "s": 60}, text=[p[2]], meta={"label": p[2]}) for i, p in enumerate(pts)]
    if len(front) > 1:
        traces.append(Trace("step", x=[p[0] for p in front], y=[p[1] for p in front], name="Pareto frontier", color=NEUTRAL, style={"dash": "--"}))
    return single("Latency vs energy", Panel(traces=traces, xlabel=f"{xm} ({xu})", ylabel=f"{ym} ({yu})"))
