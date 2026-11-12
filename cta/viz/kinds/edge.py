"""Time-series plot kinds from sampled signals: Jetson power/thermals and Nsight Systems GPU metrics."""

from __future__ import annotations

from typing import Any

import polars as pl

from cta.viz.data import DataContext
from cta.viz.ir import Chart, Panel, Trace
from cta.viz.kinds._util import grid, ordered
from cta.viz.spec import NotApplicable, Opt, register_plot

_JTOP = {
    "power_tot_w": "total power (W)", "power_gpu_w": "GPU power (W)", "gpu_util_pct": "GPU utilisation (%)", "cpu_util_pct": "CPU utilisation (%)",
    "gpu_temp_c": "GPU temperature (C)", "gpu_mem_mb": "GPU memory (MB)", "ram_mb": "RAM (MB)", "cum_energy_j": "cumulative energy (J)",
}  # fmt: skip


@register_plot("power_trace", group="edge", title="Power, utilisation and thermals over time", needs="jtop samples (edge bundle)",
               options={"metrics": Opt("list", ["power_tot_w", "gpu_util_pct", "gpu_temp_c", "gpu_mem_mb"], f"jtop columns: {', '.join(_JTOP)}.")})  # fmt: skip
def power_trace(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """One panel per sampled signal against time, one line per group (shows throttling, power caps, memory growth)."""
    j = ctx.frame("jtop", o)
    bad = [m for m in o["metrics"] if m not in _JTOP]
    if bad:
        raise ValueError(f"unknown jtop metric(s) {bad}; choose from {', '.join(_JTOP)}")
    panels = []
    for m in o["metrics"]:
        traces = []
        for li, label in enumerate(ordered(j, "label")):
            for t in sorted(j.filter(pl.col("label") == label)["trial"].unique().to_list()):
                s = j.filter((pl.col("label") == label) & (pl.col("trial") == t)).sort("elapsed_s")
                traces.append(Trace("line", x=s["elapsed_s"].to_list(), y=s[m].to_list(), name=label, style={"ci": li, "width": 1.1, "nolegend": t > 0}, meta={"label": label}))
        panels.append(Panel(traces=traces, title=_JTOP[m], xlabel="time (s)", ylabel=""))
    return grid("Sampled signals", panels, max_cols=2)


_DEFAULT_GPU_METRICS = ["SMs Active", "SM Issue", "Tensor Active", "DRAM Read Bandwidth", "DRAM Write Bandwidth", "GPC Clock Frequency"]


@register_plot("gpu_metrics_trace", group="edge", title="GPU hardware metrics over time (Nsight Systems)", needs="nsys capture with --gpu-metrics-device",
               options={"metrics": Opt("list", _DEFAULT_GPU_METRICS, "Substrings of GPU metric names."), "max_points": Opt("int", 4000, "Downsample each series to about this many points.")})  # fmt: skip
def gpu_metrics_trace(ctx: DataContext, o: dict[str, Any]) -> Chart:
    """SM activity, tensor activity, DRAM bandwidth and clocks sampled during the run, time relative to the first step."""
    g = ctx.frame("gpu_metrics", o)
    names = g["metric_name"].unique().to_list()
    panels = []
    for needle in o["metrics"]:
        match = sorted(n for n in names if needle.lower() in n.lower())
        if not match:
            continue
        name = match[0]
        traces = []
        for li, label in enumerate(ordered(g, "label")):
            for rank in sorted(g.filter(pl.col("label") == label)["rank"].unique().to_list()):
                s = g.filter((pl.col("label") == label) & (pl.col("rank") == rank) & (pl.col("metric_name") == name)).sort("t_ms")
                if s.is_empty():
                    continue
                if s.height > o["max_points"]:
                    s = s.gather_every(-(-s.height // int(o["max_points"])))
                nm = label if g.filter(pl.col("label") == label)["rank"].n_unique() == 1 else f"{label} r{rank}"
                traces.append(Trace("line", x=s["t_ms"].to_list(), y=s["value"].to_list(), name=nm, style={"ci": li, "width": 1.0}, meta={"label": label, "rank": int(rank)}))
        panels.append(Panel(traces=traces, title=name, xlabel="time since first step (ms)"))
    if not panels:
        raise NotApplicable(f"none of {o['metrics']} matches a captured GPU metric")
    return grid("GPU hardware metrics", panels, max_cols=2)
