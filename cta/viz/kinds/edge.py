"""Time-series plot kinds from sampled signals.

Covers Jetson power, utilisation and thermals (jtop) and Nsight Systems GPU
hardware metrics.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from cta.viz import data
from cta.viz import ir
from cta.viz import spec
from cta.viz.kinds import chart_helpers

# Panel title of every plottable jtop column; the order feeds the help text.
_TITLE_BY_JTOP_METRIC = {
    "power_total_w": "total power (W)",
    "power_gpu_w": "GPU power (W)",
    "gpu_util_pct": "GPU utilisation (%)",
    "cpu_util_pct": "CPU utilisation (%)",
    "gpu_temp_c": "GPU temperature (C)",
    "gpu_mem_mb": "GPU memory (MB)",
    "ram_mb": "RAM (MB)",
    "cumulative_energy_j": "cumulative energy (J)",
}


@spec.register_plot(
    "power_trace",
    group="edge",
    title="Power, utilisation and thermals over time",
    needs="jtop samples (edge bundle)",
    options={
        "metrics": spec.Opt(
            "list",
            ["power_total_w", "gpu_util_pct", "gpu_temp_c", "gpu_mem_mb"],
            f"jtop columns: {', '.join(_TITLE_BY_JTOP_METRIC)}.",
        )
    },
)
def power_trace(ctx: data.DataContext, options: dict[str, Any]) -> ir.Chart:
    """Plots each sampled jtop signal against time, one line per group.

    Shows throttling, power caps and memory growth; each trial of a group
    gets its own line.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart with one panel per requested signal.

    Raises:
        ValueError: A requested signal is not a known jtop column.
    """
    samples = ctx.frame("jtop", options)
    unknown = [
        metric
        for metric in options["metrics"]
        if metric not in _TITLE_BY_JTOP_METRIC
    ]
    if unknown:
        raise ValueError(
            f"unknown jtop metric(s) {unknown}; choose from"
            f" {', '.join(_TITLE_BY_JTOP_METRIC)}"
        )
    labels = chart_helpers.unique_in_order(samples, "label")
    panels = []
    for metric in options["metrics"]:
        traces = []
        for color_index, label in enumerate(labels):
            traces.extend(_per_trial_lines(samples, label, metric, color_index))
        panels.append(
            ir.Panel(
                traces=traces,
                title=_TITLE_BY_JTOP_METRIC[metric],
                xlabel="time (s)",
                ylabel="",
            )
        )
    return chart_helpers.grid("Sampled signals", panels, max_cols=2)


def _per_trial_lines(
    samples: pl.DataFrame, label: str, metric: str, color_index: int
) -> list[ir.Trace]:
    """Builds one line per trial of a group; only trial 0 enters the legend."""
    group_samples = samples.filter(pl.col("label") == label)
    traces = []
    for trial in sorted(group_samples["trial"].unique().to_list()):
        trial_samples = group_samples.filter(pl.col("trial") == trial).sort(
            "elapsed_s"
        )
        traces.append(
            ir.Trace(
                "line",
                x=trial_samples["elapsed_s"].to_list(),
                y=trial_samples[metric].to_list(),
                name=label,
                style={"ci": color_index, "width": 1.1, "nolegend": trial > 0},
                meta={"label": label},
            )
        )
    return traces


_DEFAULT_GPU_METRICS = [
    "SMs Active",
    "SM Issue",
    "Tensor Active",
    "DRAM Read Bandwidth",
    "DRAM Write Bandwidth",
    "GPC Clock Frequency",
]


@spec.register_plot(
    "gpu_metrics_trace",
    group="edge",
    title="GPU hardware metrics over time (Nsight Systems)",
    needs="nsys capture with --gpu-metrics-device",
    options={
        "metrics": spec.Opt(
            "list", _DEFAULT_GPU_METRICS, "Substrings of GPU metric names."
        ),
        "max_points": spec.Opt(
            "int", 4000, "Downsample each series to about this many points."
        ),
    },
)
def gpu_metrics_trace(
    ctx: data.DataContext, options: dict[str, Any]
) -> ir.Chart:
    """Plots GPU hardware metrics sampled during the run.

    Covers SM activity, tensor activity, DRAM bandwidth and clocks, with
    time relative to the first step. Each requested substring selects the
    first matching metric name in sorted order.

    Args:
        ctx: Data access for the report being plotted.
        options: Resolved options of this kind.

    Returns:
        A chart with one panel per matched metric, one line per group and
        rank.

    Raises:
        spec.NotApplicableError: No requested substring matches a captured
            metric.
    """
    samples = ctx.frame("gpu_metrics", options)
    metric_names = samples["metric_name"].unique().to_list()
    labels = chart_helpers.unique_in_order(samples, "label")
    panels = []
    for needle in options["metrics"]:
        matches = sorted(
            name for name in metric_names if needle.lower() in name.lower()
        )
        if not matches:
            continue
        metric_name = matches[0]
        traces = []
        for color_index, label in enumerate(labels):
            traces.extend(
                _per_rank_lines(
                    samples,
                    label,
                    metric_name,
                    color_index,
                    options["max_points"],
                )
            )
        panels.append(
            ir.Panel(
                traces=traces,
                title=metric_name,
                xlabel="time since first step (ms)",
            )
        )
    if not panels:
        raise spec.NotApplicableError(
            f"none of {options['metrics']} matches a captured GPU metric"
        )
    return chart_helpers.grid("GPU hardware metrics", panels, max_cols=2)


def _per_rank_lines(
    samples: pl.DataFrame,
    label: str,
    metric_name: str,
    color_index: int,
    max_points: int,
) -> list[ir.Trace]:
    """Builds one downsampled line per rank of a group for one GPU metric.

    Args:
        samples: GPU metric samples of every group.
        label: Group to draw.
        metric_name: Exact GPU metric name.
        color_index: Colour of the group.
        max_points: Each line keeps every n-th sample so that it has at most
            about this many points.

    Returns:
        The lines; ranks without samples of the metric are skipped. Lines are
        named after the rank only when the group has several ranks.
    """
    group_samples = samples.filter(pl.col("label") == label)
    has_one_rank = group_samples["rank"].n_unique() == 1
    traces = []
    for rank in sorted(group_samples["rank"].unique().to_list()):
        rank_samples = group_samples.filter(
            (pl.col("rank") == rank) & (pl.col("metric_name") == metric_name)
        ).sort("t_ms")
        if rank_samples.is_empty():
            continue
        if rank_samples.height > max_points:
            # Ceiling division: the stride that keeps at most max_points.
            stride = -(-rank_samples.height // int(max_points))
            rank_samples = rank_samples.gather_every(stride)
        traces.append(
            ir.Trace(
                "line",
                x=rank_samples["t_ms"].to_list(),
                y=rank_samples["value"].to_list(),
                name=label if has_one_rank else f"{label} r{rank}",
                style={"ci": color_index, "width": 1.0},
                meta={"label": label, "rank": int(rank)},
            )
        )
    return traces
