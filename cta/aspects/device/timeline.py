"""The ``timeline_overlap`` aspect: compute, comm, copy and idle GPU time."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta.aspects.device import step_slicing
from cta.core import intervals
from cta.core import models
from cta.core import registry
from cta.core import stats

# Summary metric and the regex its lower-cased GPU-metric names match.
_GPU_METRIC_PATTERNS = (
    ("sm_active_median_pct", r"sms? active"),
    ("tensor_active_median_pct", "tensor active"),
    ("sm_issue_median_pct", "sm issue"),
)


def _overlap_per_device(
    steps: pl.DataFrame, kernels: pl.DataFrame
) -> tuple[int, int, int, int, int]:
    """Sums compute/comm overlap over every (rank, device) timeline.

    Each device is its own timeline and each rank has its own clock, so
    overlap is computed per (rank, device) inside that rank's step window and
    then summed.

    Args:
        steps: Kept steps, used for each rank's window.
        kernels: Kept kernels with ``rank``, ``device``, ``category``,
            ``start_ns`` and ``end_ns`` columns.

    Returns:
        ``(compute_only, comm_only, overlap, idle, window)`` in ns.
    """
    window_by_rank = {
        int(row["rank"]): (int(row["w0"]), int(row["w1"]))
        for row in steps.group_by("rank")
        .agg(
            pl.col("start_ns").min().alias("w0"),
            pl.col("end_ns").max().alias("w1"),
        )
        .iter_rows(named=True)
    }
    totals_ns = [0, 0, 0, 0, 0]
    for (rank, _device), device_kernels in kernels.group_by(
        ["rank", "device"], maintain_order=True
    ):
        window_start_ns, window_end_ns = window_by_rank.get(
            int(rank), (None, None)
        )
        compute = device_kernels.filter(pl.col("category") == "compute")
        comm = device_kernels.filter(
            pl.col("category").is_in(["comm", "memcpy"])
        )
        overlap = intervals.compute_timeline_overlap(
            compute["start_ns"].to_numpy(),
            compute["end_ns"].to_numpy(),
            comm["start_ns"].to_numpy(),
            comm["end_ns"].to_numpy(),
            window_start=window_start_ns,
            window_end=window_end_ns,
        )
        parts_ns = (
            overlap.compute_only_ns,
            overlap.comm_only_ns,
            overlap.overlap_ns,
            overlap.idle_ns,
            overlap.window_ns,
        )
        for i, part_ns in enumerate(parts_ns):
            totals_ns[i] += part_ns
    return totals_ns[0], totals_ns[1], totals_ns[2], totals_ns[3], totals_ns[4]


@registry.register_aspect(
    "timeline_overlap",
    title="Timeline Utilization, Overlap & GPU Metrics",
    required_formats=step_slicing.TIMELINE_FORMATS,
    any_of_formats=True,
)
class TimelineOverlapAspect:
    """Compares GPU busy/idle time, comm overlap and sampled GPU metrics.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide a step table.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "timeline_overlap"
    title = "Timeline Utilization, Overlap & GPU Metrics"
    required_formats = step_slicing.TIMELINE_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares timeline utilisation across the trial groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: step slicing and the significance
                level.

        Returns:
            Summary metrics plus per-group time shares (``share_Compute``,
            ``share_Exposed-Comm``, ``share_Idle-Bubble``) as scalars.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "compute_only_ms": [],
                "exposed_comm_ms": [],
                "overlapped_comm_ms": [],
                "gpu_idle_ms": [],
                "comm_overlap_pct": [],
                "gpu_busy_pct": [],
                "sm_active_median_pct": [],
                "tensor_active_median_pct": [],
                "sm_issue_median_pct": [],
            }
            sums_ns = {"compute": 0.0, "comm": 0.0, "idle": 0.0, "window": 0.0}
            for table_paths in tables[group.group_id]:
                steps_raw, kernels_raw, _ = step_slicing.pick_timeline_tables(
                    table_paths
                )
                steps, kernels = step_slicing.slice_steps_and_kernels(
                    steps_raw, kernels_raw, config
                )
                if kernels.height:
                    compute_only, comm_only, overlap, idle, window = (
                        _overlap_per_device(steps, kernels)
                    )
                    comm_total = comm_only + overlap
                    metric_values["compute_only_ms"].append(compute_only / 1e6)
                    metric_values["exposed_comm_ms"].append(comm_only / 1e6)
                    metric_values["overlapped_comm_ms"].append(overlap / 1e6)
                    metric_values["gpu_idle_ms"].append(idle / 1e6)
                    metric_values["comm_overlap_pct"].append(
                        overlap / comm_total * 100.0 if comm_total else 0.0
                    )
                    metric_values["gpu_busy_pct"].append(
                        (window - idle) / window * 100.0 if window else 0.0
                    )
                    sums_ns["compute"] += compute_only + overlap
                    sums_ns["comm"] += comm_only
                    sums_ns["idle"] += idle
                    sums_ns["window"] += window

                if "nsys.gpu_metrics" in table_paths:
                    gpu_metrics = step_slicing.gpu_metrics_in_steps(
                        pl.read_parquet(table_paths["nsys.gpu_metrics"]), steps
                    )
                    if gpu_metrics.height:
                        lower_name = pl.col("metric_name").str.to_lowercase()
                        for metric, pattern in _GPU_METRIC_PATTERNS:
                            matching = gpu_metrics.filter(
                                lower_name.str.contains(pattern)
                            )
                            if matching.height:
                                metric_values[metric].append(
                                    float(
                                        np.median(matching["value"].to_numpy())
                                    )
                                )

            trials.append(metric_values)
            window_ns = sums_ns["window"] or 1.0
            scalars[group.label] = {
                "share_Compute": sums_ns["compute"] / window_ns,
                "share_Exposed-Comm": sums_ns["comm"] / window_ns,
                "share_Idle-Bubble": sums_ns["idle"] / window_ns,
            }

        summary = stats.compare_scalar_metrics(
            [
                ("compute_only_ms", "ms", True),
                ("exposed_comm_ms", "ms", True),
                ("overlapped_comm_ms", "ms", False),
                ("gpu_idle_ms", "ms", True),
                ("comm_overlap_pct", "%", False),
                ("gpu_busy_pct", "%", False),
                ("sm_active_median_pct", "%", False),
                ("tensor_active_median_pct", "%", False),
                ("sm_issue_median_pct", "%", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name, title=self.title, summary=summary, scalars=scalars
        )
