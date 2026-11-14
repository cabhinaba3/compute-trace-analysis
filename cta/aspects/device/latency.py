"""The ``latency_throughput`` aspect: per-step latency and throughput."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta.aspects.device import step_slicing
from cta.core import models
from cta.core import registry
from cta.core import stats


@registry.register_aspect(
    "latency_throughput",
    title="End-to-End Latency, Tail Percentiles & Throughput",
    required_formats=step_slicing.TIMELINE_FORMATS,
    any_of_formats=True,
)
class LatencyThroughputAspect:
    """Compares step latency percentiles, throughput and CUDA runtime time.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide a step table.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "latency_throughput"
    title = "End-to-End Latency, Tail Percentiles & Throughput"
    required_formats = step_slicing.TIMELINE_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares per-step latency and throughput across the trial groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: step slicing and the significance
                level.

        Returns:
            Summary metrics only; trials without steps are skipped.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "mean_step_ms": [],
                "median_step_ms": [],
                "p90_step_ms": [],
                "p95_step_ms": [],
                "p99_step_ms": [],
                "throughput_per_s": [],
                "total_steps": [],
                "cuda_runtime_ms": [],
            }
            for table_paths in tables[group.group_id]:
                steps_raw, kernels_raw, runtime_calls = (
                    step_slicing.pick_timeline_tables(table_paths)
                )
                steps, _ = step_slicing.slice_steps_and_kernels(
                    steps_raw, kernels_raw, config
                )
                if steps.is_empty():
                    continue
                step_ms = (
                    steps["duration_ns"].to_numpy().astype(np.float64) / 1e6
                )
                # Ranks have independent clocks: throughput is steps/wall per
                # rank, averaged over ranks.
                per_rank = steps.group_by("rank").agg(
                    pl.len().alias("n"),
                    (pl.col("end_ns").max() - pl.col("start_ns").min()).alias(
                        "wall_ns"
                    ),
                )
                wall_s = np.maximum(1e-9, per_rank["wall_ns"].to_numpy() / 1e9)
                throughput_per_s = float(
                    np.mean(per_rank["n"].to_numpy() / wall_s)
                )

                metric_values["mean_step_ms"].append(float(np.mean(step_ms)))
                metric_values["median_step_ms"].append(
                    float(np.median(step_ms))
                )
                metric_values["p90_step_ms"].append(
                    float(np.percentile(step_ms, 90))
                )
                metric_values["p95_step_ms"].append(
                    float(np.percentile(step_ms, 95))
                )
                metric_values["p99_step_ms"].append(
                    float(np.percentile(step_ms, 99))
                )
                metric_values["throughput_per_s"].append(throughput_per_s)
                # Steps per rank, not summed over ranks.
                metric_values["total_steps"].append(
                    float(per_rank["n"].to_numpy().mean())
                )
                runtime_ms = (
                    float(runtime_calls["duration_ns"].to_numpy().sum() / 1e6)
                    if runtime_calls.height
                    else 0.0
                )
                metric_values["cuda_runtime_ms"].append(runtime_ms)
            trials.append(metric_values)

        summary = stats.compare_scalar_metrics(
            [
                ("mean_step_ms", "ms", True),
                ("median_step_ms", "ms", True),
                ("p90_step_ms", "ms", True),
                ("p95_step_ms", "ms", True),
                ("p99_step_ms", "ms", True),
                ("throughput_per_s", "steps/s", False),
                ("total_steps", "count", False),
                ("cuda_runtime_ms", "ms", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name, title=self.title, summary=summary
        )
