"""Optional ``hta_kineto`` aspect built on Meta's Holistic Trace Analysis.

HTA (PyPI `traceinsight`) already implements GPU idle-time attribution,
kernel-launch statistics and stream queue-length analysis for Kineto traces,
so this module only adapts its output to CTA's summary schema instead of
reimplementing it. It runs on the whole raw trace (CTA's
--warmup-steps/--max-steps do not apply) and is opt-in:
``--aspects hta_kineto``.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import importlib.util
import logging
import pathlib
import shutil
import tempfile
from typing import Any

import polars as pl

from cta.core import models
from cta.core import registry
from cta.core import stats

# Optional dependency (pip install "compute-trace-analysis[hta]"). find_spec
# avoids importing HTA, which drags in pandas and plotly (>1 s), until the
# aspect actually runs.
_HTA_AVAILABLE = importlib.util.find_spec("hta") is not None

_METRICS: list[tuple[str, str, bool]] = [
    ("idle_host_wait_us", "us", True),
    ("idle_kernel_wait_us", "us", True),
    ("idle_other_us", "us", True),
    ("launch_delay_mean_us", "us", True),
    ("launch_delay_max_us", "us", True),
    ("queue_length_mean", "ops", False),
    ("queue_length_max", "ops", False),
]


@contextlib.contextmanager
def _trace_dir(paths: tuple[pathlib.Path, ...]) -> Iterator[str]:
    """Exposes the trace files through a temporary directory for HTA.

    HTA reads a directory of per-rank traces. Each file is symlinked into the
    directory, or copied where symlinks are not allowed.

    Args:
        paths: Trace files of one Kineto artifact.

    Yields:
        The temporary directory, removed on exit.
    """
    with tempfile.TemporaryDirectory(prefix="cta_hta_") as tmp_dir:
        for path in paths:
            link_path = pathlib.Path(tmp_dir) / path.name
            try:
                link_path.symlink_to(path)
            except OSError:
                shutil.copy2(path, link_path)
        yield tmp_dir


def _to_polars(df: Any) -> pl.DataFrame:
    """Converts an HTA pandas frame, flattening multi-level column names."""
    flat = df.copy()
    if getattr(flat.columns, "nlevels", 1) > 1:
        flat.columns = [
            "_".join(str(level) for level in column if level)
            for column in flat.columns
        ]
    return pl.from_pandas(flat.reset_index())


def _run_hta(
    paths: tuple[pathlib.Path, ...],
) -> tuple[dict[str, float], dict[str, pl.DataFrame]]:
    """Runs HTA's idle-time, launch, queue-length and memory-bandwidth analyses.

    Args:
        paths: Trace files of one Kineto artifact.

    Returns:
        The summary metrics HTA could compute (keys from ``_METRICS``) and
        its raw tables: always ``idle_time``, plus ``launch_stats``,
        ``queue_length`` and ``memory_bw`` when HTA returned data for them.
    """
    from hta import trace_analysis

    logging.getLogger("hta").setLevel(logging.WARNING)
    with _trace_dir(paths) as trace_dir:
        analysis = trace_analysis.TraceAnalysis(trace_dir=trace_dir)
        idle_time, _ = analysis.get_idle_time_breakdown(visualize=False)
        launch_stats_by_rank = analysis.get_cuda_kernel_launch_stats(
            visualize=False
        )
        queue_length = analysis.get_queue_length_summary()
        memory_bw = analysis.get_memory_bw_summary()

    metrics: dict[str, float] = {}
    for category in ("host_wait", "kernel_wait", "other"):
        is_category = idle_time["idle_category"] == category
        metrics[f"idle_{category}_us"] = float(
            idle_time.loc[is_category, "idle_time"].sum()
        )
    delays = [
        float(delay)
        for launch_stats in launch_stats_by_rank.values()
        for delay in launch_stats["launch_delay"]
    ]
    if delays:
        metrics["launch_delay_mean_us"] = sum(delays) / len(delays)
        metrics["launch_delay_max_us"] = max(delays)
    if queue_length is not None and not queue_length.empty:
        metrics["queue_length_mean"] = float(
            queue_length[("queue_length", "mean")].mean()
        )
        metrics["queue_length_max"] = float(
            queue_length[("queue_length", "max")].max()
        )

    details = {"idle_time": _to_polars(idle_time)}
    if launch_stats_by_rank:
        details["launch_stats"] = pl.concat(
            [
                _to_polars(launch_stats).with_columns(
                    pl.lit(rank).alias("rank")
                )
                for rank, launch_stats in launch_stats_by_rank.items()
            ],
            how="diagonal_relaxed",
        )
    if queue_length is not None and not queue_length.empty:
        details["queue_length"] = _to_polars(queue_length)
    if memory_bw is not None and not memory_bw.empty:
        details["memory_bw"] = _to_polars(memory_bw)
    return metrics, details


if _HTA_AVAILABLE:

    @registry.register_aspect(
        "hta_kineto",
        title="HTA GPU Health: Idle Attribution, Launch Delay & Queue Length",
        required_formats=("kineto",),
        default_enabled=False,
    )
    class HtaKinetoAspect:
        """Compares HTA's idle-time, launch-delay and queue-length metrics.

        Registered only when HTA is installed.

        Attributes:
            name: Aspect key used by ``--aspects`` and in reports.
            title: Report heading.
            required_formats: Kineto, the only trace format HTA reads.
            any_of_formats: Whether one of ``required_formats`` is enough.
            default_enabled: False: runs only when requested by name.
        """

        name = "hta_kineto"
        title = "HTA GPU Health: Idle Attribution, Launch Delay & Queue Length"
        required_formats = ("kineto",)
        any_of_formats = False
        default_enabled = False

        def run(
            self,
            groups: tuple[models.TrialGroup, ...],
            tables: dict[str, list[dict[str, pathlib.Path]]],
            config: models.CompareConfig,
        ) -> models.AspectResult:
            """Runs HTA on every trial and compares its metrics.

            HTA reads the raw Kineto traces of each trial, so ``tables`` is
            not used and step slicing does not apply.

            Args:
                groups: Trial groups to compare, baseline first.
                tables: Ingested tables per group id; unused.
                config: Comparison settings; only the significance level is
                    used.

            Returns:
                Summary metrics plus HTA's raw tables for the first trial of
                each group, keyed ``<table>[<label>]``.
            """
            labels = [group.label for group in groups]
            trials: list[dict[str, list[float]]] = []
            details: dict[str, pl.DataFrame] = {}
            for group in groups:
                metric_values: dict[str, list[float]] = {
                    key: [] for key, _, _ in _METRICS
                }
                for trial_index, trial_bundle in enumerate(group.trials):
                    metrics, trial_details = _run_hta(
                        trial_bundle.artifacts["kineto"].paths
                    )
                    for key, value in metrics.items():
                        metric_values[key].append(value)
                    # Raw HTA tables for the first trial of each group.
                    if trial_index == 0:
                        details.update(
                            {
                                f"{table_name}[{group.label}]": table
                                for table_name, table in trial_details.items()
                            }
                        )
                trials.append(metric_values)
            summary = stats.compare_scalar_metrics(
                _METRICS, labels, trials, alpha=config.alpha
            )
            return models.AspectResult(
                name=self.name,
                title=self.title,
                summary=summary,
                details=details,
            )
