"""Optional aspect that wraps Meta's Holistic Trace Analysis (HTA, PyPI `traceinsight`) for Kineto traces.

HTA already implements GPU idle-time attribution, kernel-launch statistics and stream queue-length analysis, so
this module only adapts its output to CTA's summary schema instead of reimplementing it. It runs on the whole raw
trace (CTA's --warmup-steps/--max-steps do not apply) and is opt-in: ``--aspects hta_kineto``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl

from cta.core.models import AspectResult, CompareConfig, TrialGroup
from cta.core.registry import register_aspect
from cta.core.stats import compare_scalar_metrics

try:
    from hta.trace_analysis import TraceAnalysis
except ImportError:  # optional dependency: pip install "compute-trace-analysis[hta]"
    TraceAnalysis = None

_METRICS: list[tuple[str, str, bool]] = [
    ("idle_host_wait_us", "us", True),
    ("idle_kernel_wait_us", "us", True),
    ("idle_other_us", "us", True),
    ("launch_delay_mean_us", "us", True),
    ("launch_delay_max_us", "us", True),
    ("queue_length_mean", "ops", False),
    ("queue_length_max", "ops", False),
]


@contextmanager
def _trace_dir(paths: tuple[Path, ...]) -> Iterator[str]:
    """HTA reads a directory of per-rank traces; expose the artifact's files through a temp dir."""
    with tempfile.TemporaryDirectory(prefix="cta_hta_") as tmp:
        for p in paths:
            dst = Path(tmp) / p.name
            try:
                dst.symlink_to(p)
            except OSError:
                shutil.copy2(p, dst)
        yield tmp


def _to_polars(df: Any) -> pl.DataFrame:
    flat = df.copy()
    if getattr(flat.columns, "nlevels", 1) > 1:
        flat.columns = ["_".join(str(x) for x in col if x) for col in flat.columns]
    return pl.from_pandas(flat.reset_index())


def _run_hta(paths: tuple[Path, ...]) -> tuple[dict[str, float], dict[str, pl.DataFrame]]:
    logging.getLogger("hta").setLevel(logging.WARNING)
    with _trace_dir(paths) as tdir:
        ta = TraceAnalysis(trace_dir=tdir)
        idle_df, _ = ta.get_idle_time_breakdown(visualize=False)
        launch = ta.get_cuda_kernel_launch_stats(visualize=False)
        queue = ta.get_queue_length_summary()
        membw = ta.get_memory_bw_summary()

    metrics: dict[str, float] = {}
    for cat in ("host_wait", "kernel_wait", "other"):
        metrics[f"idle_{cat}_us"] = float(idle_df.loc[idle_df["idle_category"] == cat, "idle_time"].sum())
    delays = [float(d) for df in launch.values() for d in df["launch_delay"]]
    if delays:
        metrics["launch_delay_mean_us"] = sum(delays) / len(delays)
        metrics["launch_delay_max_us"] = max(delays)
    if queue is not None and not queue.empty:
        metrics["queue_length_mean"] = float(queue[("queue_length", "mean")].mean())
        metrics["queue_length_max"] = float(queue[("queue_length", "max")].max())

    details = {"idle_time": _to_polars(idle_df)}
    if launch:
        details["launch_stats"] = pl.concat(
            [_to_polars(df).with_columns(pl.lit(rank).alias("rank")) for rank, df in launch.items()],
            how="diagonal_relaxed",
        )
    if queue is not None and not queue.empty:
        details["queue_length"] = _to_polars(queue)
    if membw is not None and not membw.empty:
        details["memory_bw"] = _to_polars(membw)
    return metrics, details


if TraceAnalysis is not None:

    @register_aspect(
        "hta_kineto",
        title="HTA GPU Health: Idle Attribution, Launch Delay & Queue Length",
        required_formats=("kineto",),
        default_enabled=False,
    )
    class HtaKinetoAspect:
        name = "hta_kineto"
        title = "HTA GPU Health: Idle Attribution, Launch Delay & Queue Length"
        required_formats = ("kineto",)
        any_of_formats = False
        default_enabled = False

        def run(
            self,
            groups: tuple[TrialGroup, ...],
            tables: dict[str, list[dict[str, Path]]],
            config: CompareConfig,
        ) -> AspectResult:
            labels = [g.label for g in groups]
            trials: list[dict[str, list[float]]] = []
            details: dict[str, pl.DataFrame] = {}
            for g in groups:
                acc: dict[str, list[float]] = {key: [] for key, _, _ in _METRICS}
                for t_idx, bundle in enumerate(g.trials):
                    metrics, det = _run_hta(bundle.artifacts["kineto"].paths)
                    for key, val in metrics.items():
                        acc[key].append(val)
                    if t_idx == 0:  # raw HTA tables for the first trial of each group
                        details.update({f"{name}[{g.label}]": df for name, df in det.items()})
                trials.append(acc)
            summary = compare_scalar_metrics(_METRICS, labels, trials, alpha=config.alpha)
            return AspectResult(name=self.name, title=self.title, summary=summary, details=details)
