"""Step-window slicing shared by the device aspects and the plot data layer."""

from __future__ import annotations

import pathlib
import warnings

import polars as pl

from cta.core import models

# Formats whose adapters write a ``<format>.steps`` table, in the order
# ``pick_timeline_tables`` tries them.
TIMELINE_FORMATS = ("nsys", "kineto", "jax", "trtexec_timing")


def _step_windows(steps: pl.DataFrame) -> pl.DataFrame:
    """Returns per-rank step windows: ``_w0`` first start, ``_w1`` last end."""
    return steps.group_by("rank").agg(
        pl.col("start_ns").min().alias("_w0"),
        pl.col("end_ns").max().alias("_w1"),
    )


def _kernels_in_steps(
    kernels: pl.DataFrame, kept_steps: pl.DataFrame
) -> pl.DataFrame:
    """Keeps kernels of a kept (rank, step) or inside their rank's window."""
    kept_step_keys = (
        kept_steps.select("rank", "step")
        .unique()
        .with_columns(pl.lit(True).alias("_ok"))
    )
    tagged_kernels = kernels.join(
        kept_step_keys, on=["rank", "step"], how="left"
    ).join(_step_windows(kept_steps), on="rank", how="left")
    in_window = (pl.col("start_ns") >= pl.col("_w0")) & (
        pl.col("end_ns") <= pl.col("_w1")
    )
    return tagged_kernels.filter(
        pl.col("_ok").fill_null(False) | in_window
    ).drop("_ok", "_w0", "_w1")


def slice_steps_and_kernels(
    steps: pl.DataFrame,
    kernels: pl.DataFrame,
    config: models.CompareConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Restricts steps and kernels to the requested ranks, devices and steps.

    Applies ``--ranks``/``--devices`` first, then ``--warmup-steps`` and
    ``--max-steps`` per rank (ranks have independent clocks). A kernel is
    kept when its (rank, step) was kept or when it lies inside its rank's
    window of kept steps. If no rank has more steps than ``--warmup-steps``,
    emits a ``RuntimeWarning`` and skips nothing.

    Args:
        steps: Step table with ``rank``, ``step``, ``start_ns`` and
            ``end_ns`` columns.
        kernels: Kernel table with ``rank``, ``device``, ``step``,
            ``start_ns`` and ``end_ns`` columns; may be empty.
        config: Comparison settings that select ranks, devices and steps.

    Returns:
        The kept steps (sorted by rank and start time when any remain) and
        the kept kernels.
    """
    if config.ranks is not None:
        steps = steps.filter(pl.col("rank").is_in(list(config.ranks)))
        if kernels.height:
            kernels = kernels.filter(pl.col("rank").is_in(list(config.ranks)))
    if config.devices is not None and kernels.height:
        kernels = kernels.filter(pl.col("device").is_in(list(config.devices)))
    if steps.is_empty():
        return steps, kernels
    kept_steps = steps.sort("rank", "start_ns").with_columns(
        pl.col("start_ns").cum_count().over("rank").alias("_i")
    )
    steps_on_longest_rank = int(kept_steps["_i"].to_numpy().max())
    if config.warmup_steps > 0 and steps_on_longest_rank <= config.warmup_steps:
        warnings.warn(
            f"--warmup-steps={config.warmup_steps} but a rank has at most "
            f"{steps_on_longest_rank} step(s) "
            "(no NVTX step markers?); nothing was skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif config.warmup_steps > 0:
        kept_steps = kept_steps.filter(pl.col("_i") > config.warmup_steps)
    if config.max_steps is not None and config.max_steps > 0:
        kept_steps = (
            kept_steps.with_columns(
                pl.col("start_ns").cum_count().over("rank").alias("_j")
            )
            .filter(pl.col("_j") <= config.max_steps)
            .drop("_j")
        )
    kept_steps = kept_steps.drop("_i")

    if kernels.is_empty() or kept_steps.height == steps.height:
        return kept_steps, kernels
    return kept_steps, _kernels_in_steps(kernels, kept_steps)


def gpu_metrics_in_steps(
    gpu_metrics: pl.DataFrame, steps: pl.DataFrame
) -> pl.DataFrame:
    """Keeps only the GPU-metric samples inside each rank's step window.

    The sampler runs for the whole capture, so medians over every sample
    mostly measure idle time before and after the profiled steps.

    Args:
        gpu_metrics: GPU-metric samples with ``rank`` and ``timestamp_ns``
            columns.
        steps: Step table with ``rank``, ``start_ns`` and ``end_ns``.

    Returns:
        The samples inside the step windows. Returns all samples when there
        are no steps, the samples have no ``rank`` column or nothing falls
        inside the windows (e.g. sources whose sample timestamps are not on
        the kernel clock).
    """
    if (
        gpu_metrics.is_empty()
        or steps.is_empty()
        or "rank" not in gpu_metrics.columns
    ):
        return gpu_metrics
    inside = (
        gpu_metrics.join(_step_windows(steps), on="rank", how="inner")
        .filter(pl.col("timestamp_ns").is_between(pl.col("_w0"), pl.col("_w1")))
        .drop("_w0", "_w1")
    )
    return inside if inside.height else gpu_metrics


def _read_if_present(
    table_paths: dict[str, pathlib.Path], key: str
) -> pl.DataFrame:
    """Reads the ``key`` table, or returns an empty frame when it is absent."""
    if key in table_paths:
        return pl.read_parquet(table_paths[key])
    return pl.DataFrame()


def pick_timeline_tables(
    table_paths: dict[str, pathlib.Path],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Reads the step, kernel and runtime tables of one trial.

    Uses the first format in ``TIMELINE_FORMATS`` that has a steps table.

    Args:
        table_paths: Mapping from ``<format>.<table>`` keys to Parquet paths for
            one trial.

    Returns:
        The ``(steps, kernels, runtime)`` tables; a missing kernel or
        runtime table is an empty frame.

    Raises:
        KeyError: If no timeline format has a steps table.
    """
    for trace_format in TIMELINE_FORMATS:
        if f"{trace_format}.steps" in table_paths:
            steps = pl.read_parquet(table_paths[f"{trace_format}.steps"])
            kernels = _read_if_present(table_paths, f"{trace_format}.kernels")
            runtime = _read_if_present(table_paths, f"{trace_format}.runtime")
            return steps, kernels, runtime
    raise KeyError("No timeline steps table found in bundle")
