"""The ``kernel_diff`` aspect: kernel time by kernel, op family and scope."""

from __future__ import annotations

import pathlib

import polars as pl

from cta.aspects.device import step_slicing
from cta.core import models
from cta.core import registry
from cta.core import stats

_KERNEL_FORMATS = ("nsys", "kineto", "jax")


def with_scope_level(kernels: pl.DataFrame, depth: int | None) -> pl.DataFrame:
    """Returns the kernels with ``scope`` re-picked at a coarser level.

    ``scope_path`` lists the enclosing ranges joined by ``" > "``, outermost
    first. The new ``scope`` is the ``depth``-th of them, clamped to the
    innermost; kernels outside every range keep an empty scope.

    Args:
        kernels: Kernel table; returned unchanged without a ``scope_path``
            column.
        depth: 1-based level counted from the outermost range, or None to
            keep the existing ``scope``.

    Returns:
        The kernel table with the re-picked ``scope`` column.

    Raises:
        ValueError: If ``depth`` is below 1 and ``kernels`` has a
            ``scope_path`` column.
    """
    if depth is None or "scope_path" not in kernels.columns:
        return kernels
    if depth < 1:
        raise ValueError(f"scope_depth must be >= 1, got {depth}")
    path_parts = pl.col("scope_path").str.split(" > ")
    scope_at_depth = path_parts.list.get(depth - 1, null_on_oob=True).fill_null(
        path_parts.list.last()
    )
    return kernels.with_columns(
        pl.when(pl.col("scope_path") == "")
        .then(pl.lit(""))
        .otherwise(scope_at_depth)
        .alias("scope")
    )


def _category_ms(kernels: pl.DataFrame, category: str) -> float:
    """Returns the total duration in ms of the kernels in ``category``."""
    durations_ns = kernels.filter(pl.col("category") == category)["duration_ns"]
    return float(durations_ns.to_numpy().sum() / 1e6)


def _time_by(kernels: pl.DataFrame, key: str) -> pl.DataFrame:
    """Sums kernel time (``duration_ms``) and ``launches`` per ``key``."""
    return kernels.group_by(key).agg(
        (pl.col("duration_ns").sum().cast(pl.Float64) / 1e6).alias(
            "duration_ms"
        ),
        pl.len().alias("launches"),
    )


def _mean_over_trials(
    trial_totals: list[pl.DataFrame], key: str
) -> pl.DataFrame:
    """Averages per-trial ``_time_by`` frames; empty frame if there are none."""
    if not trial_totals:
        return pl.DataFrame({key: [], "duration_ms": [], "launches": []})
    return (
        pl.concat(trial_totals)
        .group_by(key)
        .agg(
            pl.col("duration_ms").mean(),
            pl.col("launches").mean().round(0).cast(pl.Int64),
        )
    )


@registry.register_aspect(
    "kernel_diff",
    title="Kernel & Operator Family Diff",
    required_formats=_KERNEL_FORMATS,
    any_of_formats=True,
)
class KernelDiffAspect:
    """Compares GPU kernel time by category, kernel, op family and scope.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide a kernel table.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "kernel_diff"
    title = "Kernel & Operator Family Diff"
    required_formats = _KERNEL_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares kernel time across the trial groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: step slicing, scope depth and the
                significance level.

        Returns:
            Summary metrics plus ``by_family`` and ``by_kernel`` detail
            tables, and ``by_scope`` when any kernel ran inside a scope.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        family_totals: list[pl.DataFrame] = []
        kernel_totals: list[pl.DataFrame] = []
        scope_totals: list[pl.DataFrame] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_kernel_ms": [],
                "compute_kernel_ms": [],
                "comm_kernel_ms": [],
                "memcpy_ms": [],
                "kernel_launches": [],
                "unique_kernels": [],
            }
            trial_families: list[pl.DataFrame] = []
            trial_kernels: list[pl.DataFrame] = []
            trial_scopes: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                trace_format = next(
                    candidate
                    for candidate in _KERNEL_FORMATS
                    if f"{candidate}.kernels" in table_paths
                )
                steps_raw = pl.read_parquet(
                    table_paths[f"{trace_format}.steps"]
                )
                kernels_raw = pl.read_parquet(
                    table_paths[f"{trace_format}.kernels"]
                )
                _, kernels = step_slicing.slice_steps_and_kernels(
                    steps_raw, kernels_raw, config
                )

                if kernels.is_empty():
                    for values in metric_values.values():
                        values.append(0.0)
                    continue

                total_ms = float(kernels["duration_ns"].to_numpy().sum() / 1e6)
                metric_values["total_kernel_ms"].append(total_ms)
                metric_values["compute_kernel_ms"].append(
                    _category_ms(kernels, "compute")
                )
                metric_values["comm_kernel_ms"].append(
                    _category_ms(kernels, "comm")
                )
                metric_values["memcpy_ms"].append(
                    _category_ms(kernels, "memcpy")
                )
                metric_values["kernel_launches"].append(float(kernels.height))
                metric_values["unique_kernels"].append(
                    float(kernels["canonical_name"].n_unique())
                )

                trial_families.append(_time_by(kernels, "op_family"))
                trial_kernels.append(_time_by(kernels, "canonical_name"))
                kernels = with_scope_level(kernels, config.scope_depth)
                scoped = kernels.filter(pl.col("scope") != "")
                if scoped.height:
                    trial_scopes.append(_time_by(scoped, "scope"))

            trials.append(metric_values)
            family_totals.append(_mean_over_trials(trial_families, "op_family"))
            kernel_totals.append(
                _mean_over_trials(trial_kernels, "canonical_name")
            )
            scope_totals.append(_mean_over_trials(trial_scopes, "scope"))

        summary = stats.compare_scalar_metrics(
            [
                ("total_kernel_ms", "ms", True),
                ("compute_kernel_ms", "ms", True),
                ("comm_kernel_ms", "ms", True),
                ("memcpy_ms", "ms", True),
                ("kernel_launches", "calls", True),
                ("unique_kernels", "count", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        details = {
            "by_family": stats.compare_entity_tables(
                labels,
                family_totals,
                key_column="op_family",
                value_column="duration_ms",
                count_column="launches",
            ),
            "by_kernel": stats.compare_entity_tables(
                labels,
                kernel_totals,
                key_column="canonical_name",
                value_column="duration_ms",
                count_column="launches",
            ),
        }
        if any(totals.height > 0 for totals in scope_totals):
            details["by_scope"] = stats.compare_entity_tables(
                labels,
                scope_totals,
                key_column="scope",
                value_column="duration_ms",
                count_column="launches",
            )

        return models.AspectResult(
            name=self.name, title=self.title, summary=summary, details=details
        )
