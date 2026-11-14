"""The ``os_runtime`` aspect: host OS-runtime call time from Nsight Systems."""

from __future__ import annotations

import pathlib

import polars as pl

from cta.core import models
from cta.core import registry
from cta.core import stats

# Lower-cased call-name patterns of the lock/sync and sleep/poll families.
_LOCK_CALL_PATTERN = "mutex|cond|rwlock|sem|futex|spin"
_POLL_CALL_PATTERN = "sleep|poll|epoll|select|wait"


@registry.register_aspect(
    "os_runtime",
    title="OS Runtime Lock & Thread Contention",
    required_formats=("nsys",),
)
class OsRuntimeAspect:
    """Compares host OS-runtime call time, lock waits and sleeps/polls.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Nsight Systems, which records OS-runtime calls.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "os_runtime"
    title = "OS Runtime Lock & Thread Contention"
    required_formats = ("nsys",)
    any_of_formats = False

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares OS-runtime call time across the trial groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics plus a ``by_syscall`` detail table.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        call_totals: list[pl.DataFrame] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_osrt_ms": [],
                "lock_sync_ms": [],
                "sleep_poll_ms": [],
                "osrt_calls": [],
            }
            trial_calls: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                osrt_calls = pl.read_parquet(table_paths["nsys.osrt"])
                if osrt_calls.is_empty():
                    continue
                metric_values["total_osrt_ms"].append(
                    float(osrt_calls["duration_ns"].to_numpy().sum() / 1e6)
                )
                lower_name = pl.col("name").str.to_lowercase()
                lock_calls = osrt_calls.filter(
                    lower_name.str.contains(_LOCK_CALL_PATTERN)
                )
                poll_calls = osrt_calls.filter(
                    lower_name.str.contains(_POLL_CALL_PATTERN)
                )
                metric_values["lock_sync_ms"].append(
                    float(lock_calls["duration_ns"].to_numpy().sum() / 1e6)
                    if lock_calls.height
                    else 0.0
                )
                metric_values["sleep_poll_ms"].append(
                    float(poll_calls["duration_ns"].to_numpy().sum() / 1e6)
                    if poll_calls.height
                    else 0.0
                )
                metric_values["osrt_calls"].append(
                    float(osrt_calls["calls"].to_numpy().sum())
                )
                trial_calls.append(
                    osrt_calls.with_columns(
                        (pl.col("duration_ns").cast(pl.Float64) / 1e6).alias(
                            "duration_ms"
                        )
                    ).select("name", "duration_ms", "calls")
                )
            trials.append(metric_values)
            call_totals.append(
                pl.concat(trial_calls)
                .group_by("name")
                .agg(
                    pl.col("duration_ms").mean(),
                    pl.col("calls").mean().round(0).cast(pl.Int64),
                )
                if trial_calls
                else pl.DataFrame({"name": [], "duration_ms": [], "calls": []})
            )

        summary = stats.compare_scalar_metrics(
            [
                ("total_osrt_ms", "ms", True),
                ("lock_sync_ms", "ms", True),
                ("sleep_poll_ms", "ms", True),
                ("osrt_calls", "calls", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        by_syscall = stats.compare_entity_tables(
            labels,
            call_totals,
            key_column="name",
            value_column="duration_ms",
            count_column="calls",
        )
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details={"by_syscall": by_syscall},
        )
