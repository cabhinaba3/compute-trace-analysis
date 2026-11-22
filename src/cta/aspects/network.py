"""The ``network`` aspect: communication calls by operation and size."""

from __future__ import annotations

import pathlib

import polars as pl

from cta import models
from cta import plugins
from cta import table_schemas
from cta.analysis import network
from cta.analysis import stats
from cta.analysis import step_slicing


@plugins.register_aspect(
    "network",
    title="Network API: Bandwidth and Cross-Rank Wait by Operation and Size",
    required_formats=("nsys", "kineto", "jax"),
    any_of_formats=True,
)
class NetworkAspect(plugins.Aspect):
    """Compares the bandwidth of communication calls per operation and size.

    Calls are matched across groups by logical operation and message-size
    bucket, so different libraries running the same workload line up.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that can provide a ``network_ops`` table.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
    ) -> models.AspectResult:
        """Compares algorithm and bus bandwidth, time and wait across groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: step slicing and the significance
                level.

        Returns:
            Summary metrics per API, operation and size bucket, and two detail
            tables: time by API, and bus bandwidth by operation and bucket
            with the APIs pooled, for groups that each ran one library.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        per_api: list[pl.DataFrame] = []
        for group in groups:
            metrics: dict[str, list[float]] = {}
            group_rows: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                ops = _read_ops(table_paths, config)
                if ops.is_empty():
                    continue
                per_key = network.per_operation(ops)
                group_rows.append(per_key)
                _add_trial_metrics(metrics, ops, per_key)
            trials.append(metrics)
            per_api.append(
                pl.concat(group_rows)
                if group_rows
                else pl.DataFrame(schema=network.PER_OPERATION_SCHEMA)
            )

        summary = stats.compare_scalar_metrics(
            _metric_specs(trials), labels, trials, alpha=config.alpha
        )
        details = {
            "by_api": stats.compare_entity_tables(
                labels,
                [_totals(rows, "api", "op", "bucket") for rows in per_api],
                key_column="key",
                value_column="duration_ms",
                count_column="calls",
            ),
            # Calls of every API pooled, so a group that ran one library
            # lines up with a group that ran another.
            "by_operation": stats.compare_entity_tables(
                labels,
                [_bandwidth(rows) for rows in per_api],
                key_column="key",
                value_column="busbw",
                count_column="calls",
            ),
        }
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details=details if summary.height else {},
        )


def _read_ops(
    table_paths: dict[str, pathlib.Path], config: models.AnalysisConfig
) -> pl.DataFrame:
    """Reads one trial's network calls, restricted like the other aspects."""
    for trace_format in step_slicing.TIMELINE_FORMATS:
        key = f"{trace_format}.network_ops"
        if key not in table_paths:
            continue
        ops = pl.read_parquet(table_paths[key])
        steps_key = f"{trace_format}.steps"
        if steps_key in table_paths:
            _, ops = step_slicing.slice_steps_and_kernels(
                pl.read_parquet(table_paths[steps_key]), ops, config
            )
        return ops
    return pl.DataFrame(schema=table_schemas.NETWORK_OPS)


def _totals(rows: pl.DataFrame, *parts: str) -> pl.DataFrame:
    """Sums calls and time over the trials by the joined ``parts``."""
    return (
        rows.with_columns(
            pl.concat_str(list(parts), separator="|").alias("key")
        )
        .group_by("key")
        .agg(pl.col("calls").sum(), pl.col("duration_ms").sum())
    )


def _bandwidth(rows: pl.DataFrame) -> pl.DataFrame:
    """Pools the APIs: bus bandwidth in GB/s per operation and bucket."""
    return (
        rows.with_columns(
            pl.concat_str(["op", "bucket"], separator="|").alias("key")
        )
        .group_by("key")
        .agg(
            pl.col("calls").sum(),
            pl.col("weighted_bytes").sum(),
            pl.col("timed_ns").sum(),
        )
        .filter(pl.col("timed_ns") > 0)
        .select(
            "key",
            "calls",
            (pl.col("weighted_bytes") / pl.col("timed_ns")).alias("busbw"),
        )
    )


def _add_trial_metrics(
    metrics: dict[str, list[float]], ops: pl.DataFrame, per_key: pl.DataFrame
) -> None:
    """Appends one trial's values to the metric lists."""
    timed = ops.filter(~pl.col("posted"))
    for (api,), rows in ops.group_by("api"):
        per_rank_ns = (
            timed.filter(pl.col("api") == api)
            .group_by("rank")
            .agg(pl.col("duration_ns").sum())
        )
        metrics.setdefault(f"comm_time_ms|{api}", []).append(
            float(per_rank_ns["duration_ns"].to_numpy().mean() / 1e6)
            if per_rank_ns.height
            else 0.0
        )
        metrics.setdefault(f"comm_calls|{api}", []).append(float(rows.height))
    for row in per_key.iter_rows(named=True):
        key = f"{row['api']}|{row['op']}|{row['bucket']}"
        metrics.setdefault(f"time_ms|{key}", []).append(row["duration_ms"])
        if row["timed_bytes"] > 0 and row["timed_ns"] > 0:
            algbw = row["timed_bytes"] / row["timed_ns"]
            busbw = row["weighted_bytes"] / row["timed_ns"]
            metrics.setdefault(f"algbw_GBps|{key}", []).append(algbw)
            metrics.setdefault(f"busbw_GBps|{key}", []).append(busbw)
    for row in network.wait_skew_us(ops).iter_rows(named=True):
        metrics.setdefault(f"wait_skew_us|{row['api']}|{row['op']}", []).append(
            row["skew_us"]
        )
    for row in network.launch_gap_us(ops).iter_rows(named=True):
        metrics.setdefault(
            f"launch_gap_us|{row['api']}|{row['op']}", []
        ).append(row["gap_us"])


_UNITS = {
    "comm_time_ms": ("ms", True),
    "comm_calls": ("count", True),
    "time_ms": ("ms", True),
    "algbw_GBps": ("GB/s", False),
    "busbw_GBps": ("GB/s", False),
    "wait_skew_us": ("us", True),
    "launch_gap_us": ("us", True),
}


def _metric_specs(
    trials: list[dict[str, list[float]]],
) -> list[tuple[str, str, bool]]:
    """Lists every metric any trial has, with its unit and direction.

    Metrics come in the order of ``_UNITS``, then by name.
    """
    kinds = list(_UNITS)
    names = {name for trial in trials for name in trial}
    ordered = sorted(
        names, key=lambda name: (kinds.index(name.partition("|")[0]), name)
    )
    return [(name, *_UNITS[name.partition("|")[0]]) for name in ordered]
