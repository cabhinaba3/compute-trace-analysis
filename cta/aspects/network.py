"""The ``network_api`` aspect: communication calls by operation and size."""

from __future__ import annotations

import pathlib

import polars as pl

from cta.aspects.device import step_slicing
from cta.core import models
from cta.core import network
from cta.core import registry
from cta.core import stats


@registry.register_aspect(
    "network_api",
    title="Network API: Bandwidth and Cross-Rank Wait by Operation and Size",
    required_formats=("nsys", "kineto"),
    any_of_formats=True,
)
class NetworkApiAspect(registry.ComparisonAspect):
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
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares algorithm and bus bandwidth, time and wait across groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: step slicing and the significance
                level.

        Returns:
            Summary metrics per operation and size bucket, and a per-group
            table of totals for the same keys.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        totals: list[pl.DataFrame] = []
        for group in groups:
            metrics: dict[str, list[float]] = {}
            group_rows: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                ops = _read_ops(table_paths, config)
                if ops.is_empty():
                    continue
                per_key = per_operation(ops)
                group_rows.append(per_key)
                _add_trial_metrics(metrics, ops, per_key)
            trials.append(metrics)
            totals.append(
                pl.concat(group_rows)
                .group_by("key")
                .agg(
                    pl.col("calls").sum(),
                    pl.col("duration_ms").sum(),
                )
                if group_rows
                else pl.DataFrame(
                    schema={
                        "key": pl.String,
                        "calls": pl.UInt32,
                        "duration_ms": pl.Float64,
                    }
                )
            )

        summary = stats.compare_scalar_metrics(
            _metric_specs(trials), labels, trials, alpha=config.alpha
        )
        by_operation = stats.compare_entity_tables(
            labels,
            totals,
            key_column="key",
            value_column="duration_ms",
            count_column="calls",
        )
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details={"by_operation": by_operation} if summary.height else {},
        )


def _read_ops(
    table_paths: dict[str, pathlib.Path], config: models.CompareConfig
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
    return pl.DataFrame(schema=network.NETWORK_OPS_SCHEMA)


def per_operation(ops: pl.DataFrame) -> pl.DataFrame:
    """Aggregates calls by logical operation and message-size bucket.

    Args:
        ops: A ``network_ops`` table.

    Returns:
        One row per ``key`` (``op|bucket``) with ``calls``, ``duration_ms``,
        the total ``bytes`` and ``algbw`` / ``busbw`` in GB/s (0 where the
        trace records no sizes).
    """
    rank_count = max(1, ops["rank"].n_unique())
    sized = ops.with_columns(
        pl.Series(
            "bucket", network.bucket_bytes(ops["bytes"].to_numpy()), pl.String
        ),
        pl.when(pl.col("group_size") > 0)
        .then(pl.col("group_size"))
        .otherwise(rank_count)
        .alias("ranks"),
    ).with_columns(
        pl.struct("op", "ranks")
        .map_batches(
            lambda s: pl.Series(
                [
                    network.busbw_factor(row["op"], row["ranks"])
                    for row in s.to_list()
                ],
                dtype=pl.Float64,
            )
        )
        .alias("factor")
    )
    return (
        sized.group_by("op", "bucket")
        .agg(
            pl.len().alias("calls"),
            (pl.col("duration_ns").sum() / 1e6).alias("duration_ms"),
            pl.col("bytes").sum(),
            pl.col("factor").mean(),
            pl.col("duration_ns").sum().alias("_ns"),
        )
        .with_columns(
            (pl.col("op") + "|" + pl.col("bucket")).alias("key"),
            (pl.col("bytes") / pl.col("_ns")).alias("algbw"),
        )
        .with_columns((pl.col("algbw") * pl.col("factor")).alias("busbw"))
        .select("key", "calls", "duration_ms", "bytes", "algbw", "busbw")
    )


def wait_skew_us(ops: pl.DataFrame) -> pl.DataFrame:
    """Measures how unevenly ranks spend time in the same collective.

    The k-th call of an operation with one message size in one process group
    is taken to be the same collective on every rank. The rank that arrives
    last waits least, so the spread of the call durations is the arrival skew.
    It does not depend on the ranks' clocks.

    Args:
        ops: A ``network_ops`` table.

    Returns:
        One row per ``op`` with ``skew_us``, the mean spread over collectives
        seen on more than one rank; no row for an operation seen on one rank.
    """
    seq = (
        ops.sort("rank", "start_ns")
        .with_columns(
            pl.col("start_ns")
            .cum_count()
            .over("rank", "group", "op", "bytes")
            .alias("seq")
        )
        .group_by("group", "op", "bytes", "seq")
        .agg(
            pl.col("rank").n_unique().alias("n"),
            (pl.col("duration_ns").max() - pl.col("duration_ns").min()).alias(
                "spread"
            ),
        )
        .filter(pl.col("n") > 1)
    )
    return seq.group_by("op").agg(
        (pl.col("spread").mean() / 1e3).alias("skew_us")
    )


def _add_trial_metrics(
    metrics: dict[str, list[float]], ops: pl.DataFrame, per_key: pl.DataFrame
) -> None:
    """Appends one trial's values to the metric lists."""
    per_rank_ns = ops.group_by("rank").agg(pl.col("duration_ns").sum())
    metrics.setdefault("comm_time_ms", []).append(
        float(per_rank_ns["duration_ns"].to_numpy().mean() / 1e6)
    )
    metrics.setdefault("comm_calls", []).append(float(ops.height))
    for row in per_key.iter_rows(named=True):
        key = row["key"]
        metrics.setdefault(f"time_ms|{key}", []).append(row["duration_ms"])
        if row["bytes"] > 0:
            metrics.setdefault(f"algbw_GBps|{key}", []).append(row["algbw"])
            metrics.setdefault(f"busbw_GBps|{key}", []).append(row["busbw"])
    for row in wait_skew_us(ops).iter_rows(named=True):
        metrics.setdefault(f"wait_skew_us|{row['op']}", []).append(
            row["skew_us"]
        )


_UNITS = {
    "comm_time_ms": ("ms", True),
    "comm_calls": ("count", True),
    "time_ms": ("ms", True),
    "algbw_GBps": ("GB/s", False),
    "busbw_GBps": ("GB/s", False),
    "wait_skew_us": ("us", True),
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
