"""Aspect ``rank_skew``: how evenly work and traffic spread over ranks."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats
from cta.analysis import streaming
from cta.aspects.distributed import graph_tables


def _per_rank_load(nodes: pl.DataFrame) -> pl.DataFrame:
    """Sums compute time, comm volume and op count per rank, sorted by rank."""
    is_comm = pl.col("op_class").is_in(["comm_coll", "comm_send"])
    return (
        nodes.group_by("rank")
        .agg(
            (
                pl.col("duration_ns")
                .filter(pl.col("op_class") == "compute")
                .sum()
                .cast(pl.Float64)
                / 1e3
            ).alias("compute_us"),
            (
                pl.col("size_bytes").filter(is_comm).sum().cast(pl.Float64)
                / 1e6
            ).alias("comm_mb"),
            pl.len().alias("op_count"),
        )
        .sort("rank")
    )


@plugins.register_aspect(
    "rank_skew",
    title="Multi-Rank Load Balance & Straggler Skew",
    required_formats=graph_tables.FORMATS,
    any_of_formats=True,
)
class RankSkewAspect(plugins.Aspect):
    """Compares load balance across ranks and straggler skew.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide an execution graph.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
    ) -> models.AspectResult:
        """Compares per-rank compute and communication load across groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics, the last non-empty trial's per-rank table of
            each group keyed by its label, and a ``share_Rank-Skew`` scalar
            per group.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}
        per_rank_by_label: dict[str, pl.DataFrame] = {}

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "world_size": [],
                "max_rank_compute_us": [],
                "mean_rank_compute_us": [],
                "compute_skew_ratio": [],
                "max_rank_comm_mb": [],
                "comm_jains_fairness": [],
            }
            last_per_rank = pl.DataFrame()
            for trial, table_paths in zip(
                group.trials, tables[group.group_id], strict=True
            ):
                artifact = streaming.pick_graph_artifact(trial)
                has_parquet = any(
                    f"{fmt}.nodes" in table_paths
                    for fmt in graph_tables.FORMATS
                )
                use_stream = artifact is not None and (
                    not has_parquet
                    or streaming.resolve_streaming_tier(artifact, config)
                    != "off"
                )
                if use_stream and artifact is not None:
                    agg = streaming.aggregate_artifact_cached(
                        artifact,
                        quantile_sketch=config.quantile_sketch,
                        goal_semantics=config.goal_semantics,
                    )
                    if agg.num_nodes == 0:
                        continue
                    last_per_rank = agg.per_rank
                    metric_values["world_size"].append(float(agg.world_size))
                    metric_values["max_rank_compute_us"].append(
                        agg.max_rank_compute_us
                    )
                    metric_values["mean_rank_compute_us"].append(
                        agg.mean_rank_compute_us
                    )
                    metric_values["compute_skew_ratio"].append(
                        agg.compute_skew_ratio
                    )
                    metric_values["max_rank_comm_mb"].append(
                        agg.max_rank_comm_mb
                    )
                    metric_values["comm_jains_fairness"].append(
                        agg.comm_jains_fairness
                    )
                    continue

                nodes, _, _ = graph_tables.read_graph_tables(table_paths)
                if nodes.is_empty():
                    continue
                per_rank = _per_rank_load(nodes)
                last_per_rank = per_rank
                compute_us = per_rank["compute_us"].to_numpy()
                comm_mb = per_rank["comm_mb"].to_numpy()
                world_size = len(per_rank)
                max_compute_us = float(compute_us.max()) if world_size else 0.0
                mean_compute_us = (
                    float(compute_us.mean()) if world_size else 0.0
                )
                skew_ratio = (
                    (max_compute_us / mean_compute_us)
                    if mean_compute_us > 1e-12
                    else 1.0
                )
                # Jain's fairness index of the per-rank comm volume.
                jains_denominator = float(world_size * np.sum(comm_mb**2))
                jains_index = (
                    float((np.sum(comm_mb) ** 2) / jains_denominator)
                    if jains_denominator > 1e-12
                    else 1.0
                )

                metric_values["world_size"].append(float(world_size))
                metric_values["max_rank_compute_us"].append(max_compute_us)
                metric_values["mean_rank_compute_us"].append(mean_compute_us)
                metric_values["compute_skew_ratio"].append(skew_ratio)
                metric_values["max_rank_comm_mb"].append(
                    float(comm_mb.max()) if world_size else 0.0
                )
                metric_values["comm_jains_fairness"].append(jains_index)

            trials.append(metric_values)
            per_rank_by_label[group.label] = last_per_rank
            mean_skew = (
                float(np.mean(metric_values["compute_skew_ratio"]))
                if metric_values["compute_skew_ratio"]
                else 1.0
            )
            scalars[group.label] = {
                "share_Rank-Skew": max(
                    0.0, (mean_skew - 1.0) / max(mean_skew, 1.0)
                )
            }

        summary = stats.compare_scalar_metrics(
            [
                ("world_size", "ranks", False),
                ("max_rank_compute_us", "us", True),
                ("mean_rank_compute_us", "us", True),
                ("compute_skew_ratio", "max/mean", True),
                ("max_rank_comm_mb", "MB", True),
                ("comm_jains_fairness", "index", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details=per_rank_by_label,
            scalars=scalars,
        )
