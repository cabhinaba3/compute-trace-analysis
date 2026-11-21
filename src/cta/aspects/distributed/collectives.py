"""Aspect ``collectives``: collective counts and volumes by type."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats
from cta.analysis import streaming
from cta.aspects.distributed import graph_tables


@plugins.register_aspect(
    "collectives",
    title="Communication & Collective Characterization",
    required_formats=graph_tables.FORMATS,
    any_of_formats=True,
)
class CollectivesAspect(plugins.Aspect):
    """Compares communication volume, op count and message sizes.

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
        """Compares collective and send nodes across the trial groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics plus a ``by_comm_type`` detail table.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        breakdown_totals: list[pl.DataFrame] = []
        prov_rows: list[dict[str, Any]] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_comm_mb": [],
                "comm_op_count": [],
                "median_msg_kb": [],
                "p95_msg_kb": [],
                "max_msg_kb": [],
            }
            trial_breakdowns: list[pl.DataFrame] = []
            for trial_idx, (trial, table_paths) in enumerate(
                zip(group.trials, tables[group.group_id], strict=True)
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
                    prov_rows.append(
                        streaming.provenance_row(
                            group.label, trial_idx, agg.provenance
                        )
                    )
                    metric_values["total_comm_mb"].append(agg.total_comm_mb)
                    metric_values["comm_op_count"].append(
                        float(agg.comm_op_count)
                    )
                    metric_values["median_msg_kb"].append(agg.median_msg_kb)
                    metric_values["p95_msg_kb"].append(agg.p95_msg_kb)
                    metric_values["max_msg_kb"].append(agg.max_msg_kb)
                    if agg.by_comm_type.height:
                        trial_breakdowns.append(agg.by_comm_type)
                    continue

                nodes, _, _ = graph_tables.read_graph_tables(table_paths)
                comm_nodes = nodes.filter(
                    pl.col("op_class").is_in(["comm_coll", "comm_send"])
                )
                sizes_bytes = (
                    comm_nodes["size_bytes"].to_numpy().astype(np.float64)
                    if comm_nodes.height
                    else np.zeros(0)
                )
                has_sizes = bool(len(sizes_bytes))
                metric_values["total_comm_mb"].append(
                    float(sizes_bytes.sum() / 1e6)
                )
                metric_values["comm_op_count"].append(float(comm_nodes.height))
                metric_values["median_msg_kb"].append(
                    float(np.median(sizes_bytes) / 1e3) if has_sizes else 0.0
                )
                metric_values["p95_msg_kb"].append(
                    float(np.percentile(sizes_bytes, 95) / 1e3)
                    if has_sizes
                    else 0.0
                )
                metric_values["max_msg_kb"].append(
                    float(np.max(sizes_bytes) / 1e3) if has_sizes else 0.0
                )

                if comm_nodes.height:
                    trial_breakdowns.append(
                        comm_nodes.group_by("comm_type").agg(
                            (
                                pl.col("size_bytes").sum().cast(pl.Float64)
                                / 1e6
                            ).alias("volume_mb"),
                            pl.len().alias("op_count"),
                        )
                    )

            trials.append(metric_values)
            if trial_breakdowns:
                breakdown_totals.append(
                    pl.concat(trial_breakdowns)
                    .group_by("comm_type")
                    .agg(
                        pl.col("volume_mb").mean(),
                        pl.col("op_count").mean().round(0).cast(pl.Int64),
                    )
                )
            else:
                breakdown_totals.append(
                    pl.DataFrame(
                        {"comm_type": [], "volume_mb": [], "op_count": []}
                    )
                )

        summary = stats.compare_scalar_metrics(
            [
                ("total_comm_mb", "MB", True),
                ("comm_op_count", "ops", True),
                ("median_msg_kb", "KB", False),
                ("p95_msg_kb", "KB", False),
                ("max_msg_kb", "KB", False),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        by_comm_type = stats.compare_entity_tables(
            labels,
            breakdown_totals,
            key_column="comm_type",
            value_column="volume_mb",
            count_column="op_count",
        )
        details: dict[str, pl.DataFrame] = {"by_comm_type": by_comm_type}
        if prov_rows:
            details["provenance"] = pl.DataFrame(prov_rows)
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details=details,
        )
