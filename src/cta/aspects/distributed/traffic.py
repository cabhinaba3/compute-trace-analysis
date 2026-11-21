"""Aspect ``traffic_matrix``: bytes sent between rank pairs."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats
from cta.analysis import streaming
from cta.aspects.distributed import graph_tables


def _link_volumes(traffic: pl.DataFrame) -> pl.DataFrame:
    """Labels each (src, dst) traffic row ``src->dst`` with its volume in MB."""
    return traffic.with_columns(
        pl.format("{}->{}", pl.col("src_rank"), pl.col("dst_rank")).alias(
            "link"
        ),
        (pl.col("total_bytes").cast(pl.Float64) / 1e6).alias("volume_mb"),
    ).select("link", "src_rank", "dst_rank", "volume_mb", "message_count")


@plugins.register_aspect(
    "traffic_matrix",
    title="Rank-to-Rank Traffic Matrix & Topology Locality",
    required_formats=graph_tables.FORMATS,
    any_of_formats=True,
)
class TrafficMatrixAspect(plugins.Aspect):
    """Compares the rank-to-rank traffic matrix and its hotspots.

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
        """Compares point-to-point traffic between ranks across groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics, a ``links`` detail table and one
            ``matrix_<label>`` table per group.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        link_totals: list[pl.DataFrame] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "active_rank_links": [],
                "hotspot_link_mb": [],
                "mean_peer_degree": [],
                "total_p2p_mb": [],
            }
            trial_links: list[pl.DataFrame] = []
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
                    metric_values["active_rank_links"].append(
                        float(agg.active_rank_links)
                    )
                    metric_values["hotspot_link_mb"].append(agg.hotspot_link_mb)
                    metric_values["mean_peer_degree"].append(
                        agg.mean_peer_degree
                    )
                    metric_values["total_p2p_mb"].append(agg.total_p2p_mb)
                    if agg.traffic.height:
                        trial_links.append(_link_volumes(agg.traffic))
                    continue

                nodes, _, traffic = graph_tables.read_graph_tables(table_paths)
                world_size = (
                    int(nodes["rank"].n_unique()) if nodes.height else 1
                )
                n_links = traffic.height
                link_bytes = (
                    traffic["total_bytes"].to_numpy()
                    if n_links
                    else np.zeros(0)
                )
                total_mb = float(np.sum(link_bytes) / 1e6) if n_links else 0.0
                hotspot_mb = float(np.max(link_bytes) / 1e6) if n_links else 0.0
                metric_values["active_rank_links"].append(float(n_links))
                metric_values["hotspot_link_mb"].append(hotspot_mb)
                metric_values["mean_peer_degree"].append(
                    n_links / max(1, world_size)
                )
                metric_values["total_p2p_mb"].append(total_mb)
                if n_links:
                    trial_links.append(_link_volumes(traffic))

            trials.append(metric_values)
            if trial_links:
                link_totals.append(
                    pl.concat(trial_links)
                    .group_by(["link", "src_rank", "dst_rank"])
                    .agg(
                        pl.col("volume_mb").mean(),
                        pl.col("message_count").mean().round(0).cast(pl.Int64),
                    )
                )
            else:
                link_totals.append(
                    pl.DataFrame(
                        {"link": [], "volume_mb": [], "message_count": []}
                    )
                )

        summary = stats.compare_scalar_metrics(
            [
                ("total_p2p_mb", "MB", True),
                ("active_rank_links", "links", True),
                ("mean_peer_degree", "peers/rank", True),
                ("hotspot_link_mb", "MB", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        link_diff = stats.compare_entity_tables(
            labels,
            link_totals,
            key_column="link",
            value_column="volume_mb",
            count_column="message_count",
        )
        details = {
            "links": link_diff,
            **{
                f"matrix_{label}": links
                for label, links in zip(labels, link_totals, strict=True)
            },
        }
        return models.AspectResult(
            name=self.name, title=self.title, summary=summary, details=details
        )
