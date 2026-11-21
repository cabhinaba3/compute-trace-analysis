"""Aspect ``critical_path``: graph size, depth, critical path and slack."""

from __future__ import annotations

import pathlib
from typing import Any

import polars as pl

from cta import models
from cta import plugins
from cta.analysis import dag
from cta.analysis import stats
from cta.analysis import streaming
from cta.aspects.distributed import graph_tables


@plugins.register_aspect(
    "critical_path",
    title="DAG Structure & Critical Path",
    required_formats=graph_tables.FORMATS,
    any_of_formats=True,
)
class CriticalPathAspect(plugins.Aspect):
    """Compares the execution DAG's size, depth, critical path and slack.

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
        """Compares the critical path of each trial's DAG across the groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; only the significance level is
                used.

        Returns:
            Summary metrics plus the critical path's compute and
            exposed-communication shares as scalars.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        scalars: dict[str, dict[str, float]] = {}
        details: dict[str, pl.DataFrame] = {}
        prov_rows: list[dict[str, Any]] = []

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "critical_path_us": [],
                "cp_compute_us": [],
                "cp_exposed_comm_us": [],
                "cp_comm_mb": [],
                "dag_depth": [],
                "num_nodes": [],
                "num_edges": [],
                "mean_slack_us": [],
                "zero_slack_pct": [],
            }
            for trial_idx, (trial, table_paths) in enumerate(
                zip(group.trials, tables[group.group_id], strict=True)
            ):
                artifact = streaming.pick_graph_artifact(trial)
                tier = (
                    streaming.resolve_streaming_tier(artifact, config)
                    if artifact is not None
                    else "off"
                )
                has_parquet = any(
                    f"{fmt}.nodes" in table_paths
                    for fmt in graph_tables.FORMATS
                )
                use_stream = artifact is not None and (
                    tier != "off" or not has_parquet
                )
                if use_stream and artifact is not None:
                    gc_path = table_paths.get(
                        f"{artifact.format}.graph_cache"
                    ) or table_paths.get("graph_cache")
                    res = streaming.solve_artifact_with_config(
                        artifact, config, cache_path=gc_path
                    )
                    prov_rows.append(
                        streaming.provenance_row(
                            group.label, trial_idx, res.provenance
                        )
                    )
                    if (
                        config.extract_critical_path_ops
                        and not res.critical_path_ops.is_empty()
                    ):
                        key = (
                            "critical_path_ops"
                            if len(groups) == 1
                            else f"critical_path_ops_{group.label}"
                        )
                        details[key] = res.critical_path_ops
                    if (
                        config.delay_attribution
                        and not res.root_cause_delays.is_empty()
                    ):
                        key = (
                            "root_cause_delays"
                            if len(groups) == 1
                            else f"root_cause_delays_{group.label}"
                        )
                        details[key] = res.root_cause_delays
                    metric_values["critical_path_us"].append(
                        res.critical_path_ns / 1e3
                    )
                    metric_values["cp_compute_us"].append(
                        res.cp_compute_ns / 1e3
                    )
                    metric_values["cp_exposed_comm_us"].append(
                        res.cp_exposed_comm_ns / 1e3
                    )
                    metric_values["cp_comm_mb"].append(res.cp_comm_bytes / 1e6)
                    metric_values["dag_depth"].append(float(res.dag_depth))
                    metric_values["num_nodes"].append(float(res.num_nodes))
                    metric_values["num_edges"].append(float(res.num_edges))
                    metric_values["mean_slack_us"].append(
                        res.mean_slack_ns / 1e3
                    )
                    metric_values["zero_slack_pct"].append(
                        res.zero_slack_ratio * 100.0
                    )
                else:
                    nodes, edges, _ = graph_tables.read_graph_tables(
                        table_paths
                    )
                    dag_stats = dag.solve_distributed_dag(nodes, edges)
                    metric_values["critical_path_us"].append(
                        dag_stats.critical_path_ns / 1e3
                    )
                    metric_values["cp_compute_us"].append(
                        dag_stats.cp_compute_ns / 1e3
                    )
                    metric_values["cp_exposed_comm_us"].append(
                        dag_stats.cp_exposed_comm_ns / 1e3
                    )
                    metric_values["cp_comm_mb"].append(
                        dag_stats.cp_comm_bytes / 1e6
                    )
                    metric_values["dag_depth"].append(
                        float(dag_stats.dag_depth)
                    )
                    metric_values["num_nodes"].append(
                        float(dag_stats.num_nodes)
                    )
                    metric_values["num_edges"].append(
                        float(dag_stats.num_edges)
                    )
                    metric_values["mean_slack_us"].append(
                        dag_stats.mean_slack_ns / 1e3
                    )
                    metric_values["zero_slack_pct"].append(
                        dag_stats.zero_slack_ratio * 100.0
                    )
            trials.append(metric_values)
            compute_us = sum(metric_values["cp_compute_us"])
            exposed_comm_us = sum(metric_values["cp_exposed_comm_us"])
            cp_total_us = (compute_us + exposed_comm_us) or 1.0
            scalars[group.label] = {
                "share_Compute": compute_us / cp_total_us,
                "share_Exposed-Comm": exposed_comm_us / cp_total_us,
            }

        if prov_rows:
            details["provenance"] = pl.DataFrame(prov_rows)

        summary = stats.compare_scalar_metrics(
            [
                ("critical_path_us", "us", True),
                ("cp_compute_us", "us", True),
                ("cp_exposed_comm_us", "us", True),
                ("cp_comm_mb", "MB", True),
                ("dag_depth", "hops", True),
                ("num_nodes", "count", True),
                ("num_edges", "count", True),
                ("mean_slack_us", "us", False),
                ("zero_slack_pct", "%", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details=details,
            scalars=scalars,
        )
