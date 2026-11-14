"""Distributed graph & schedule comparison aspects (Chakra-ET & GOAL)."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import polars as pl

from cta.core import bundle
from cta.core import cache
from cta.core import dag
from cta.core import models
from cta.core import registry
from cta.core import stats

_DISTRIBUTED_FORMATS = ("chakra_et", "chakra_host_et", "goal")


def _pick_distributed_tables(
    table_paths: dict[str, pathlib.Path],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Reads the node, edge and traffic tables of one trial.

    Uses the first format in ``_DISTRIBUTED_FORMATS`` that has a node table.

    Args:
        table_paths: Mapping of table key to Parquet path for one trial.

    Returns:
        The ``(nodes, edges, traffic)`` tables.

    Raises:
        KeyError: If no distributed format has a node table.
    """
    for trace_format in _DISTRIBUTED_FORMATS:
        if f"{trace_format}.nodes" in table_paths:
            return (
                pl.read_parquet(table_paths[f"{trace_format}.nodes"]),
                pl.read_parquet(table_paths[f"{trace_format}.edges"]),
                pl.read_parquet(table_paths[f"{trace_format}.traffic"]),
            )
    raise KeyError("No distributed graph tables found in trial bundle")


@registry.register_aspect(
    "dag_critical_path",
    title="DAG Structure & Critical Path",
    required_formats=_DISTRIBUTED_FORMATS,
    any_of_formats=True,
)
class DagCriticalPathAspect:
    """Compares the execution DAG's size, depth, critical path and slack.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide an execution graph.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "dag_critical_path"
    title = "DAG Structure & Critical Path"
    required_formats = _DISTRIBUTED_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
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
            for table_paths in tables[group.group_id]:
                nodes, edges, _ = _pick_distributed_tables(table_paths)
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
                metric_values["dag_depth"].append(float(dag_stats.dag_depth))
                metric_values["num_nodes"].append(float(dag_stats.num_nodes))
                metric_values["num_edges"].append(float(dag_stats.num_edges))
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
            name=self.name, title=self.title, summary=summary, scalars=scalars
        )


@registry.register_aspect(
    "comm_collectives",
    title="Communication & Collective Characterization",
    required_formats=_DISTRIBUTED_FORMATS,
    any_of_formats=True,
)
class CommCollectivesAspect:
    """Compares communication volume, op count and message sizes.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide an execution graph.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "comm_collectives"
    title = "Communication & Collective Characterization"
    required_formats = _DISTRIBUTED_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
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

        for group in groups:
            metric_values: dict[str, list[float]] = {
                "total_comm_mb": [],
                "comm_op_count": [],
                "median_msg_kb": [],
                "p95_msg_kb": [],
                "max_msg_kb": [],
            }
            trial_breakdowns: list[pl.DataFrame] = []
            for table_paths in tables[group.group_id]:
                nodes, _, _ = _pick_distributed_tables(table_paths)
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
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details={"by_comm_type": by_comm_type},
        )


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


@registry.register_aspect(
    "rank_skew",
    title="Multi-Rank Load Balance & Straggler Skew",
    required_formats=_DISTRIBUTED_FORMATS,
    any_of_formats=True,
)
class RankSkewAspect:
    """Compares load balance across ranks and straggler skew.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide an execution graph.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "rank_skew"
    title = "Multi-Rank Load Balance & Straggler Skew"
    required_formats = _DISTRIBUTED_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
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
            for table_paths in tables[group.group_id]:
                nodes, _, _ = _pick_distributed_tables(table_paths)
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


def _link_volumes(traffic: pl.DataFrame) -> pl.DataFrame:
    """Labels each (src, dst) traffic row ``src->dst`` with its volume in MB."""
    return traffic.with_columns(
        pl.format("{}->{}", pl.col("src_rank"), pl.col("dst_rank")).alias(
            "link"
        ),
        (pl.col("total_bytes").cast(pl.Float64) / 1e6).alias("volume_mb"),
    ).select("link", "src_rank", "dst_rank", "volume_mb", "message_count")


@registry.register_aspect(
    "traffic_matrix",
    title="Rank-to-Rank Traffic Matrix & Topology Locality",
    required_formats=_DISTRIBUTED_FORMATS,
    any_of_formats=True,
)
class TrafficMatrixAspect:
    """Compares the rank-to-rank traffic matrix and its hotspots.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide an execution graph.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "traffic_matrix"
    title = "Rank-to-Rank Traffic Matrix & Topology Locality"
    required_formats = _DISTRIBUTED_FORMATS
    any_of_formats = True

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
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
            for table_paths in tables[group.group_id]:
                nodes, _, traffic = _pick_distributed_tables(table_paths)
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


@dataclasses.dataclass(slots=True, frozen=True)
class EquivalenceVerdict:
    """Result of checking that a GOAL schedule conserves a Chakra-ET trace.

    Attributes:
        equivalent: Whether every check in ``checks`` passed.
        world_size_chakra: Number of ranks in the Chakra-ET trace.
        world_size_goal: Number of ranks in the GOAL schedule.
        compute_rel_error: Relative difference in total compute time.
        comm_volume_rel_error: Relative difference between the GOAL send
            volume and the volume expected from the Chakra-ET trace.
        checks: One row per check with ``check``, ``expected``,
            ``observed``, ``rel_error`` and ``passed`` columns.
    """

    equivalent: bool
    world_size_chakra: int
    world_size_goal: int
    compute_rel_error: float
    comm_volume_rel_error: float
    checks: pl.DataFrame


def _op_class_sum(nodes: pl.DataFrame, op_class: str, column: str) -> float:
    """Returns the sum of ``column`` over the nodes of ``op_class``."""
    return float(nodes.filter(pl.col("op_class") == op_class)[column].sum())


def _collective_send_factor(
    comm_type: str, n_ranks: int, algorithm: str
) -> float:
    """Returns the bytes each rank sends per byte of a collective, on average.

    Args:
        comm_type: Chakra collective type, e.g. ``ALL_REDUCE``.
        n_ranks: Number of ranks P (at least 2).
        algorithm: Lower-cased collective algorithm name.

    Returns:
        The send factor; 0.0 for collective types without a model.
    """
    if comm_type == "ALL_REDUCE":
        if algorithm == "tree":
            # Binomial/Rabenseifner or 2x tree broadcast/reduce approx
            # 2 * ((P-1)/P) or 2*B; standard ring/rabenseifner =
            # 2*(P-1)/P * B.
            return 2.0 * (n_ranks - 1) / n_ranks
        if algorithm == "direct":
            return float(n_ranks - 1)
        return 2.0 * (n_ranks - 1) / n_ranks
    if comm_type in (
        "ALL_GATHER",
        "REDUCE_SCATTER",
        "REDUCE_SCATTER_BLOCK",
        "ALL_TO_ALL",
    ):
        return (n_ranks - 1) / n_ranks
    if comm_type in ("BROADCAST", "REDUCE"):
        # Across all P ranks total bytes sent is (P-1)*B -> average
        # ((P-1)/P)*B per rank.
        return (n_ranks - 1) / n_ranks
    return 0.0


def _expected_goal_send_bytes(
    chakra_nodes: pl.DataFrame, world_size: int, collective_algorithm: str
) -> float:
    """Returns the send bytes a GOAL decomposition of the Chakra-ET nodes has.

    Point-to-point sends count in full; each collective adds its size times
    a per-rank factor for its type and ``collective_algorithm``.

    Args:
        chakra_nodes: Chakra-ET node table.
        world_size: Number of ranks; with one rank collectives send nothing.
        collective_algorithm: ``tree`` or ``direct``; any other value is
            treated as ring.

    Returns:
        The expected GOAL send volume in bytes.
    """
    n_ranks = max(1, world_size)
    p2p_bytes = _op_class_sum(chakra_nodes, "comm_send", "size_bytes")
    collectives = chakra_nodes.filter(pl.col("op_class") == "comm_coll")
    if collectives.is_empty() or n_ranks <= 1:
        return p2p_bytes

    algorithm = collective_algorithm.lower()
    collective_bytes = 0.0
    for row in collectives.select("comm_type", "size_bytes").iter_rows(
        named=True
    ):
        factor = _collective_send_factor(
            str(row["comm_type"]), n_ranks, algorithm
        )
        collective_bytes += float(row["size_bytes"]) * factor
    return p2p_bytes + collective_bytes


def verify_chakra_goal_equivalence(
    chakra_source: str | pathlib.Path,
    goal_source: str | pathlib.Path,
    *,
    collective_algorithm: str = "ring",
    rel_tol: float = 0.02,
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> EquivalenceVerdict:
    """Checks that a GOAL schedule conserves the work of a Chakra-ET trace.

    Compares the world sizes, the total compute time, the GOAL send volume
    against the volume expected from decomposing the Chakra-ET collectives
    with ``collective_algorithm``, and the GOAL send/receive balance.

    Args:
        chakra_source: Chakra-ET trace, or a bundle that contains one.
        goal_source: GOAL schedule (``.goal`` or ``.bin``), or a bundle that
            contains one.
        collective_algorithm: Collective algorithm the GOAL schedule was
            generated with: ``ring``, ``tree`` or ``direct``.
        rel_tol: Largest relative error at which the compute and
            communication checks still pass.
        cache_dir: Directory of the Parquet ingest cache.

    Returns:
        The verdict, with one row per check.

    Raises:
        ValueError: If ``chakra_source`` has no Chakra-ET trace or
            ``goal_source`` has no GOAL schedule.
    """
    registry.load_plugins()
    store = cache.CacheStore(pathlib.Path(cache_dir))
    chakra_bundle = bundle.resolve_bundle(chakra_source)
    goal_bundle = bundle.resolve_bundle(goal_source)

    chakra_format = next(
        (
            trace_format
            for trace_format in ("chakra_et", "chakra_host_et")
            if trace_format in chakra_bundle.artifacts
        ),
        None,
    )
    if chakra_format is None:
        raise ValueError(
            "Expected Chakra-ET trace, found formats: "
            f"{sorted(chakra_bundle.formats)}"
        )
    if "goal" not in goal_bundle.artifacts:
        raise ValueError(
            "Expected GOAL (.goal/.bin) schedule, found formats: "
            f"{sorted(goal_bundle.formats)}"
        )

    chakra_table_paths = store.get_or_ingest(
        chakra_bundle.artifacts[chakra_format],
        registry.ADAPTERS[chakra_format],
    )
    goal_table_paths = store.get_or_ingest(
        goal_bundle.artifacts["goal"], registry.ADAPTERS["goal"]
    )
    chakra_nodes = pl.read_parquet(chakra_table_paths["nodes"])
    goal_nodes = pl.read_parquet(goal_table_paths["nodes"])

    world_size_chakra = (
        int(chakra_nodes["rank"].n_unique()) if chakra_nodes.height else 0
    )
    world_size_goal = (
        int(goal_nodes["rank"].n_unique()) if goal_nodes.height else 0
    )

    chakra_compute_ns = _op_class_sum(chakra_nodes, "compute", "duration_ns")
    goal_compute_ns = _op_class_sum(goal_nodes, "compute", "duration_ns")
    compute_rel_error = (
        abs(goal_compute_ns - chakra_compute_ns) / max(chakra_compute_ns, 1.0)
        if (chakra_compute_ns > 0 or goal_compute_ns > 0)
        else 0.0
    )

    expected_comm_bytes = _expected_goal_send_bytes(
        chakra_nodes,
        max(world_size_chakra, world_size_goal),
        collective_algorithm,
    )
    observed_comm_bytes = _op_class_sum(goal_nodes, "comm_send", "size_bytes")
    comm_rel_error = (
        abs(observed_comm_bytes - expected_comm_bytes)
        / max(expected_comm_bytes, 1.0)
        if (expected_comm_bytes > 0 or observed_comm_bytes > 0)
        else 0.0
    )

    goal_sends = int(
        goal_nodes.filter(pl.col("op_class") == "comm_send").height
    )
    goal_recvs = int(
        goal_nodes.filter(pl.col("op_class") == "comm_recv").height
    )
    send_recv_rel_error = (
        0.0
        if goal_sends == goal_recvs
        else abs(goal_recvs - goal_sends) / max(1, goal_sends)
    )

    checks = pl.DataFrame(
        [
            {
                "check": "world_size_match",
                "expected": float(world_size_chakra),
                "observed": float(world_size_goal),
                "rel_error": 0.0
                if world_size_chakra == world_size_goal
                else 1.0,
                "passed": world_size_chakra == world_size_goal,
            },
            {
                "check": "compute_duration_ns",
                "expected": chakra_compute_ns,
                "observed": goal_compute_ns,
                "rel_error": compute_rel_error,
                "passed": compute_rel_error <= rel_tol,
            },
            {
                "check": f"decomposed_comm_bytes ({collective_algorithm})",
                "expected": expected_comm_bytes,
                "observed": observed_comm_bytes,
                "rel_error": comm_rel_error,
                "passed": comm_rel_error <= rel_tol,
            },
            {
                "check": "goal_send_recv_balance",
                "expected": float(goal_sends),
                "observed": float(goal_recvs),
                "rel_error": send_recv_rel_error,
                "passed": goal_sends == goal_recvs,
            },
        ]
    )
    return EquivalenceVerdict(
        equivalent=bool(checks["passed"].all()),
        world_size_chakra=world_size_chakra,
        world_size_goal=world_size_goal,
        compute_rel_error=compute_rel_error,
        comm_volume_rel_error=comm_rel_error,
        checks=checks,
    )


@registry.register_aspect(
    "schedule_equivalence",
    title="Chakra-ET to GOAL Conversion Conservation",
    required_formats=("chakra_et", "goal"),
    any_of_formats=False,
)
class ScheduleEquivalenceAspect:
    """Compares how well each group's GOAL schedule conserves its Chakra-ET.

    Each trial bundle must hold both a Chakra-ET trace and the GOAL schedule
    converted from it.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Chakra-ET and GOAL, both needed in every bundle.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    name = "schedule_equivalence"
    title = "Chakra-ET to GOAL Conversion Conservation"
    required_formats = ("chakra_et", "goal")
    any_of_formats = False

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.CompareConfig,
    ) -> models.AspectResult:
        """Compares compute and comm-volume mismatch across the groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings: the collective algorithm for the
                expected send volume and the significance level.

        Returns:
            Summary metrics only.
        """
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        for group in groups:
            compute_mismatch_pct: list[float] = []
            comm_mismatch_pct: list[float] = []
            for table_paths in tables[group.group_id]:
                chakra_nodes = pl.read_parquet(table_paths["chakra_et.nodes"])
                goal_nodes = pl.read_parquet(table_paths["goal.nodes"])
                world_size = max(
                    int(chakra_nodes["rank"].n_unique()),
                    int(goal_nodes["rank"].n_unique()),
                )
                chakra_compute_ns = _op_class_sum(
                    chakra_nodes, "compute", "duration_ns"
                )
                goal_compute_ns = _op_class_sum(
                    goal_nodes, "compute", "duration_ns"
                )
                compute_mismatch_pct.append(
                    abs(goal_compute_ns - chakra_compute_ns)
                    / max(chakra_compute_ns, 1.0)
                    * 100.0
                )
                expected_bytes = _expected_goal_send_bytes(
                    chakra_nodes, world_size, config.collective_algorithm
                )
                observed_bytes = _op_class_sum(
                    goal_nodes, "comm_send", "size_bytes"
                )
                comm_mismatch_pct.append(
                    abs(observed_bytes - expected_bytes)
                    / max(expected_bytes, 1.0)
                    * 100.0
                )
            trials.append(
                {
                    "compute_mismatch_pct": compute_mismatch_pct,
                    "comm_volume_mismatch_pct": comm_mismatch_pct,
                }
            )

        summary = stats.compare_scalar_metrics(
            [
                ("compute_mismatch_pct", "%", True),
                ("comm_volume_mismatch_pct", "%", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name, title=self.title, summary=summary
        )
