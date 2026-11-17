"""Checks that a GOAL schedule conserves the Chakra-ET trace it came from.

Holds the ``schedule_equivalence`` aspect and
``verify_chakra_goal_equivalence``, which ``cta verify-equivalence`` runs.
"""

from __future__ import annotations

import dataclasses
import pathlib

import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats
from cta.ingest import cache
from cta.ingest import discovery


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
        if algorithm == "direct":
            return float(n_ranks - 1)
        return 2.0 * (n_ranks - 1) / n_ranks
    if comm_type in (
        "ALL_GATHER",
        "REDUCE_SCATTER",
        "REDUCE_SCATTER_BLOCK",
        "ALL_TO_ALL",
        "BROADCAST",
        "REDUCE",
    ):
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
    by_type = collectives.group_by("comm_type").agg(pl.col("size_bytes").sum())
    collective_bytes = sum(
        float(bytes_sum)
        * _collective_send_factor(str(comm_type), n_ranks, algorithm)
        for comm_type, bytes_sum in by_type.iter_rows()
    )
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
    store = cache.CacheStore(pathlib.Path(cache_dir))
    chakra_bundle = discovery.resolve_bundle(chakra_source)
    goal_bundle = discovery.resolve_bundle(goal_source)

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
        plugins.ADAPTERS[chakra_format],
    )
    goal_table_paths = store.get_or_ingest(
        goal_bundle.artifacts["goal"], plugins.ADAPTERS["goal"]
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


@plugins.register_aspect(
    "schedule_equivalence",
    title="Chakra-ET to GOAL Conversion Conservation",
    required_formats=("chakra_et", "goal"),
    any_of_formats=False,
)
class ScheduleEquivalenceAspect(plugins.Aspect):
    """Compares how well each group's GOAL schedule conserves its Chakra-ET.

    Each trial bundle must hold both a Chakra-ET trace and the GOAL schedule
    converted from it.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Chakra-ET and GOAL, both needed in every bundle.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
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
