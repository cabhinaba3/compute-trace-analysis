"""Streaming graph analysis orchestration and honest provenance models.

Wraps the native C++ streaming sources, quantile sketches, and execution
engines behind typed Python dataclasses and helpers shared across CTA's
distributed aspects and CLI.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pathlib
from typing import Any

import numpy as np
import polars as pl

from cta import _native as native
from cta import models
from cta.adapters import common
from cta.adapters.chakra import reader as chakra_reader
from cta.adapters.goal import reader as goal_reader


@dataclasses.dataclass(slots=True, frozen=True)
class StreamProvenance:
    """Honest fidelity and diagnostic provenance of a streaming analysis run."""

    engine: str = "in_memory"
    exact: bool = True
    slack_scope: str = "global_exact"
    goal_semantics: str = "cta_legacy"
    collective_sync: str = "none"
    makespan_lb: float = 0.0
    makespan_ub: float = 0.0
    evicted_dep_hits: int = 0
    lookahead_drops: int = 0
    straddling_edges: int = 0
    dead_end_sinks: int = 0
    unmatched_p2p_sends: int = 0
    unmatched_p2p_recvs: int = 0
    has_cycle: bool = False
    unordered_nodes: int = 0
    cycle_witness: tuple[tuple[int, int], ...] = ()
    quantile_sketch: str = "exact"
    quantile_error_bound: float = 0.0
    delay_timing_basis: str = "none"

    @classmethod
    def from_native(cls, raw: dict[str, Any]) -> StreamProvenance:
        """Builds a ``StreamProvenance`` from a native provenance dict."""
        ranks = raw.get("cycle_witness_rank", ())
        ids = raw.get("cycle_witness_node_id", ())
        witness = tuple(
            (int(r), int(nid)) for r, nid in zip(ranks, ids, strict=True)
        )
        return cls(
            engine=str(raw["engine"]),
            exact=bool(raw["exact"]),
            slack_scope=str(raw["slack_scope"]),
            goal_semantics=str(raw["goal_semantics"]),
            collective_sync=str(raw["collective_sync"]),
            makespan_lb=float(raw["makespan_lb"]),
            makespan_ub=float(raw["makespan_ub"]),
            evicted_dep_hits=int(raw["evicted_dep_hits"]),
            lookahead_drops=int(raw["lookahead_drops"]),
            straddling_edges=int(raw["straddling_edges"]),
            dead_end_sinks=int(raw["dead_end_sinks"]),
            unmatched_p2p_sends=int(raw["unmatched_p2p_sends"]),
            unmatched_p2p_recvs=int(raw["unmatched_p2p_recvs"]),
            has_cycle=bool(raw["has_cycle"]),
            unordered_nodes=int(raw["unordered_nodes"]),
            cycle_witness=witness,
            quantile_sketch=str(raw["quantile_sketch"]),
            quantile_error_bound=float(raw["quantile_error_bound"]),
            delay_timing_basis=str(raw["delay_timing_basis"]),
        )


@dataclasses.dataclass(slots=True, frozen=True)
class StreamingAggregateSummary:
    """Outputs of Engine A (1-pass O(P^2) aggregate analysis)."""

    num_nodes: int
    num_raw_edges: int
    world_size: int
    per_rank: pl.DataFrame
    max_rank_compute_us: float
    mean_rank_compute_us: float
    compute_skew_ratio: float
    max_rank_comm_mb: float
    comm_jains_fairness: float
    total_comm_mb: float
    comm_op_count: int
    median_msg_kb: float
    p95_msg_kb: float
    max_msg_kb: float
    by_comm_type: pl.DataFrame
    traffic: pl.DataFrame
    active_rank_links: int
    hotspot_link_mb: float
    mean_peer_degree: float
    total_p2p_mb: float
    total_compute_ns: int
    total_send_bytes: int
    send_count: int
    recv_count: int
    coll_bytes_by_type: dict[str, int]
    provenance: StreamProvenance


def open_stream_for_artifact(
    artifact: models.TraceArtifact,
    *,
    goal_semantics: str = "cta_legacy",
    chunk_nodes: int = 524288,
) -> tuple[native.GraphStream, bool]:
    """Opens a native ``GraphStream`` for a GOAL or Chakra artifact.

    Args:
        artifact: Trace artifact of format ``goal``, ``chakra``, or
            ``chakra_host``.
        goal_semantics: ``"cta_legacy"`` or ``"strict"`` for GOAL text.
        chunk_nodes: Target node batch size per chunk.

    Returns:
        ``(stream, include_collectives_in_traffic)``.

    Raises:
        ValueError: If the format is unsupported or a GOAL artifact has
            multiple files.
    """
    paths = sorted(artifact.paths)
    if artifact.format == "goal":
        if len(paths) != 1:
            raise ValueError(
                "Streaming GOAL expects a single .goal or .bin file, "
                f"got {len(paths)}"
            )
        path = paths[0]
        if goal_reader._is_goal_bin(path):
            return (
                native.open_goal_bin_stream(str(path), chunk_nodes=chunk_nodes),
                False,
            )
        return (
            native.open_goal_text_stream(
                str(path),
                goal_semantics=goal_semantics,
                chunk_nodes=chunk_nodes,
            ),
            False,
        )
    if artifact.format == "chakra":
        ranks = [
            common.infer_rank(path, index) for index, path in enumerate(paths)
        ]
        return (
            native.open_chakra_et_stream(
                [str(path) for path in paths], ranks, chunk_nodes=chunk_nodes
            ),
            True,
        )
    if artifact.format == "chakra_host":
        ranks = [
            common.infer_rank(path, index) for index, path in enumerate(paths)
        ]
        return (
            native.open_chakra_host_stream(
                [str(path) for path in paths], ranks, chunk_nodes=chunk_nodes
            ),
            True,
        )
    raise ValueError(
        f"Format '{artifact.format}' does not support streaming graph analysis"
    )


def open_stream_for_path(
    path: str | pathlib.Path,
    *,
    fmt: str,
    rank: int = 0,
    goal_semantics: str = "cta_legacy",
    chunk_nodes: int = 524288,
) -> tuple[native.GraphStream, bool]:
    """Opens a native ``GraphStream`` directly for a file, FIFO, or ``-``."""
    target = str(path)
    if fmt == "goal_bin":
        return (
            native.open_goal_bin_stream(target, chunk_nodes=chunk_nodes),
            False,
        )
    if fmt == "goal":
        is_bin = target != "-" and goal_reader._is_goal_bin(
            pathlib.Path(target)
        )
        if is_bin:
            return (
                native.open_goal_bin_stream(target, chunk_nodes=chunk_nodes),
                False,
            )
        return (
            native.open_goal_text_stream(
                target,
                goal_semantics=goal_semantics,
                chunk_nodes=chunk_nodes,
            ),
            False,
        )
    if fmt == "chakra":
        return (
            native.open_chakra_et_stream(
                [target], [rank], chunk_nodes=chunk_nodes
            ),
            True,
        )
    if fmt == "chakra_host":
        return (
            native.open_chakra_host_stream(
                [target], [rank], chunk_nodes=chunk_nodes
            ),
            True,
        )
    raise ValueError(
        f"Unknown streaming format '{fmt}'; use goal, goal_bin, chakra, "
        "or chakra_host"
    )


def run_streaming_aggregates(
    stream: native.GraphStream,
    *,
    include_collectives_in_traffic: bool = True,
    min_world_size: int = 1,
    quantile_sketch: str = "ddsketch",
    quantile_exact_cap: int = 65536,
    quantile_accuracy: float = 0.001,
    chunk_nodes: int = 524288,
) -> StreamingAggregateSummary:
    """Runs Engine A (1-pass O(P^2) aggregate analysis) over ``stream``."""
    raw = native.stream_aggregates(
        stream,
        include_collectives_in_traffic=include_collectives_in_traffic,
        min_world_size=min_world_size,
        quantile_sketch=quantile_sketch,
        quantile_exact_cap=quantile_exact_cap,
        quantile_accuracy=quantile_accuracy,
        chunk_nodes=chunk_nodes,
    )
    per_rank = pl.DataFrame(
        {
            "rank": np.asarray(raw["rank_ids"], dtype=np.uint32),
            "compute_us": (
                np.asarray(raw["rank_compute_ns"], dtype=np.float64) / 1e3
            ),
            "comm_mb": (
                np.asarray(raw["rank_comm_bytes"], dtype=np.float64) / 1e6
            ),
            "op_count": np.asarray(raw["rank_op_count"], dtype=np.uint64),
        }
    )
    by_comm_type = pl.DataFrame(
        {
            "comm_type": list(raw["comm_type_names"]),
            "volume_mb": (
                np.asarray(raw["comm_type_bytes"], dtype=np.float64) / 1e6
            ),
            "op_count": np.asarray(raw["comm_type_count"], dtype=np.uint64),
        }
    )
    traffic = pl.DataFrame(
        {
            "src_rank": np.asarray(raw["traffic_src"], dtype=np.uint32),
            "dst_rank": np.asarray(raw["traffic_dst"], dtype=np.uint32),
            "message_count": np.asarray(raw["traffic_msgs"], dtype=np.uint64),
            "total_bytes": np.asarray(raw["traffic_bytes"], dtype=np.uint64),
        },
        schema=chakra_reader.TRAFFIC_SCHEMA,
    )
    coll_bytes_by_type = {
        str(name): int(nbytes)
        for name, nbytes in zip(
            raw["coll_type_names"], raw["coll_type_bytes"], strict=True
        )
    }
    return StreamingAggregateSummary(
        num_nodes=int(raw["num_nodes"]),
        num_raw_edges=int(raw["num_raw_edges"]),
        world_size=int(raw["world_size"]),
        per_rank=per_rank,
        max_rank_compute_us=float(raw["max_rank_compute_us"]),
        mean_rank_compute_us=float(raw["mean_rank_compute_us"]),
        compute_skew_ratio=float(raw["compute_skew_ratio"]),
        max_rank_comm_mb=float(raw["max_rank_comm_mb"]),
        comm_jains_fairness=float(raw["comm_jains_fairness"]),
        total_comm_mb=float(raw["total_comm_mb"]),
        comm_op_count=int(raw["comm_op_count"]),
        median_msg_kb=float(raw["median_msg_kb"]),
        p95_msg_kb=float(raw["p95_msg_kb"]),
        max_msg_kb=float(raw["max_msg_kb"]),
        by_comm_type=by_comm_type,
        traffic=traffic,
        active_rank_links=int(raw["active_rank_links"]),
        hotspot_link_mb=float(raw["hotspot_link_mb"]),
        mean_peer_degree=float(raw["mean_peer_degree"]),
        total_p2p_mb=float(raw["total_p2p_mb"]),
        total_compute_ns=int(raw["total_compute_ns"]),
        total_send_bytes=int(raw["total_send_bytes"]),
        send_count=int(raw["send_count"]),
        recv_count=int(raw["recv_count"]),
        coll_bytes_by_type=coll_bytes_by_type,
        provenance=StreamProvenance.from_native(raw["provenance"]),
    )


@dataclasses.dataclass(slots=True, frozen=True)
class StreamingDagResult:
    """Outputs of a streaming/out-of-core DAG solve plus honest provenance."""

    num_nodes: int
    num_edges: int
    unordered: int
    dag_depth: int
    critical_path_ns: float
    cp_compute_ns: float
    cp_exposed_comm_ns: float
    cp_comm_bytes: int
    mean_slack_ns: float
    zero_slack_ratio: float
    critical_path_ops: pl.DataFrame
    root_cause_delays: pl.DataFrame
    provenance: StreamProvenance


def dag_result_from_native(raw: dict[str, Any]) -> StreamingDagResult:
    """Converts a native ``DagStreamResult`` dict to ``StreamingDagResult``."""
    op_codes = np.asarray(raw["cp_op_class"], dtype=np.uint8)
    if len(op_codes) > 0:
        cp_ops = pl.DataFrame(
            {
                "rank": np.asarray(raw["cp_op_rank"], dtype=np.uint32),
                "node_id": np.asarray(raw["cp_op_node_id"], dtype=np.uint64),
                "name": list(raw["cp_op_name"]),
                "op_class": pl.Series(op_codes).replace_strict(
                    common.STREAM_OP_CLASSES, return_dtype=pl.String
                ),
                "weight_ns": np.asarray(
                    raw["cp_op_weight_ns"], dtype=np.float64
                ),
                "size_bytes": np.asarray(
                    raw["cp_op_size_bytes"], dtype=np.uint64
                ),
                "earliest_start_ns": np.asarray(
                    raw["cp_op_earliest_start_ns"], dtype=np.float64
                ),
                "earliest_finish_ns": np.asarray(
                    raw["cp_op_earliest_finish_ns"], dtype=np.float64
                ),
            }
        )
    else:
        cp_ops = pl.DataFrame(
            schema={
                "rank": pl.UInt32,
                "node_id": pl.UInt64,
                "name": pl.String,
                "op_class": pl.String,
                "weight_ns": pl.Float64,
                "size_bytes": pl.UInt64,
                "earliest_start_ns": pl.Float64,
                "earliest_finish_ns": pl.Float64,
            }
        )

    delay_ranks = np.asarray(raw["delay_rank"], dtype=np.uint32)
    if len(delay_ranks) > 0:
        delays = pl.DataFrame(
            {
                "rank": delay_ranks,
                "op_name": list(raw["delay_op_name"]),
                "direct_wait_ns": np.asarray(
                    raw["delay_direct_wait_ns"], dtype=np.float64
                ),
                "propagated_delay_ns": np.asarray(
                    raw["delay_propagated_ns"], dtype=np.float64
                ),
                "op_count": np.asarray(raw["delay_op_count"], dtype=np.uint64),
            }
        )
    else:
        delays = pl.DataFrame(
            schema={
                "rank": pl.UInt32,
                "op_name": pl.String,
                "direct_wait_ns": pl.Float64,
                "propagated_delay_ns": pl.Float64,
                "op_count": pl.UInt64,
            }
        )

    return StreamingDagResult(
        num_nodes=int(raw["num_nodes"]),
        num_edges=int(raw["num_edges"]),
        unordered=int(raw["unordered"]),
        dag_depth=int(raw["depth"]),
        critical_path_ns=float(raw["makespan"]),
        cp_compute_ns=float(raw["critical_compute"]),
        cp_exposed_comm_ns=float(raw["critical_comm"]),
        cp_comm_bytes=int(raw["critical_comm_bytes"]),
        mean_slack_ns=float(raw["mean_slack"]),
        zero_slack_ratio=float(raw["zero_slack_ratio"]),
        critical_path_ops=cp_ops,
        root_cause_delays=delays,
        provenance=StreamProvenance.from_native(raw["provenance"]),
    )


def solve_artifact_frontier(
    artifact: models.TraceArtifact,
    *,
    goal_semantics: str = "cta_legacy",
    collective_sync: str = "none",
    coll_group_sizes: dict[str, int] | None = None,
    window_size: int = 262144,
    max_lookahead: int = 65536,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    scratch_dir: str = "",
) -> StreamingDagResult:
    """Runs Engine B (1-pass frontier / windowed solver) on an artifact."""
    paths = sorted(artifact.paths)
    groups = dict(coll_group_sizes) if coll_group_sizes else {}
    if (
        artifact.format == "goal"
        and len(paths) == 1
        and goal_reader._is_goal_bin(paths[0])
    ):
        raw = native.solve_goal_bin_frontier(
            str(paths[0]),
            goal_semantics=goal_semantics,
            fail_on_cycle=fail_on_cycle,
            extract_critical_path_ops=extract_critical_path_ops,
            scratch_dir=scratch_dir,
        )
        return dag_result_from_native(raw)
    if artifact.format == "chakra":
        ranks = [
            common.infer_rank(path, index) for index, path in enumerate(paths)
        ]
        raw = native.solve_chakra_et_frontier(
            [str(path) for path in paths],
            ranks,
            collective_sync=collective_sync,
            coll_group_sizes=groups,
            window_size=window_size,
            max_lookahead=max_lookahead,
            fail_on_cycle=fail_on_cycle,
            extract_critical_path_ops=extract_critical_path_ops,
            scratch_dir=scratch_dir,
        )
        return dag_result_from_native(raw)

    stream, _ = open_stream_for_artifact(
        artifact, goal_semantics=goal_semantics
    )
    raw = native.solve_dag_stream(
        stream,
        engine="frontier",
        goal_semantics=goal_semantics,
        collective_sync=collective_sync,
        world_size_hint=artifact.world_size,
        coll_group_sizes=groups,
        window_size=window_size,
        max_lookahead=max_lookahead,
        fail_on_cycle=fail_on_cycle,
        extract_critical_path_ops=extract_critical_path_ops,
        scratch_dir=scratch_dir,
    )
    return dag_result_from_native(raw)


def ensure_graph_cache(
    artifact: models.TraceArtifact,
    cache_path: pathlib.Path,
    *,
    goal_semantics: str = "cta_legacy",
    collective_sync: str = "none",
    memory_budget_bytes: int = 0,
    coll_group_sizes: dict[str, int] | None = None,
    fail_on_cycle: bool = False,
    scratch_dir: str = "",
) -> pathlib.Path:
    """Returns a readable `.cta_graph` at ``cache_path``, building it if needed.

    The file is built beside its final name and renamed into place, so an
    interrupted build never leaves a half-written file that looks valid. A file
    that is already there but cannot be opened (cut short by an earlier crash,
    or written by another format version) is deleted and built again.

    Args:
        artifact: The execution graph (Chakra, Chakra host or GOAL).
        cache_path: Final path of the cache.
        goal_semantics: `"cta_legacy"` or `"strict"`.
        collective_sync: `"none"` or `"barrier"`.
        memory_budget_bytes: RAM cap for the build (0 = auto).
        coll_group_sizes: Collective group sizes for the build.
        fail_on_cycle: Whether a cycle in the graph is an error. A cache built
            earlier without it is checked for held-back vertices too.
        scratch_dir: Optional scratch directory for out-of-core passes.

    Returns:
        ``cache_path``.

    Raises:
        RuntimeError: If the graph has a cycle and ``fail_on_cycle`` is set.
    """
    if cache_path.exists():
        try:
            unordered = int(native.GraphCache(str(cache_path)).unordered)
        except (RuntimeError, ValueError, MemoryError, OSError):
            cache_path.unlink(missing_ok=True)
        else:
            if fail_on_cycle and unordered > 0:
                raise RuntimeError(
                    "cycle detected in execution graph "
                    f"({unordered} unordered vertices)"
                )
            return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    try:
        stream, _ = open_stream_for_artifact(
            artifact, goal_semantics=goal_semantics
        )
        native.build_graph_cache(
            stream,
            str(tmp_path),
            memory_budget_bytes=memory_budget_bytes,
            goal_semantics=goal_semantics,
            collective_sync=collective_sync,
            world_size_hint=artifact.world_size,
            coll_group_sizes=dict(coll_group_sizes) if coll_group_sizes else {},
            fail_on_cycle=fail_on_cycle,
            scratch_dir=scratch_dir,
        )
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return cache_path


def solve_artifact_outofcore(
    artifact: models.TraceArtifact,
    *,
    engine: str = "gridgraph",
    memory_budget_bytes: int = 0,
    goal_semantics: str = "cta_legacy",
    collective_sync: str = "none",
    coll_group_sizes: dict[str, int] | None = None,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    delay_attribution: bool = False,
    scratch_dir: str = "",
    cache_path: pathlib.Path | None = None,
) -> StreamingDagResult:
    """Runs an out-of-core GraphEngine (default GridGraph) on an artifact."""
    groups = dict(coll_group_sizes) if coll_group_sizes else {}
    if cache_path is not None and engine == "gridgraph":
        ensure_graph_cache(
            artifact,
            cache_path,
            goal_semantics=goal_semantics,
            collective_sync=collective_sync,
            memory_budget_bytes=memory_budget_bytes,
            coll_group_sizes=groups,
            fail_on_cycle=fail_on_cycle,
            scratch_dir=scratch_dir,
        )
        raw = native.solve_graph_cache(
            str(cache_path),
            memory_budget_bytes=memory_budget_bytes,
            goal_semantics=goal_semantics,
            extract_critical_path_ops=extract_critical_path_ops,
            delay_attribution=delay_attribution,
            scratch_dir=scratch_dir,
        )
        return dag_result_from_native(raw)

    stream, _ = open_stream_for_artifact(
        artifact, goal_semantics=goal_semantics
    )
    raw = native.solve_dag_stream(
        stream,
        engine=engine,
        memory_budget_bytes=memory_budget_bytes,
        goal_semantics=goal_semantics,
        collective_sync=collective_sync,
        world_size_hint=artifact.world_size,
        coll_group_sizes=groups,
        fail_on_cycle=fail_on_cycle,
        extract_critical_path_ops=extract_critical_path_ops,
        delay_attribution=delay_attribution,
        scratch_dir=scratch_dir,
    )
    return dag_result_from_native(raw)


STREAMING_MODES = ("auto", "exact", "frontier", "windowed", "aggregate", "off")
STREAMING_FORMATS = ("chakra", "chakra_host", "goal")


def artifact_raw_bytes(artifact: models.TraceArtifact) -> int:
    """Returns total raw byte size on disk across an artifact's source files."""
    total = 0
    for path in artifact.paths:
        if path.exists() and path.is_file():
            total += path.stat().st_size
    return total


def pick_graph_artifact(
    trial: models.RunBundle,
) -> models.TraceArtifact | None:
    """Returns the preferred execution-graph artifact of a trial, or None."""
    for trace_format in STREAMING_FORMATS:
        if trace_format in trial.artifacts:
            return trial.artifacts[trace_format]
    return None


def _needs_streaming_feature(config: models.AnalysisConfig) -> bool:
    """Returns True when config asks for a streaming-only capability."""
    return (
        config.goal_semantics != "cta_legacy"
        or config.collective_sync != "none"
        or config.delay_attribution
        or config.extract_critical_path_ops
        or config.fail_on_cycle
        or config.graph_engine != "gridgraph"
    )


def resolve_streaming_tier(
    artifact: models.TraceArtifact, config: models.AnalysisConfig
) -> str:
    """Resolves the execution tier for an artifact and AnalysisConfig.

    Returns one of ``"off"``, ``"exact"``, ``"frontier"``, ``"windowed"``, or
    ``"aggregate"``. Raises ``ValueError`` on an unrecognized mode.
    """
    mode = config.streaming.lower().strip()
    if mode not in STREAMING_MODES:
        raise ValueError(
            f"Unknown streaming mode {config.streaming!r}; "
            f"expected one of {STREAMING_MODES}"
        )
    if artifact.format not in STREAMING_FORMATS or mode == "off":
        return "off"
    if mode != "auto":
        return mode
    raw_bytes = artifact_raw_bytes(artifact)
    if raw_bytes >= config.parquet_bypass_bytes:
        return "exact"
    if config.memory_budget_bytes > 0:
        # Rough conservative working-set multiplier for full in-memory Polars +
        # DagPlan materialization (~10x raw binary / ~4x raw text).
        est_ram = raw_bytes * 6
        if est_ram > config.memory_budget_bytes:
            return "exact"
    if _needs_streaming_feature(config):
        return "exact"
    return "off"


def should_bypass_parquet(
    artifact: models.TraceArtifact, config: models.AnalysisConfig
) -> bool:
    """Returns True when full nodes/edges Parquet ingest should be bypassed."""
    if artifact.format not in STREAMING_FORMATS:
        return False
    mode = config.streaming.lower().strip()
    if mode == "off":
        return False
    if mode in ("exact", "frontier", "windowed", "aggregate"):
        return True
    if mode != "auto":
        raise ValueError(
            f"Unknown streaming mode {config.streaming!r}; "
            f"expected one of {STREAMING_MODES}"
        )
    raw_bytes = artifact_raw_bytes(artifact)
    if raw_bytes >= config.parquet_bypass_bytes:
        return True
    return (
        config.memory_budget_bytes > 0
        and raw_bytes * 6 > config.memory_budget_bytes
    )


def aggregate_artifact(
    artifact: models.TraceArtifact,
    *,
    quantile_sketch: str = "ddsketch",
    goal_semantics: str = "cta_legacy",
    quantile_exact_cap: int = 65536,
    quantile_accuracy: float = 0.001,
    chunk_nodes: int = 524288,
) -> StreamingAggregateSummary:
    """Runs Engine A (1-pass online aggregates) directly on an artifact."""
    stream, inc_coll = open_stream_for_artifact(
        artifact, goal_semantics=goal_semantics, chunk_nodes=chunk_nodes
    )
    return run_streaming_aggregates(
        stream,
        include_collectives_in_traffic=inc_coll,
        min_world_size=max(1, artifact.world_size),
        quantile_sketch=quantile_sketch,
        quantile_exact_cap=quantile_exact_cap,
        quantile_accuracy=quantile_accuracy,
        chunk_nodes=chunk_nodes,
    )


@functools.lru_cache(maxsize=32)
def _aggregate_cached_key(
    fmt: str,
    path_sigs: tuple[tuple[str, int, int], ...],
    world_size: int,
    quantile_sketch: str,
    goal_semantics: str,
) -> StreamingAggregateSummary:
    """Caches one-pass Engine A summaries across aspects in the same run."""
    art = models.TraceArtifact(
        format=fmt,
        paths=tuple(pathlib.Path(p) for p, _, _ in path_sigs),
        world_size=world_size,
    )
    return aggregate_artifact(
        art,
        quantile_sketch=quantile_sketch,
        goal_semantics=goal_semantics,
    )


def aggregate_artifact_cached(
    artifact: models.TraceArtifact,
    *,
    quantile_sketch: str = "ddsketch",
    goal_semantics: str = "cta_legacy",
) -> StreamingAggregateSummary:
    """Runs Engine A once per artifact and shares results across aspects."""
    sigs: list[tuple[str, int, int]] = []
    for path in sorted(artifact.paths):
        stat = path.stat()
        sigs.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return _aggregate_cached_key(
        artifact.format,
        tuple(sigs),
        artifact.world_size,
        quantile_sketch,
        goal_semantics,
    )


def solve_artifact_with_config(
    artifact: models.TraceArtifact,
    config: models.AnalysisConfig,
    *,
    cache_path: pathlib.Path | None = None,
) -> StreamingDagResult:
    """Dispatches an artifact to Engine B or Engine C according to config."""
    tier = resolve_streaming_tier(artifact, config)
    scratch = str(config.scratch_dir) if config.scratch_dir else ""
    if tier in ("frontier", "windowed") or config.graph_engine == "frontier":
        coll_sync = (
            "barrier"
            if tier == "windowed" and config.collective_sync == "none"
            else config.collective_sync
        )
        return solve_artifact_frontier(
            artifact,
            goal_semantics=config.goal_semantics,
            collective_sync=coll_sync,
            window_size=config.window_size,
            fail_on_cycle=config.fail_on_cycle,
            extract_critical_path_ops=config.extract_critical_path_ops,
            scratch_dir=scratch,
        )
    return solve_artifact_outofcore(
        artifact,
        engine=config.graph_engine,
        memory_budget_bytes=config.memory_budget_bytes,
        goal_semantics=config.goal_semantics,
        collective_sync=config.collective_sync,
        fail_on_cycle=config.fail_on_cycle,
        extract_critical_path_ops=config.extract_critical_path_ops,
        delay_attribution=config.delay_attribution,
        scratch_dir=scratch,
        cache_path=cache_path,
    )


def provenance_row(
    group_label: str, trial_index: int, prov: StreamProvenance
) -> dict[str, Any]:
    """Converts a ``StreamProvenance`` into a flat row dict for Polars."""
    return {
        "group": group_label,
        "trial": trial_index,
        "engine": prov.engine,
        "exact": prov.exact,
        "slack_scope": prov.slack_scope,
        "goal_semantics": prov.goal_semantics,
        "collective_sync": prov.collective_sync,
        "makespan_lb_ns": prov.makespan_lb,
        "makespan_ub_ns": prov.makespan_ub,
        "evicted_dep_hits": prov.evicted_dep_hits,
        "lookahead_drops": prov.lookahead_drops,
        "straddling_edges": prov.straddling_edges,
        "dead_end_sinks": prov.dead_end_sinks,
        "has_cycle": prov.has_cycle,
        "unordered_nodes": prov.unordered_nodes,
        "quantile_sketch": prov.quantile_sketch,
        "quantile_error_bound": prov.quantile_error_bound,
        "delay_timing_basis": prov.delay_timing_basis,
    }
