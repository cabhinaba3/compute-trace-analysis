"""Critical-path and slack analysis of multi-rank execution graphs."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import polars as pl

from cta import _native as native

# Nominal 400GbE (50 GB/s) LogGP parameters, used ONLY when comm ops have no
# recorded duration_ns (e.g. GOAL).
_NOMINAL_LATENCY_NS = 500.0
_NOMINAL_OVERHEAD_NS = 150.0
_NOMINAL_INV_BW_NS_PER_BYTE = 0.02  # 50 GB/s
# Columns that identify one point-to-point channel.
_P2P_CHANNEL_COLUMNS = ["src_rank", "dst_rank", "message_tag"]


@dataclasses.dataclass(slots=True, frozen=True)
class DagStats:
    """Size, critical path and slack statistics of an execution graph.

    Attributes:
        num_nodes: Number of nodes (ops) across all ranks.
        num_edges: Number of distinct edges: intra-rank dependencies plus
            matched inter-rank send -> recv links.
        dag_depth: Largest number of nodes on a longest (by time) path into
            any node.
        critical_path_ns: Length of the critical path (the makespan).
        cp_compute_ns: Compute time on the critical path.
        cp_exposed_comm_ns: Communication time on the critical path.
        cp_comm_bytes: Bytes moved by the communication ops on the critical
            path.
        mean_slack_ns: Mean slack over all nodes.
        zero_slack_ratio: Fraction of nodes with (near-)zero slack.
    """

    num_nodes: int
    num_edges: int
    dag_depth: int
    critical_path_ns: float
    cp_compute_ns: float
    cp_exposed_comm_ns: float
    cp_comm_bytes: int
    mean_slack_ns: float
    zero_slack_ratio: float


def solve_distributed_dag(nodes: pl.DataFrame, edges: pl.DataFrame) -> DagStats:
    """Computes the critical path and slack of a multi-rank execution graph.

    The graph spans all ranks: intra-rank dependency edges plus inter-rank
    links that pair the k-th send with the k-th receive of each
    (src_rank, dst_rank, message_tag) channel. A Kahn topological
    longest-path pass, O(V + E), finds the critical path and a backward
    pass the slack of every node. Where predecessors tie on finish time and
    depth, the lowest vertex id is the critical-path parent, so the path
    does not depend on the order of the rows. Comm ops without a recorded
    duration are weighted with a nominal 400GbE LogGP cost; a send/recv
    pair splits that cost between its two ends.

    If the graph has a cycle, a ``RuntimeWarning`` is issued and the nodes
    left unordered are processed in index order, so the results are
    approximate.

    Args:
        nodes: One row per op, with columns ``rank``, ``node_id``,
            ``op_class`` (``"compute"``, ``"comm_send"``, ``"comm_recv"`` or
            another ``"comm_*"`` class), ``duration_ns``, ``size_bytes``,
            ``src_rank``, ``dst_rank`` and ``message_tag``.
        edges: One row per intra-rank dependency, with columns ``rank``,
            ``src_id`` and ``dst_id`` (node ids within that rank).

    Returns:
        The graph statistics; all zero for a graph without nodes.
    """
    num_nodes = nodes.height
    if num_nodes == 0:
        return DagStats(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    # Contiguous global vertex ids 0..N-1 across all ranks.
    indexed = nodes.with_row_index("vid")
    edge_src, edge_dst = _global_edges(indexed, edges)

    op_class = indexed["op_class"]
    is_comm = op_class.str.starts_with("comm_").fill_null(False).to_numpy()
    is_compute = (op_class == "compute").fill_null(False).to_numpy()
    is_p2p = (
        op_class.is_in(["comm_send", "comm_recv"]).fill_null(False).to_numpy()
    )
    duration_ns = (
        indexed["duration_ns"].to_numpy().astype(np.float64, copy=False)
    )
    size_bytes = indexed["size_bytes"].to_numpy().astype(np.int64, copy=False)
    weights_ns = _node_weights_ns(duration_ns, is_comm, is_p2p, size_bytes)

    solved = native.solve_dag(
        weights_ns,
        np.ascontiguousarray(is_comm),
        np.ascontiguousarray(is_compute),
        np.ascontiguousarray(size_bytes),
        np.ascontiguousarray(edge_src, dtype=np.uint32),
        np.ascontiguousarray(edge_dst, dtype=np.uint32),
    )
    if solved["unordered"]:
        warnings.warn(
            f"Execution graph has a cycle ({solved['unordered']} of "
            f"{num_nodes} nodes unordered); "
            "critical-path results are approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return DagStats(
        num_nodes=num_nodes,
        num_edges=solved["num_edges"],
        dag_depth=solved["depth"],
        critical_path_ns=solved["makespan"],
        cp_compute_ns=solved["critical_compute"],
        cp_exposed_comm_ns=solved["critical_comm"],
        cp_comm_bytes=solved["critical_comm_bytes"],
        mean_slack_ns=solved["mean_slack"],
        zero_slack_ratio=solved["zero_slack_ratio"],
    )


def _global_edges(
    indexed: pl.DataFrame, edges: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the source and destination vertex ids of all edges.

    Args:
        indexed: The nodes, with their global vertex id in column ``vid``.
        edges: Intra-rank dependency edges in per-rank node ids.

    Returns:
        Two equally long arrays: edge sources and edge destinations. An
        edge can appear more than once.
    """
    vertex_ids = indexed.select("rank", "node_id", "vid")

    # Map intra-rank edges to global vertex ids.
    intra = (
        edges.join(
            vertex_ids,
            left_on=["rank", "src_id"],
            right_on=["rank", "node_id"],
            how="inner",
        )
        .rename({"vid": "u"})
        .join(
            vertex_ids,
            left_on=["rank", "dst_id"],
            right_on=["rank", "node_id"],
            how="inner",
        )
        .rename({"vid": "v"})
        .select("u", "v")
    )

    # Pair the k-th send with the k-th recv of each channel, vectorised.
    sends = _p2p_endpoints(indexed, "comm_send", "u")
    recvs = _p2p_endpoints(indexed, "comm_recv", "v")
    inter = sends.join(
        recvs, on=[*_P2P_CHANNEL_COLUMNS, "_seq"], how="inner"
    ).select("u", "v")

    # Repeated edges are dropped by the solver.
    all_edges = pl.concat([intra, inter])
    return (
        all_edges["u"].to_numpy().astype(np.uint32, copy=False),
        all_edges["v"].to_numpy().astype(np.uint32, copy=False),
    )


def _p2p_endpoints(
    indexed: pl.DataFrame, op_class: str, vertex_column: str
) -> pl.DataFrame:
    """Returns the send or recv ops numbered by order within their channel.

    Args:
        indexed: The nodes, with their global vertex id in column ``vid``.
        op_class: ``"comm_send"`` or ``"comm_recv"``.
        vertex_column: Name to give the vertex id column.

    Returns:
        The channel columns, the sequence number ``_seq`` within the channel
        and the vertex id.
    """
    return (
        indexed.filter(pl.col("op_class") == op_class)
        .with_columns(
            pl.cum_count("vid").over(_P2P_CHANNEL_COLUMNS).alias("_seq")
        )
        .select(
            pl.col("src_rank"),
            pl.col("dst_rank"),
            pl.col("message_tag"),
            pl.col("_seq"),
            pl.col("vid").alias(vertex_column),
        )
    )


def _nominal_comm_cost_ns(size_bytes: np.ndarray) -> np.ndarray:
    """Returns the nominal LogGP cost of moving ``size_bytes`` over the wire."""
    return (
        _NOMINAL_LATENCY_NS
        + 2 * _NOMINAL_OVERHEAD_NS
        + size_bytes * _NOMINAL_INV_BW_NS_PER_BYTE
    )


def _node_weights_ns(
    duration_ns: np.ndarray,
    is_comm: np.ndarray,
    is_p2p: np.ndarray,
    size_bytes: np.ndarray,
) -> np.ndarray:
    """Returns node weights: the recorded duration, or a nominal comm cost.

    Comm ops without a positive recorded duration get the nominal wire cost;
    for a send or recv it is halved, since each end carries half of it.

    Args:
        duration_ns: Recorded duration of each node.
        is_comm: Whether each node is a communication op.
        is_p2p: Whether each node is a send or a receive.
        size_bytes: Bytes moved by each node.

    Returns:
        The weight of each node.
    """
    weights_ns = duration_ns.copy()
    is_unrecorded_comm = is_comm & (weights_ns <= 0.0)
    if np.any(is_unrecorded_comm):
        is_unrecorded_p2p = is_p2p & is_unrecorded_comm
        is_collective = is_unrecorded_comm & ~is_unrecorded_p2p
        weights_ns[is_unrecorded_p2p] = 0.5 * _nominal_comm_cost_ns(
            size_bytes[is_unrecorded_p2p]
        )
        weights_ns[is_collective] = _nominal_comm_cost_ns(
            size_bytes[is_collective]
        )
    return weights_ns
