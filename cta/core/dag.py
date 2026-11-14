"""Critical-path and slack analysis of multi-rank execution graphs."""

from __future__ import annotations

import collections
import dataclasses
import warnings

import numpy as np
import polars as pl

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
    pass the slack of every node. Comm ops without a recorded duration are
    weighted with a nominal 400GbE LogGP cost; a send/recv pair splits
    that cost between its two ends.

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

    op_classes = indexed["op_class"].to_list()
    duration_ns = (
        indexed["duration_ns"].to_numpy().astype(np.float64, copy=False)
    )
    size_bytes = indexed["size_bytes"].to_numpy().astype(np.int64, copy=False)
    is_comm = np.array(
        [op_class.startswith("comm_") for op_class in op_classes], dtype=bool
    )
    is_compute = np.array(
        [op_class == "compute" for op_class in op_classes], dtype=bool
    )
    weights_ns = _node_weights_ns(duration_ns, op_classes, is_comm, size_bytes)

    row_ptr, successors, in_degree = _csr_adjacency(
        edge_src, edge_dst, num_nodes
    )
    topo_order, earliest_finish_ns, depth, parent = _forward_pass(
        row_ptr, successors, in_degree, weights_ns
    )
    if len(topo_order) < num_nodes:
        warnings.warn(
            f"Execution graph has a cycle ({num_nodes - len(topo_order)} of "
            f"{num_nodes} nodes unordered); "
            "critical-path results are approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Cycle fallback: process the remaining vertices in index order.
        seen = set(topo_order)
        topo_order.extend(i for i in range(num_nodes) if i not in seen)

    sink = int(np.argmax(earliest_finish_ns))
    makespan_ns = float(earliest_finish_ns[sink])
    latest_finish_ns = _backward_pass(
        topo_order, row_ptr, successors, weights_ns, makespan_ns
    )
    slack_ns = np.maximum(0.0, latest_finish_ns - earliest_finish_ns)
    tolerance_ns = max(1e-6, makespan_ns * 1e-9)
    zero_slack_ratio = float(np.mean(slack_ns <= tolerance_ns))
    mean_slack_ns = float(np.mean(slack_ns))

    critical_nodes = _critical_chain(parent, sink)
    critical_compute = critical_nodes[is_compute[critical_nodes]]
    critical_comm = critical_nodes[is_comm[critical_nodes]]
    cp_compute_ns = float(np.sum(weights_ns[critical_compute]))
    cp_comm_ns = float(np.sum(weights_ns[critical_comm]))
    cp_comm_bytes = int(np.sum(size_bytes[critical_comm]))

    return DagStats(
        num_nodes=num_nodes,
        num_edges=len(edge_src),
        dag_depth=int(np.max(depth)),
        critical_path_ns=makespan_ns,
        cp_compute_ns=cp_compute_ns,
        cp_exposed_comm_ns=cp_comm_ns,
        cp_comm_bytes=cp_comm_bytes,
        mean_slack_ns=mean_slack_ns,
        zero_slack_ratio=zero_slack_ratio,
    )


def _global_edges(
    indexed: pl.DataFrame, edges: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the source and destination vertex ids of all distinct edges.

    Args:
        indexed: The nodes, with their global vertex id in column ``vid``.
        edges: Intra-rank dependency edges in per-rank node ids.

    Returns:
        Two equally long arrays: edge sources and edge destinations.
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

    all_edges = (
        pl.concat([intra, inter]).unique()
        if (intra.height or inter.height)
        else intra
    )
    edge_src = (
        all_edges["u"].to_numpy()
        if all_edges.height
        else np.empty(0, dtype=np.uint32)
    )
    edge_dst = (
        all_edges["v"].to_numpy()
        if all_edges.height
        else np.empty(0, dtype=np.uint32)
    )
    return edge_src, edge_dst


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
    op_classes: list[str],
    is_comm: np.ndarray,
    size_bytes: np.ndarray,
) -> np.ndarray:
    """Returns node weights: the recorded duration, or a nominal comm cost.

    Comm ops without a positive recorded duration get the nominal wire cost;
    for a send or recv it is halved, since each end carries half of it.
    """
    weights_ns = duration_ns.copy()
    is_unrecorded_comm = is_comm & (weights_ns <= 0.0)
    if np.any(is_unrecorded_comm):
        is_p2p = (
            np.array(
                [
                    op_class in ("comm_send", "comm_recv")
                    for op_class in op_classes
                ],
                dtype=bool,
            )
            & is_unrecorded_comm
        )
        is_collective = is_unrecorded_comm & ~is_p2p
        weights_ns[is_p2p] = 0.5 * _nominal_comm_cost_ns(size_bytes[is_p2p])
        weights_ns[is_collective] = _nominal_comm_cost_ns(
            size_bytes[is_collective]
        )
    return weights_ns


def _csr_adjacency(
    edge_src: np.ndarray, edge_dst: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds a CSR adjacency list for an O(V + E) topological pass.

    Args:
        edge_src: Source vertex of each edge.
        edge_dst: Destination vertex of each edge.
        num_nodes: Number of vertices.

    Returns:
        ``(row_ptr, successors, in_degree)``. The successors of vertex ``u``
        are ``successors[row_ptr[u]:row_ptr[u + 1]]``.
    """
    in_degree = np.zeros(num_nodes, dtype=np.int32)
    if len(edge_src) > 0:
        order = np.argsort(edge_src, kind="stable")
        sorted_src = edge_src[order]
        successors = edge_dst[order]
        np.add.at(in_degree, successors, 1)
        row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.add.at(row_ptr[1:], sorted_src, 1)
        np.cumsum(row_ptr, out=row_ptr)
    else:
        successors = np.empty(0, dtype=np.uint32)
        row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    return row_ptr, successors, in_degree


def _forward_pass(
    row_ptr: np.ndarray,
    successors: np.ndarray,
    in_degree: np.ndarray,
    weights_ns: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Runs Kahn's algorithm with a longest-path DP over the vertices.

    Consumes ``in_degree``. On equal start times the deeper predecessor
    wins.

    Returns:
        ``(topo_order, earliest_finish_ns, depth, parent)``. ``topo_order``
        lacks the vertices on or behind a cycle. ``parent`` is the
        predecessor on the longest path (-1 for none).
    """
    num_nodes = len(weights_ns)
    earliest_start_ns = np.zeros(num_nodes, dtype=np.float64)
    earliest_finish_ns = weights_ns.copy()
    depth = np.ones(num_nodes, dtype=np.int32)
    parent = np.full(num_nodes, -1, dtype=np.int64)

    ready: collections.deque[int] = collections.deque(
        int(i) for i in np.flatnonzero(in_degree == 0)
    )
    topo_order: list[int] = []
    while ready:
        u = ready.popleft()
        topo_order.append(u)
        u_finish_ns = earliest_finish_ns[u]
        u_depth = depth[u]
        for edge_idx in range(row_ptr[u], row_ptr[u + 1]):
            v = int(successors[edge_idx])
            if u_finish_ns > earliest_start_ns[v] or (
                u_finish_ns == earliest_start_ns[v] and u_depth + 1 > depth[v]
            ):
                earliest_start_ns[v] = u_finish_ns
                earliest_finish_ns[v] = u_finish_ns + weights_ns[v]
                depth[v] = u_depth + 1
                parent[v] = u
            in_degree[v] -= 1
            if in_degree[v] == 0:
                ready.append(v)
    return topo_order, earliest_finish_ns, depth, parent


def _backward_pass(
    topo_order: list[int],
    row_ptr: np.ndarray,
    successors: np.ndarray,
    weights_ns: np.ndarray,
    makespan_ns: float,
) -> np.ndarray:
    """Returns each vertex's latest finish time that keeps the makespan.

    A vertex without successors may finish as late as the makespan; any
    other vertex must finish before the latest start of each successor.
    """
    latest_finish_ns = np.full(len(weights_ns), makespan_ns, dtype=np.float64)
    for u in reversed(topo_order):
        min_successor_start_ns = makespan_ns
        has_successor = False
        for edge_idx in range(row_ptr[u], row_ptr[u + 1]):
            v = int(successors[edge_idx])
            has_successor = True
            latest_start_ns = latest_finish_ns[v] - weights_ns[v]
            if latest_start_ns < min_successor_start_ns:
                min_successor_start_ns = latest_start_ns
        if has_successor:
            latest_finish_ns[u] = min_successor_start_ns
    return latest_finish_ns


def _critical_chain(parent: np.ndarray, sink: int) -> np.ndarray:
    """Returns the critical-path vertices, from the sink back to its source."""
    chain: list[int] = []
    visited: set[int] = set()
    node = sink
    while node != -1 and node not in visited:
        visited.add(node)
        chain.append(node)
        node = int(parent[node])
    return np.asarray(chain, dtype=np.int64)
