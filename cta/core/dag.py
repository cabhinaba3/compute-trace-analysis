from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np
import polars as pl

# Nominal 400GbE (50 GB/s) LogGP parameters used ONLY when comm ops lack recorded duration_ns (e.g. GOAL)
_NOMINAL_LATENCY_NS = 500.0
_NOMINAL_OVERHEAD_NS = 150.0
_NOMINAL_INV_BW_NS_PER_BYTE = 0.02  # 50 GB/s


@dataclass(slots=True, frozen=True)
class DagStats:
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
    """Compute multi-rank DAG critical path (Kahn topological longest-path DP + P2P send->recv links)."""
    n_total = nodes.height
    if n_total == 0:
        return DagStats(0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    # Assign contiguous global vertex indices 0..N-1
    indexed = nodes.with_row_index("vid")
    key_to_vid = indexed.select("rank", "node_id", "vid")

    # Map intra-rank edges to global vertex indices
    intra = (
        edges.join(key_to_vid, left_on=["rank", "src_id"], right_on=["rank", "node_id"], how="inner")
        .rename({"vid": "u"})
        .join(key_to_vid, left_on=["rank", "dst_id"], right_on=["rank", "node_id"], how="inner")
        .rename({"vid": "v"})
        .select("u", "v")
    )

    # Match inter-rank P2P send -> recv pairs vectorially by (src_rank, dst_rank, tag, sequence_ordinal)
    sends = (
        indexed.filter(pl.col("op_class") == "comm_send")
        .with_columns(pl.cum_count("vid").over(["src_rank", "dst_rank", "tag"]).alias("_seq"))
        .select(pl.col("src_rank"), pl.col("dst_rank"), pl.col("tag"), pl.col("_seq"), pl.col("vid").alias("u"))
    )
    recvs = (
        indexed.filter(pl.col("op_class") == "comm_recv")
        .with_columns(pl.cum_count("vid").over(["src_rank", "dst_rank", "tag"]).alias("_seq"))
        .select(pl.col("src_rank"), pl.col("dst_rank"), pl.col("tag"), pl.col("_seq"), pl.col("vid").alias("v"))
    )
    inter = sends.join(recvs, on=["src_rank", "dst_rank", "tag", "_seq"], how="inner").select("u", "v")

    all_edges = pl.concat([intra, inter]).unique() if (intra.height or inter.height) else intra
    u_arr = all_edges["u"].to_numpy() if all_edges.height else np.empty(0, dtype=np.uint32)
    v_arr = all_edges["v"].to_numpy() if all_edges.height else np.empty(0, dtype=np.uint32)

    # Compute effective node weights in ns
    op_cls = indexed["op_class"].to_list()
    raw_dur = indexed["duration_ns"].to_numpy().astype(np.float64, copy=False)
    sz_bytes = indexed["size_bytes"].to_numpy().astype(np.int64, copy=False)
    is_comm = np.array([c.startswith("comm_") for c in op_cls], dtype=bool)
    is_comp = np.array([c == "compute" for c in op_cls], dtype=bool)

    weights = raw_dur.copy()
    unrecorded_comm = is_comm & (weights <= 0.0)
    if np.any(unrecorded_comm):
        # For send+recv pair, attribute half wire cost to send and half to recv; full to collective
        is_p2p = np.array([c in ("comm_send", "comm_recv") for c in op_cls], dtype=bool) & unrecorded_comm
        is_coll = unrecorded_comm & ~is_p2p
        weights[is_p2p] = 0.5 * (_NOMINAL_LATENCY_NS + 2 * _NOMINAL_OVERHEAD_NS + sz_bytes[is_p2p] * _NOMINAL_INV_BW_NS_PER_BYTE)
        weights[is_coll] = _NOMINAL_LATENCY_NS + 2 * _NOMINAL_OVERHEAD_NS + sz_bytes[is_coll] * _NOMINAL_INV_BW_NS_PER_BYTE

    # Build CSR adjacency list for O(V + E) topological DP
    m = len(u_arr)
    indeg = np.zeros(n_total, dtype=np.int32)
    if m > 0:
        order = np.argsort(u_arr, kind="stable")
        u_sorted = u_arr[order]
        v_sorted = v_arr[order]
        np.add.at(indeg, v_sorted, 1)
        row_ptr = np.zeros(n_total + 1, dtype=np.int64)
        np.add.at(row_ptr[1:], u_sorted, 1)
        np.cumsum(row_ptr, out=row_ptr)
    else:
        v_sorted = np.empty(0, dtype=np.uint32)
        row_ptr = np.zeros(n_total + 1, dtype=np.int64)

    est_start = np.zeros(n_total, dtype=np.float64)
    est_finish = weights.copy()
    depth = np.ones(n_total, dtype=np.int32)
    parent = np.full(n_total, -1, dtype=np.int64)

    q: deque[int] = deque(int(i) for i in np.flatnonzero(indeg == 0))
    topo: list[int] = []
    while q:
        u = q.popleft()
        topo.append(u)
        fu = est_finish[u]
        du = depth[u]
        for idx in range(row_ptr[u], row_ptr[u + 1]):
            v = int(v_sorted[idx])
            if fu > est_start[v] or (fu == est_start[v] and du + 1 > depth[v]):
                est_start[v] = fu
                est_finish[v] = fu + weights[v]
                depth[v] = du + 1
                parent[v] = u
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) < n_total:
        warnings.warn(
            f"Execution graph has a cycle ({n_total - len(topo)} of {n_total} nodes unordered); "
            "critical-path results are approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Cycle fallback: process remaining vertices in index order
        seen = set(topo)
        topo.extend(i for i in range(n_total) if i not in seen)

    sink = int(np.argmax(est_finish))
    makespan = float(est_finish[sink])

    # Backward pass for node slack
    latest_finish = np.full(n_total, makespan, dtype=np.float64)
    for u in reversed(topo):
        start_limit = makespan
        has_child = False
        for idx in range(row_ptr[u], row_ptr[u + 1]):
            v = int(v_sorted[idx])
            has_child = True
            cand_lf = latest_finish[v] - weights[v]
            if cand_lf < start_limit:
                start_limit = cand_lf
        if has_child:
            latest_finish[u] = start_limit

    slack = np.maximum(0.0, latest_finish - est_finish)
    tol = max(1e-6, makespan * 1e-9)
    zero_slack_ratio = float(np.mean(slack <= tol))
    mean_slack = float(np.mean(slack))

    # Trace exact critical-path chain back from sink
    cp_nodes: list[int] = []
    curr = sink
    visited_cp: set[int] = set()
    while curr != -1 and curr not in visited_cp:
        visited_cp.add(curr)
        cp_nodes.append(curr)
        curr = int(parent[curr])

    cp_idx = np.asarray(cp_nodes, dtype=np.int64)
    cp_comp = float(np.sum(weights[cp_idx[is_comp[cp_idx]]]))
    cp_comm = float(np.sum(weights[cp_idx[is_comm[cp_idx]]]))
    cp_bytes = int(np.sum(sz_bytes[cp_idx[is_comm[cp_idx]]]))

    return DagStats(
        num_nodes=n_total,
        num_edges=m,
        dag_depth=int(depth[sink]),
        critical_path_ns=makespan,
        cp_compute_ns=cp_comp,
        cp_exposed_comm_ns=cp_comm,
        cp_comm_bytes=cp_bytes,
        mean_slack_ns=mean_slack,
        zero_slack_ratio=zero_slack_ratio,
    )
