"""Synthetic trace fixture builders for offline CI tests and SoftwareX quickstarts."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from cta.adapters.chakra.schema import (
    COMM_COLL_NODE,
    COMP_NODE,
    GlobalMetadataMsg,
    NodeMsg,
    encode_varint,
)
from cta.adapters.goal.reader import (
    GOAL_BIN_NODE_DTYPE,
    GOAL_MAGIC_COOKIE,
    OPTYPE_CALC,
    OPTYPE_RECV,
    OPTYPE_SEND,
    _parse_goal_text,
)


def _add_int_attr(node: Any, name: str, val: int) -> None:
    a = node.attr.add()
    a.name = name
    a.int64_val = val


def write_chakra_et_run(
    out_dir: Path,
    *,
    world_size: int = 4,
    compute_us: int = 1000,
    allreduce_bytes: int = 400_000,
    steps: int = 2,
    straggler_rank: int | None = None,
    straggler_factor: float = 1.0,
) -> list[Path]:
    """Write `world_size` valid MLCommons Chakra `.et` binary protobuf shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for r in range(world_size):
        buf = bytearray()
        meta = GlobalMetadataMsg(version="0.0.4")
        mbytes = meta.SerializeToString()
        buf.extend(encode_varint(len(mbytes)))
        buf.extend(mbytes)

        nid = 0
        prev_id: int | None = None
        comp_dur = int(compute_us * straggler_factor) if r == straggler_rank else compute_us

        for s in range(steps):
            cnode = NodeMsg(id=nid, name=f"aten::mm_step{s}", type=COMP_NODE, duration_micros=comp_dur)
            if prev_id is not None:
                cnode.data_deps.append(prev_id)
            _add_int_attr(cnode, "num_ops", 2 * 1024 * 1024)
            _add_int_attr(cnode, "tensor_size", allreduce_bytes)
            cb = cnode.SerializeToString()
            buf.extend(encode_varint(len(cb)))
            buf.extend(cb)
            comp_id = nid
            nid += 1

            ar_node = NodeMsg(
                id=nid,
                name=f"nccl:all_reduce_step{s}",
                type=COMM_COLL_NODE,
                duration_micros=max(10, allreduce_bytes // 1000),
            )
            ar_node.data_deps.append(comp_id)
            _add_int_attr(ar_node, "comm_type", 0)  # ALL_REDUCE
            _add_int_attr(ar_node, "comm_size", allreduce_bytes)
            ab = ar_node.SerializeToString()
            buf.extend(encode_varint(len(ab)))
            buf.extend(ab)
            prev_id = nid
            nid += 1

        shard = out_dir / f"trace.rank{r}.et"
        shard.write_bytes(bytes(buf))
        paths.append(shard)

    return paths


def write_chakra_host_et(path: Path, *, steps: int = 2, compute_us: float = 500.0, comm_bytes: int = 200_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    nid = 1
    prev = None
    for s in range(steps):
        nodes.append(
            {
                "id": nid,
                "name": f"aten::linear_{s}",
                "ctrl_deps": [prev] if prev is not None else [],
                "dur_us": compute_us,
            }
        )
        comp_id = nid
        nid += 1
        nodes.append(
            {
                "id": nid,
                "name": f"c10d::all_reduce_{s}",
                "ctrl_deps": [comp_id],
                "dur_us": comm_bytes / 1000.0,
                "comm_bytes": comm_bytes,
            }
        )
        prev = nid
        nid += 1
    path.write_text(json.dumps({"schema": "1.0.2-chakra.0.0.4", "nodes": nodes}))
    return path


def write_goal_allreduce(
    path: Path,
    *,
    world_size: int = 4,
    compute_ns: int = 1_000_000,
    allreduce_bytes: int = 400_000,
    steps: int = 2,
    algo: str = "ring",
) -> Path:
    """Generate a valid multi-rank `.goal` text schedule implementing Ring or Direct AllReduce."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"num_ranks {world_size}", ""]
    p = world_size

    for r in range(p):
        lines.append(f"rank {r} {{")
        prev_lbl: str | None = None
        lid = 0

        def _next_lbl() -> str:
            nonlocal lid
            lid += 1
            return f"l{lid}"

        for s in range(steps):
            c_lbl = _next_lbl()
            lines.append(f"{c_lbl}: calc {compute_ns} cpu 0")
            if prev_lbl:
                lines.append(f"{c_lbl} requires {prev_lbl}")
            prev_lbl = c_lbl

            if p <= 1:
                continue

            if algo == "ring":
                # Ring AllReduce: 2 * (P - 1) sequential phases, chunk size = B / P
                chunk = allreduce_bytes // p
                nxt = (r + 1) % p
                prv = (r - 1 + p) % p
                for phase in range(2 * (p - 1)):
                    tag = s * 100 + phase
                    s_lbl = _next_lbl()
                    r_lbl = _next_lbl()
                    lines.append(f"{s_lbl}: send {chunk}b to {nxt} tag {tag} cpu 0 nic 0")
                    lines.append(f"{r_lbl}: recv {chunk}b from {prv} tag {tag} cpu 0 nic 0")
                    lines.append(f"{s_lbl} requires {prev_lbl}")
                    lines.append(f"{r_lbl} requires {prev_lbl}")
                    # Synchronize on recv before next ring phase
                    prev_lbl = r_lbl
            else:
                # Direct all-to-all exchange: each rank sends full B bytes to every peer concurrently
                barrier_calc = _next_lbl()
                lines.append(f"{barrier_calc}: calc 0 cpu 0")
                for peer in range(p):
                    if peer == r:
                        continue
                    tag = s * 100 + 1
                    s_lbl = _next_lbl()
                    r_lbl = _next_lbl()
                    lines.append(f"{s_lbl}: send {allreduce_bytes}b to {peer} tag {tag} cpu 0 nic 0")
                    lines.append(f"{r_lbl}: recv {allreduce_bytes}b from {peer} tag {tag} cpu 0 nic 0")
                    lines.append(f"{s_lbl} requires {prev_lbl}")
                    lines.append(f"{r_lbl} requires {prev_lbl}")
                    lines.append(f"{barrier_calc} requires {s_lbl}")
                    lines.append(f"{barrier_calc} requires {r_lbl}")
                prev_lbl = barrier_calc

        lines.append("}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


def compile_goal_to_bin(goal_path: Path, bin_path: Path) -> Path:
    """Compile a `.goal` text file into a spec-compliant SPCL LogGOPSim `.bin` schedule (`MAGIC_COOKIE = 4223`)."""
    nodes_df, edges_df = _parse_goal_text(goal_path)
    num_ranks = int(np.max(nodes_df["rank"].to_numpy())) + 1 if nodes_df.height else 1

    rank_blobs: list[bytes] = []
    for r in range(num_ranks):
        rn = nodes_df.filter(nodes_df["rank"] == r).sort("node_id")
        re_df = edges_df.filter(edges_df["rank"] == r)
        n_cnt = rn.height
        if n_cnt == 0:
            rank_blobs.append(struct.pack("<II", 0, 0))
            continue

        indeg = np.zeros(n_cnt, dtype=np.uint32)
        req_out: list[list[int]] = [[] for _ in range(n_cnt)]
        ireq_out: list[list[int]] = [[] for _ in range(n_cnt)]
        for erow in re_df.iter_rows(named=True):
            u, v, k = int(erow["src_id"]), int(erow["dst_id"]), str(erow["dep_kind"])
            indeg[v] += 1
            if k == "irequires":
                ireq_out[u].append(v)
            else:
                req_out[u].append(v)

        roots = np.flatnonzero(indeg == 0).astype(np.uint32)
        rec = np.zeros(n_cnt, dtype=GOAL_BIN_NODE_DTYPE)
        apdx_list: list[int] = []

        for idx, row in enumerate(rn.iter_rows(named=True)):
            op_cls = str(row["op_class"])
            rec[idx]["deps_cnt"] = indeg[idx]
            if op_cls == "comm_send":
                rec[idx]["op_type"] = OPTYPE_SEND
                rec[idx]["peer"] = int(row["dst_rank"])
                rec[idx]["size"] = int(row["size_bytes"])
            elif op_cls == "comm_recv":
                rec[idx]["op_type"] = OPTYPE_RECV
                rec[idx]["peer"] = int(row["src_rank"])
                rec[idx]["size"] = int(row["size_bytes"])
            else:
                rec[idx]["op_type"] = OPTYPE_CALC
                rec[idx]["peer"] = 0
                rec[idx]["size"] = int(row["duration_ns"])
            rec[idx]["tag"] = int(row["tag"])
            rec[idx]["proc"] = 0
            rec[idx]["nic"] = 0

            r_deps = req_out[idx]
            rec[idx]["num_deps"] = len(r_deps)
            rec[idx]["deps_apdx"] = len(apdx_list) if r_deps else 999
            apdx_list.extend(r_deps)

            ir_deps = ireq_out[idx]
            rec[idx]["num_startdeps"] = len(ir_deps)
            rec[idx]["startdeps_apdx"] = len(apdx_list) if ir_deps else 0xFFFFFFFF
            apdx_list.extend(ir_deps)

        apdx_arr = np.asarray(apdx_list, dtype="<u4")
        blob = b"".join(
            [
                struct.pack("<II", n_cnt, len(roots)),
                roots.astype("<u4").tobytes(),
                rec.tobytes(),
                apdx_arr.tobytes(),
            ]
        )
        rank_blobs.append(blob)

    header_rel = 6 + 16 * num_ranks
    jtable: list[int] = []
    cur = header_rel
    for b in rank_blobs:
        jtable.extend([cur, cur + len(b)])
        cur += len(b)

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("wb") as fh:
        fh.write(struct.pack("<QIBB", GOAL_MAGIC_COOKIE, num_ranks, 0, 0))
        fh.write(np.asarray(jtable, dtype="<u8").tobytes())
        for b in rank_blobs:
            fh.write(b)
    return bin_path
