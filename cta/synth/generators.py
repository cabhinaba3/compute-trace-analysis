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


def write_nsys_sqlite(
    path: Path,
    *,
    steps: int = 4,
    gemm_us: int = 800,
    comm_us: int = 400,
    comm_overlap_us: int = 100,
    warmup_penalty_us: int = 1200,
    lock_wait_us: int = 50,
    tensor_active_pct: float = 65.0,
    legacy_gpu_metrics: bool = False,
) -> Path:
    """Create a realistic Nsight Systems `.sqlite` database (supports modern and legacy GPU_METRICS schemas)."""
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                demangledName INTEGER, shortName INTEGER, start INTEGER, end INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                copyKind INTEGER, start INTEGER, end INTEGER, bytes INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                nameId INTEGER, start INTEGER, end INTEGER
            );
            CREATE TABLE NVTX_EVENTS (
                textId INTEGER, text TEXT, start INTEGER, end INTEGER
            );
            CREATE TABLE OSRT_API (
                nameId INTEGER, start INTEGER, end INTEGER
            );
            """
        )
        strings = {
            1: "void cutlass::Kernel<fp16>(half const*, int)",
            2: "ncclDevKernel_AllReduce_Sum_f16_RING_LL",
            3: "cudaLaunchKernel",
            4: "pthread_mutex_lock",
            5: "poll",
            6: "TransformerBlock_0",
        }
        conn.executemany("INSERT INTO StringIds VALUES (?, ?)", list(strings.items()))

        cur_ns = 1_000_000
        for s in range(steps):
            extra_ns = warmup_penalty_us * 1000 if s == 0 else 0
            g_dur = gemm_us * 1000 + extra_ns
            c_dur = comm_us * 1000
            ov_ns = min(comm_overlap_us * 1000, g_dur)

            s_start = cur_ns
            g_start = cur_ns
            g_end = g_start + g_dur
            c_start = g_end - ov_ns
            c_end = c_start + c_dur
            s_end = max(g_end, c_end) + 50_000  # 50 us idle gap per step

            conn.execute("INSERT INTO NVTX_EVENTS VALUES (NULL, ?, ?, ?)", (f"ProfilerStep#{s}", s_start, s_end))
            conn.execute("INSERT INTO NVTX_EVENTS VALUES (6, NULL, ?, ?)", (g_start, g_end))
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, ?, ?, 7)", (g_start, g_end))
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 2, ?, ?, 8)", (c_start, c_end))
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1, ?, ?, 65536, 9)", (g_start, g_start + 20_000))
            conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3, ?, ?)", (g_start - 5_000, g_start))
            conn.execute("INSERT INTO OSRT_API VALUES (4, ?, ?)", (g_start, g_start + lock_wait_us * 1000))
            cur_ns = s_end

        if legacy_gpu_metrics:
            conn.execute("CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, value TEXT)")
            payload = str(
                {
                    "SMs Active [Throughput %]": str(tensor_active_pct + 10.0),
                    "Tensor Active [Throughput %]": str(tensor_active_pct),
                    "SM Issue [Throughput %]": str(tensor_active_pct - 5.0),
                }
            )
            conn.executemany("INSERT INTO GPU_METRICS VALUES (?, ?)", [(1000, payload), (2000, payload)])
        else:
            conn.executescript(
                """
                CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER PRIMARY KEY, metricName TEXT);
                CREATE TABLE GPU_METRICS (metricId INTEGER, timestamp INTEGER, value REAL);
                """
            )
            conn.executemany(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)",
                [
                    (1, "SMs Active [Throughput %]"),
                    (2, "Tensor Active [Throughput %]"),
                    (3, "SM Issue [Throughput %]"),
                ],
            )
            conn.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, ?, ?)",
                [
                    (1, 1000, tensor_active_pct + 10.0),
                    (2, 1000, tensor_active_pct),
                    (3, 1000, tensor_active_pct - 5.0),
                ],
            )
        conn.commit()
    finally:
        conn.close()
    return path


def write_kineto_trace(
    path: Path,
    *,
    steps: int = 3,
    attn_us: float = 400.0,
    gemm_us: float = 600.0,
    kernel_variant: str = "fp16",
) -> Path:
    """Write a PyTorch Profiler / Kineto Chrome trace (.json or .json.gz) with CPU-GPU correlation."""
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    ts = 1000.0
    corr = 100

    gemm_kname = "ampere_fp16_s16816gemm" if kernel_variant == "fp16" else "sm90_wgmma_fp8_gemm_kernel"
    attn_kname = "fmha_cutlassF_f16" if kernel_variant == "fp16" else "flash_attn_fwd_fp8_kernel"

    for s in range(steps):
        step_dur = attn_us + gemm_us + 50.0
        events.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{s}", "ts": ts, "dur": step_dur, "pid": 1, "tid": 1})

        # Attention op -> launch -> GPU kernel
        events.append({"ph": "X", "cat": "cpu_op", "name": "aten::scaled_dot_product_attention", "ts": ts + 2.0, "dur": 15.0, "pid": 1, "tid": 1})
        events.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts + 5.0, "dur": 5.0, "pid": 1, "tid": 1, "args": {"correlation": corr}})
        events.append({"ph": "X", "cat": "kernel", "name": attn_kname, "ts": ts + 10.0, "dur": attn_us, "pid": 0, "tid": 7, "args": {"correlation": corr, "stream": 7}})
        corr += 1

        # Linear op -> launch -> GPU GEMM kernel
        g_ts = ts + 10.0 + attn_us
        events.append({"ph": "X", "cat": "cpu_op", "name": "aten::linear", "ts": ts + 20.0, "dur": 15.0, "pid": 1, "tid": 1})
        events.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts + 25.0, "dur": 5.0, "pid": 1, "tid": 1, "args": {"correlation": corr}})
        events.append({"ph": "X", "cat": "kernel", "name": gemm_kname, "ts": g_ts, "dur": gemm_us, "pid": 0, "tid": 7, "args": {"correlation": corr, "stream": 7}})
        corr += 1

        ts += step_dur + 20.0

    payload = json.dumps({"schemaVersion": 1, "traceEvents": events})
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as gz:
            gz.write(payload)
    else:
        path.write_text(payload)
    return path


def write_ncu_csv(
    path: Path,
    *,
    gemm_dur_ns: float = 500_000.0,
    sm_pct: float = 40.0,
    dram_pct: float = 82.0,
    tensor_pct: float = 35.0,
    flops: float = 2.0e10,
    dram_bytes: float = 1.0e9,
) -> Path:
    """Write a realistic Nsight Compute raw CSV export (with ==PROF== banner and unit row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "==PROF== Connected to process 12345",
        '"ID","Kernel Name","gpu__time_duration.sum","sm__throughput.avg.pct_of_peak_sustained_elapsed",'
        '"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",'
        '"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",'
        '"sm__warps_active.avg.pct_of_peak_sustained_active","FLOPs","dram__bytes.sum",'
        '"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",'
        '"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"',
        '"","","nsecond","%","%","%","%","","byte","%","%"',
        f'"0","ampere_fp16_s16816gemm","{gemm_dur_ns}","{sm_pct}","{dram_pct}","{tensor_pct}","62.5","{flops}","{dram_bytes}","{dram_pct * 0.5}","{sm_pct * 0.4}"',
        '"1","vectorized_elementwise_relu","100000.0","20.0","88.0","0.0","55.0","1e8","2e8","48.0","5.0"',
    ]
    path.write_text("\n".join(lines))
    return path
