"""GOAL (.goal text and SPCL txt2bin .bin binary) schedule adapter."""

from __future__ import annotations

import mmap
import re
import struct
from pathlib import Path

import numpy as np
import polars as pl

from cta.adapters.chakra.reader import EDGES_SCHEMA, NODES_SCHEMA, TRAFFIC_SCHEMA
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

GOAL_MAGIC_COOKIE = 4223
GOAL_MAGIC_INVALID = 2342
OPTYPE_SEND = 1
OPTYPE_RECV = 2
OPTYPE_CALC = 3

# Exact 39-byte packed struct from SPCL LogGOPSim Parser.hpp
GOAL_BIN_NODE_DTYPE = np.dtype(
    [
        ("deps_cnt", "<u4"),
        ("op_type", "<i1"),
        ("peer", "<u4"),
        ("size", "<u8"),
        ("tag", "<u4"),
        ("proc", "<u1"),
        ("nic", "<u1"),
        ("num_deps", "<u4"),
        ("deps_apdx", "<u4"),
        ("num_startdeps", "<u4"),
        ("startdeps_apdx", "<u4"),
    ],
    align=False,
)

_SIZE_MUL = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
_RANK_HDR_RE = re.compile(r"^rank\s+(\d+)\s*\{?")
_DEP_RE = re.compile(r"^(\S+)\s+(i?requires)\s+(\S+)$")
_SEND_RE = re.compile(
    r"^(?:(?P<lbl>[^:\s]+):\s*)?send\s+(?P<sz>\d+)(?P<unit>[kmgt]?b)?\s+to\s+(?P<peer>\d+)"
    r"(?:\s+tag\s+(?P<tag>\d+))?(?:\s+cpu\s+(?P<cpu>\d+))?(?:\s+nic\s+(?P<nic>\d+))?$",
    re.IGNORECASE,
)
_RECV_RE = re.compile(
    r"^(?:(?P<lbl>[^:\s]+):\s*)?recv\s+(?P<sz>\d+)(?P<unit>[kmgt]?b)?\s+from\s+(?P<peer>\d+)"
    r"(?:\s+tag\s+(?P<tag>\d+))?(?:\s+cpu\s+(?P<cpu>\d+))?(?:\s+nic\s+(?P<nic>\d+))?$",
    re.IGNORECASE,
)
_CALC_RE = re.compile(
    r"^(?:(?P<lbl>[^:\s]+):\s*)?calc\s+(?P<dur>\d+)"
    r"(?:\s+cpu\s+(?P<cpu>\d+))?(?:\s+nic\s+(?P<nic>\d+))?$",
    re.IGNORECASE,
)


def _is_goal_bin(path: Path) -> bool:
    try:
        if path.stat().st_size < 14:
            return False
        with path.open("rb") as fh:
            (cookie,) = struct.unpack("<Q", fh.read(8))
        return cookie in (GOAL_MAGIC_COOKIE, GOAL_MAGIC_INVALID)
    except OSError:
        return False


def _decode_goal_mmap(mm: mmap.mmap, path: Path) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    node_frames: list[pl.DataFrame] = []
    edge_frames: list[pl.DataFrame] = []
    (cookie,) = struct.unpack_from("<Q", mm, 0)
    if cookie not in (GOAL_MAGIC_COOKIE, GOAL_MAGIC_INVALID):
        raise ValueError(f"Invalid GOAL binary magic cookie {cookie} in {path}")
    map_base = 8
    num_ranks, _, _ = struct.unpack_from("<IBB", mm, map_base)
    jtable = np.frombuffer(mm, dtype="<u8", count=2 * num_ranks, offset=map_base + 6)

    for r in range(num_ranks):
        start_rel = int(jtable[2 * r])
        end_rel = int(jtable[2 * r + 1])
        if end_rel <= start_rel:
            continue
        r_abs = map_base + start_rel
        num_nodes, num_roots = struct.unpack_from("<II", mm, r_abs)
        if num_nodes == 0:
            continue
        nodes_abs = r_abs + 8 + 4 * num_roots
        raw = np.frombuffer(mm, dtype=GOAL_BIN_NODE_DTYPE, count=num_nodes, offset=nodes_abs)

        nids = np.arange(num_nodes, dtype=np.uint64)
        t = raw["op_type"]
        sz = raw["size"]
        peer = raw["peer"].astype(np.int32)

        op_cls = np.where(t == OPTYPE_SEND, "comm_send", np.where(t == OPTYPE_RECV, "comm_recv", "compute"))
        ctype = np.where(t == OPTYPE_SEND, "SEND", np.where(t == OPTYPE_RECV, "RECV", ""))
        dur_ns = np.where(t == OPTYPE_CALC, sz, np.uint64(0))
        bytes_sz = np.where(t != OPTYPE_CALC, sz, np.uint64(0))
        src_r = np.where(t == OPTYPE_SEND, r, np.where(t == OPTYPE_RECV, peer, -1)).astype(np.int32)
        dst_r = np.where(t == OPTYPE_SEND, peer, np.where(t == OPTYPE_RECV, r, -1)).astype(np.int32)

        node_frames.append(
            pl.DataFrame(
                {
                    "rank": np.full(num_nodes, r, dtype=np.uint32),
                    "node_id": nids,
                    "name": op_cls,
                    "op_class": op_cls,
                    "comm_type": ctype,
                    "duration_ns": dur_ns,
                    "size_bytes": bytes_sz,
                    "src_rank": src_r,
                    "dst_rank": dst_r,
                    "tag": np.array(raw["tag"], dtype=np.uint32, copy=True),
                    "num_ops": np.zeros(num_nodes, dtype=np.uint64),
                },
                schema=NODES_SCHEMA,
            )
        )

        apdx_abs = nodes_abs + GOAL_BIN_NODE_DTYPE.itemsize * num_nodes
        apdx_bytes = (map_base + end_rel) - apdx_abs
        if apdx_bytes >= 4:
            apdx = np.frombuffer(mm, dtype="<u4", count=apdx_bytes // 4, offset=apdx_abs)
            for cnt_col, idx_col, kind in (
                ("num_deps", "deps_apdx", "requires"),
                ("num_startdeps", "startdeps_apdx", "irequires"),
            ):
                cnts = raw[cnt_col]
                active = np.flatnonzero(cnts > 0)
                if len(active) == 0:
                    continue
                src_rep = np.repeat(nids[active], cnts[active])
                starts = raw[idx_col][active]
                offsets = np.repeat(starts, cnts[active]) + (
                    np.arange(len(src_rep), dtype=np.uint32)
                    - np.repeat(np.cumsum(cnts[active], dtype=np.uint32) - cnts[active], cnts[active])
                )
                dst_ids = apdx[offsets].astype(np.uint64)
                edge_frames.append(
                    pl.DataFrame(
                        {
                            "rank": np.full(len(src_rep), r, dtype=np.uint32),
                            "src_id": src_rep,
                            "dst_id": dst_ids,
                            "dep_kind": [kind] * len(src_rep),
                        },
                        schema=EDGES_SCHEMA,
                    )
                )
    return node_frames, edge_frames


def _parse_goal_bin(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        node_frames, edge_frames = _decode_goal_mmap(mm, path)
    nodes_df = pl.concat(node_frames) if node_frames else pl.DataFrame(schema=NODES_SCHEMA)
    edges_df = pl.concat(edge_frames) if edge_frames else pl.DataFrame(schema=EDGES_SCHEMA)
    return nodes_df, edges_df


def _parse_goal_text(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    curr_rank = 0
    op_counter = 0
    label_map: dict[str, int] = {}
    in_block_comment = False

    with path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if in_block_comment:
                if "*/" in line:
                    line = line.split("*/", 1)[1].strip()
                    in_block_comment = False
                else:
                    continue
            if "/*" in line:
                before, after = line.split("/*", 1)
                if "*/" in after:
                    line = (before + " " + after.split("*/", 1)[1]).strip()
                else:
                    line = before.strip()
                    in_block_comment = True
            for ctok in ("//", "#"):
                if ctok in line:
                    line = line.split(ctok, 1)[0].strip()
            if not line or line in ("{", "}") or line.startswith("num_ranks"):
                if line == "}":
                    label_map.clear()
                continue

            m_rank = _RANK_HDR_RE.match(line)
            if m_rank:
                curr_rank = int(m_rank.group(1))
                op_counter = 0
                label_map.clear()
                continue

            m_dep = _DEP_RE.match(line)
            if m_dep:
                dst_lbl, kind, src_lbl = m_dep.group(1), m_dep.group(2), m_dep.group(3)
                if src_lbl in label_map and dst_lbl in label_map:
                    edge_rows.append(
                        {"rank": curr_rank, "src_id": label_map[src_lbl], "dst_id": label_map[dst_lbl], "dep_kind": kind}
                    )
                continue

            m_send = _SEND_RE.match(line)
            if m_send:
                nid = op_counter
                op_counter += 1
                lbl = m_send.group("lbl")
                if lbl:
                    label_map[lbl] = nid
                mul = _SIZE_MUL.get((m_send.group("unit") or "b").lower(), 1)
                sz = int(m_send.group("sz")) * mul
                peer = int(m_send.group("peer"))
                tag = int(m_send.group("tag") or 0)
                node_rows.append(
                    {
                        "rank": curr_rank,
                        "node_id": nid,
                        "name": lbl or "comm_send",
                        "op_class": "comm_send",
                        "comm_type": "SEND",
                        "duration_ns": 0,
                        "size_bytes": sz,
                        "src_rank": curr_rank,
                        "dst_rank": peer,
                        "tag": tag,
                        "num_ops": 0,
                    }
                )
                continue

            m_recv = _RECV_RE.match(line)
            if m_recv:
                nid = op_counter
                op_counter += 1
                lbl = m_recv.group("lbl")
                if lbl:
                    label_map[lbl] = nid
                mul = _SIZE_MUL.get((m_recv.group("unit") or "b").lower(), 1)
                sz = int(m_recv.group("sz")) * mul
                peer = int(m_recv.group("peer"))
                tag = int(m_recv.group("tag") or 0)
                node_rows.append(
                    {
                        "rank": curr_rank,
                        "node_id": nid,
                        "name": lbl or "comm_recv",
                        "op_class": "comm_recv",
                        "comm_type": "RECV",
                        "duration_ns": 0,
                        "size_bytes": sz,
                        "src_rank": peer,
                        "dst_rank": curr_rank,
                        "tag": tag,
                        "num_ops": 0,
                    }
                )
                continue

            m_calc = _CALC_RE.match(line)
            if m_calc:
                nid = op_counter
                op_counter += 1
                lbl = m_calc.group("lbl")
                if lbl:
                    label_map[lbl] = nid
                dur = int(m_calc.group("dur"))
                node_rows.append(
                    {
                        "rank": curr_rank,
                        "node_id": nid,
                        "name": lbl or "compute",
                        "op_class": "compute",
                        "comm_type": "",
                        "duration_ns": dur,
                        "size_bytes": 0,
                        "src_rank": -1,
                        "dst_rank": -1,
                        "tag": 0,
                        "num_ops": 0,
                    }
                )

    nodes_df = pl.DataFrame(node_rows, schema=NODES_SCHEMA) if node_rows else pl.DataFrame(schema=NODES_SCHEMA)
    edges_df = pl.DataFrame(edge_rows, schema=EDGES_SCHEMA) if edge_rows else pl.DataFrame(schema=EDGES_SCHEMA)
    return nodes_df, edges_df


@register_adapter("goal", version="1.0")
class GoalAdapter:
    format_name = "goal"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        suf = path.suffix.lower()
        if suf == ".goal":
            return True
        if suf == ".bin":
            return _is_goal_bin(path)
        return False

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        n_frames, e_frames = [], []
        for p in sorted(artifact.paths):
            ndf, edf = _parse_goal_bin(p) if _is_goal_bin(p) else _parse_goal_text(p)
            n_frames.append(ndf)
            e_frames.append(edf)

        nodes_df = pl.concat(n_frames) if n_frames else pl.DataFrame(schema=NODES_SCHEMA)
        edges_df = pl.concat(e_frames) if e_frames else pl.DataFrame(schema=EDGES_SCHEMA)

        traffic_df = (
            nodes_df.filter((pl.col("op_class") == "comm_send") & (pl.col("dst_rank") >= 0))
            .group_by([pl.col("rank").alias("src_rank"), pl.col("dst_rank").cast(pl.UInt32)])
            .agg(
                pl.len().cast(pl.UInt64).alias("msg_count"),
                pl.col("size_bytes").sum().cast(pl.UInt64).alias("total_bytes"),
            )
            .cast(TRAFFIC_SCHEMA)
            if nodes_df.height
            else pl.DataFrame(schema=TRAFFIC_SCHEMA)
        )

        np_out, ep_out, tp_out = out_dir / "nodes.parquet", out_dir / "edges.parquet", out_dir / "traffic.parquet"
        nodes_df.write_parquet(np_out, compression="zstd")
        edges_df.write_parquet(ep_out, compression="zstd")
        traffic_df.write_parquet(tp_out, compression="zstd")
        return {"nodes": np_out, "edges": ep_out, "traffic": tp_out}
