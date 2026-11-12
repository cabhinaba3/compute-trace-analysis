"""Chakra-ET (.et protobuf and host execution trace JSON) adapter."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from cta.adapters.chakra.schema import (
    NODE_TYPE_CLASS,
    STORAGE_NODE,
    coll_type_names,
    extract_attr_value,
    iter_et_messages,
    storage_size_bytes,
)
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

_RANK_FILE_RE = re.compile(r"(?:rank[_.-]?|\.r?)(\d+)(?:\.[^.]+)+$", re.IGNORECASE)

NODES_SCHEMA = pl.Schema(
    {
        "rank": pl.UInt32(),
        "node_id": pl.UInt64(),
        "name": pl.String(),
        "op_class": pl.String(),
        "comm_type": pl.String(),
        "duration_ns": pl.UInt64(),
        "size_bytes": pl.UInt64(),
        "src_rank": pl.Int32(),
        "dst_rank": pl.Int32(),
        "tag": pl.UInt32(),
        "num_ops": pl.UInt64(),
    }
)

EDGES_SCHEMA = pl.Schema(
    {
        "rank": pl.UInt32(),
        "src_id": pl.UInt64(),
        "dst_id": pl.UInt64(),
        "dep_kind": pl.String(),
    }
)

TRAFFIC_SCHEMA = pl.Schema(
    {
        "src_rank": pl.UInt32(),
        "dst_rank": pl.UInt32(),
        "msg_count": pl.UInt64(),
        "total_bytes": pl.UInt64(),
    }
)


def _infer_rank(path: Path, fallback_idx: int) -> int:
    m = _RANK_FILE_RE.search(path.name)
    return int(m.group(1)) if m else fallback_idx


def _parse_single_et(args: tuple[int, Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    idx, path = args
    rank = _infer_rank(path, idx)
    _, nodes_iter = iter_et_messages(path)
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []

    coll_names = coll_type_names()
    for n in nodes_iter:
        attrs = {a.name: extract_attr_value(a) for a in n.attr}
        op_class = NODE_TYPE_CLASS.get(int(n.type), "other")
        raw_ct = attrs.get("comm_type")
        if isinstance(raw_ct, int):
            comm_type = coll_names.get(raw_ct, f"COLL_{raw_ct}")
        elif isinstance(raw_ct, str):
            comm_type = raw_ct.upper()
        elif op_class == "comm_send":
            comm_type = "SEND"
        elif op_class == "comm_recv":
            comm_type = "RECV"
        else:
            comm_type = ""

        size_bytes = int(attrs.get("comm_size", attrs.get("tensor_size", 0)) or 0)
        if int(n.type) == STORAGE_NODE and size_bytes == 0:
            size_bytes = storage_size_bytes(n)
        src_r = int(attrs.get("comm_src", rank if op_class == "comm_send" else -1))
        dst_r = int(attrs.get("comm_dst", rank if op_class == "comm_recv" else -1))
        tag = int(attrs.get("comm_tag", 0) or 0)
        num_ops = int(attrs.get("num_ops", 0) or 0)
        dur_ns = int(n.duration_micros) * 1000

        nid = int(n.id)
        node_rows.append(
            {
                "rank": rank,
                "node_id": nid,
                "name": str(n.name),
                "op_class": op_class,
                "comm_type": comm_type,
                "duration_ns": dur_ns,
                "size_bytes": max(0, size_bytes),
                "src_rank": src_r,
                "dst_rank": dst_r,
                "tag": max(0, tag),
                "num_ops": max(0, num_ops),
            }
        )
        for dep in n.data_deps:
            edge_rows.append({"rank": rank, "src_id": int(dep), "dst_id": nid, "dep_kind": "data"})
        for dep in n.ctrl_deps:
            edge_rows.append({"rank": rank, "src_id": int(dep), "dst_id": nid, "dep_kind": "ctrl"})

    return node_rows, edge_rows


def _build_traffic_df(nodes_df: pl.DataFrame, world_size: int) -> pl.DataFrame:
    if nodes_df.is_empty():
        return pl.DataFrame(schema=TRAFFIC_SCHEMA)

    # P2P sends carry explicit (src_rank, dst_rank)
    p2p = (
        nodes_df.filter((pl.col("op_class") == "comm_send") & (pl.col("dst_rank") >= 0))
        .group_by([pl.col("rank").alias("src_rank"), pl.col("dst_rank").cast(pl.UInt32)])
        .agg(pl.len().cast(pl.UInt64).alias("msg_count"), pl.col("size_bytes").sum().cast(pl.UInt64).alias("total_bytes"))
    )

    # For collectives with world_size > 1, attribute Ring peer traffic (r -> (r+1)%P) or AllToAll uniform peer bytes
    colls = nodes_df.filter((pl.col("op_class") == "comm_coll") & (pl.col("size_bytes") > 0))
    if colls.height == 0 or world_size <= 1:
        return p2p.cast(TRAFFIC_SCHEMA) if p2p.height else pl.DataFrame(schema=TRAFFIC_SCHEMA)

    p = world_size
    coll_rows: list[dict[str, int]] = []
    for row in colls.select("rank", "comm_type", "size_bytes").iter_rows(named=True):
        r = int(row["rank"])
        ct = str(row["comm_type"])
        sz = int(row["size_bytes"])
        nxt = (r + 1) % p
        if ct == "ALL_REDUCE":
            # Ring AllReduce sends 2 * (P-1)/P * B total bytes to next neighbor across 2*(P-1) steps
            coll_rows.append({"src_rank": r, "dst_rank": nxt, "msg_count": 2 * (p - 1), "total_bytes": (2 * (p - 1) * sz) // p})
        elif ct in ("ALL_GATHER", "REDUCE_SCATTER", "REDUCE_SCATTER_BLOCK", "BROADCAST", "REDUCE"):
            coll_rows.append({"src_rank": r, "dst_rank": nxt, "msg_count": p - 1, "total_bytes": ((p - 1) * sz) // p})
        elif ct == "ALL_TO_ALL":
            per_peer = sz // p
            for dst in range(p):
                if dst != r:
                    coll_rows.append({"src_rank": r, "dst_rank": dst, "msg_count": 1, "total_bytes": per_peer})

    coll_df = pl.DataFrame(coll_rows, schema=TRAFFIC_SCHEMA) if coll_rows else pl.DataFrame(schema=TRAFFIC_SCHEMA)
    return (
        pl.concat([p2p.cast(TRAFFIC_SCHEMA), coll_df])
        .group_by(["src_rank", "dst_rank"])
        .agg(pl.col("msg_count").sum(), pl.col("total_bytes").sum())
    )


@register_adapter("chakra_et", version="1.0")
class ChakraEtAdapter:
    format_name = "chakra_et"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        return path.suffix.lower() == ".et"

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        tasks = list(enumerate(sorted(artifact.paths)))
        all_nodes: list[dict[str, object]] = []
        all_edges: list[dict[str, object]] = []

        if len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
                for nrows, erows in pool.map(_parse_single_et, tasks):
                    all_nodes.extend(nrows)
                    all_edges.extend(erows)
        else:
            nrows, erows = _parse_single_et(tasks[0])
            all_nodes.extend(nrows)
            all_edges.extend(erows)

        nodes_df = pl.DataFrame(all_nodes, schema=NODES_SCHEMA) if all_nodes else pl.DataFrame(schema=NODES_SCHEMA)
        edges_df = pl.DataFrame(all_edges, schema=EDGES_SCHEMA) if all_edges else pl.DataFrame(schema=EDGES_SCHEMA)
        wsize = max(artifact.world_size, int(np.max(nodes_df["rank"].to_numpy())) + 1 if nodes_df.height else 1)
        traffic_df = _build_traffic_df(nodes_df, wsize)

        np_out = out_dir / "nodes.parquet"
        ep_out = out_dir / "edges.parquet"
        tp_out = out_dir / "traffic.parquet"
        nodes_df.write_parquet(np_out, compression="zstd")
        edges_df.write_parquet(ep_out, compression="zstd")
        traffic_df.write_parquet(tp_out, compression="zstd")
        return {"nodes": np_out, "edges": ep_out, "traffic": tp_out}


@register_adapter("chakra_host_et", version="1.0")
class ChakraHostEtAdapter:
    format_name = "chakra_host_et"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        if "chakra" in path.name.lower() or "host_et" in path.name.lower() or "execution_trace" in path.name.lower():
            return True
        try:
            with path.open("rb") as fh:
                head = fh.read(4096).decode("utf-8", errors="ignore")
            return '"ctrl_deps"' in head and '"nodes"' in head and '"schema"' in head
        except OSError:
            return False

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        all_nodes: list[dict[str, object]] = []
        all_edges: list[dict[str, object]] = []
        for idx, path in enumerate(sorted(artifact.paths)):
            rank = _infer_rank(path, idx)
            doc = json.loads(path.read_text())
            for n in doc.get("nodes", []):
                nid = int(n["id"])
                name = str(n.get("name", ""))
                low = name.lower()
                if "allreduce" in low or "all_reduce" in low:
                    op_class, comm_type = "comm_coll", "ALL_REDUCE"
                elif "allgather" in low or "all_gather" in low:
                    op_class, comm_type = "comm_coll", "ALL_GATHER"
                elif "reducescatter" in low or "reduce_scatter" in low:
                    op_class, comm_type = "comm_coll", "REDUCE_SCATTER"
                elif "alltoall" in low or "all_to_all" in low:
                    op_class, comm_type = "comm_coll", "ALL_TO_ALL"
                elif "send" in low:
                    op_class, comm_type = "comm_send", "SEND"
                elif "recv" in low:
                    op_class, comm_type = "comm_recv", "RECV"
                else:
                    op_class, comm_type = "compute", ""

                dur_ns = int(float(n.get("dur_us", n.get("duration_micros", 1))) * 1000)
                sz = int(n.get("comm_bytes", n.get("tensor_bytes", 0)) or 0)
                all_nodes.append(
                    {
                        "rank": rank,
                        "node_id": nid,
                        "name": name,
                        "op_class": op_class,
                        "comm_type": comm_type,
                        "duration_ns": max(0, dur_ns),
                        "size_bytes": max(0, sz),
                        "src_rank": rank if op_class == "comm_send" else -1,
                        "dst_rank": rank if op_class == "comm_recv" else -1,
                        "tag": 0,
                        "num_ops": 0,
                    }
                )
                for dep in n.get("ctrl_deps", []):
                    all_edges.append({"rank": rank, "src_id": int(dep), "dst_id": nid, "dep_kind": "ctrl"})
                for dep in n.get("data_deps", []):
                    all_edges.append({"rank": rank, "src_id": int(dep), "dst_id": nid, "dep_kind": "data"})

        nodes_df = pl.DataFrame(all_nodes, schema=NODES_SCHEMA) if all_nodes else pl.DataFrame(schema=NODES_SCHEMA)
        edges_df = pl.DataFrame(all_edges, schema=EDGES_SCHEMA) if all_edges else pl.DataFrame(schema=EDGES_SCHEMA)
        wsize = max(artifact.world_size, int(np.max(nodes_df["rank"].to_numpy())) + 1 if nodes_df.height else 1)
        traffic_df = _build_traffic_df(nodes_df, wsize)

        np_out, ep_out, tp_out = out_dir / "nodes.parquet", out_dir / "edges.parquet", out_dir / "traffic.parquet"
        nodes_df.write_parquet(np_out, compression="zstd")
        edges_df.write_parquet(ep_out, compression="zstd")
        traffic_df.write_parquet(tp_out, compression="zstd")
        return {"nodes": np_out, "edges": ep_out, "traffic": tp_out}
