"""Chakra-ET (.et protobuf and host execution trace JSON) adapter.

Both formats are written as three tables: ``nodes`` (one row per execution
trace node), ``edges`` (data and control dependencies) and ``traffic`` (bytes
sent between each pair of ranks, with collectives expanded to the peer traffic
of their usual algorithm).
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent import futures
import json
import pathlib
import re
from typing import Any

import numpy as np
import polars as pl

from cta.adapters.chakra import schema
from cta.core import models
from cta.core import registry

_RANK_FILE_RE = re.compile(
    r"(?:rank[_.-]?|\.r?)(\d+)(?:\.[^.]+)+$", re.IGNORECASE
)

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
        "message_tag": pl.UInt32(),
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
        "message_count": pl.UInt64(),
        "total_bytes": pl.UInt64(),
    }
)

# Collectives whose ring algorithm sends (P-1)/P of the buffer to the next rank.
_RING_COLLECTIVES = (
    "ALL_GATHER",
    "REDUCE_SCATTER",
    "REDUCE_SCATTER_BLOCK",
    "BROADCAST",
    "REDUCE",
)

# Host execution trace op names: lowercase substrings -> (op_class, comm_type).
# The first matching entry wins; names matching none are compute.
_HOST_OP_PATTERNS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("allreduce", "all_reduce"), "comm_coll", "ALL_REDUCE"),
    (("allgather", "all_gather"), "comm_coll", "ALL_GATHER"),
    (("reducescatter", "reduce_scatter"), "comm_coll", "REDUCE_SCATTER"),
    (("alltoall", "all_to_all"), "comm_coll", "ALL_TO_ALL"),
    (("send",), "comm_send", "SEND"),
    (("recv",), "comm_recv", "RECV"),
)


def _infer_rank(path: pathlib.Path, fallback_rank: int) -> int:
    """Returns the rank in a file name such as ``trace.rank3.et``.

    Falls back to ``fallback_rank`` when the name carries no rank.
    """
    match = _RANK_FILE_RE.search(path.name)
    return int(match.group(1)) if match else fallback_rank


def _resolve_comm_type(
    raw_comm_type: object,
    op_class: str,
    coll_names_by_number: dict[int, str],
) -> str:
    """Returns the comm_type of a .et node from its ``comm_type`` attribute.

    Enum numbers map to their CollectiveCommType name, strings are upper-cased
    and send/recv nodes without the attribute become SEND/RECV.
    """
    if isinstance(raw_comm_type, int):
        return coll_names_by_number.get(raw_comm_type, f"COLL_{raw_comm_type}")
    if isinstance(raw_comm_type, str):
        return raw_comm_type.upper()
    if op_class == "comm_send":
        return "SEND"
    if op_class == "comm_recv":
        return "RECV"
    return ""


def _dependency_rows(
    rank: int, node_id: int, dep_ids: Iterable[Any], dep_kind: str
) -> list[dict[str, object]]:
    """Builds one ``edges`` row per dependency of a node."""
    return [
        {
            "rank": rank,
            "src_id": int(dep_id),
            "dst_id": node_id,
            "dep_kind": dep_kind,
        }
        for dep_id in dep_ids
    ]


def _parse_single_et(
    task: tuple[int, pathlib.Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Parses one .et file into ``nodes`` and ``edges`` rows.

    Args:
        task: The file's index in sorted path order, used as its rank when
            the file name carries none, and the file path.

    Returns:
        The node rows and the edge rows.
    """
    file_index, path = task
    rank = _infer_rank(path, file_index)
    _, nodes = schema.iter_et_messages(path)
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []

    coll_names_by_number = schema.coll_type_names()
    for node in nodes:
        attrs_by_name = {
            attr.name: schema.extract_attr_value(attr) for attr in node.attr
        }
        op_class = schema.NODE_TYPE_CLASS.get(int(node.type), "other")
        comm_type = _resolve_comm_type(
            attrs_by_name.get("comm_type"), op_class, coll_names_by_number
        )

        raw_size = attrs_by_name.get(
            "comm_size", attrs_by_name.get("tensor_size", 0)
        )
        size_bytes = int(raw_size or 0)
        if int(node.type) == schema.STORAGE_NODE and size_bytes == 0:
            size_bytes = schema.storage_size_bytes(node)
        default_src_rank = rank if op_class == "comm_send" else -1
        default_dst_rank = rank if op_class == "comm_recv" else -1
        src_rank = int(attrs_by_name.get("comm_src", default_src_rank))
        dst_rank = int(attrs_by_name.get("comm_dst", default_dst_rank))
        message_tag = int(attrs_by_name.get("comm_tag", 0) or 0)
        num_ops = int(attrs_by_name.get("num_ops", 0) or 0)
        duration_ns = int(node.duration_micros) * 1000

        node_id = int(node.id)
        node_rows.append(
            {
                "rank": rank,
                "node_id": node_id,
                "name": str(node.name),
                "op_class": op_class,
                "comm_type": comm_type,
                "duration_ns": duration_ns,
                "size_bytes": max(0, size_bytes),
                "src_rank": src_rank,
                "dst_rank": dst_rank,
                "message_tag": max(0, message_tag),
                "num_ops": max(0, num_ops),
            }
        )
        edge_rows.extend(
            _dependency_rows(rank, node_id, node.data_deps, "data")
        )
        edge_rows.extend(
            _dependency_rows(rank, node_id, node.ctrl_deps, "ctrl")
        )

    return node_rows, edge_rows


def _collective_traffic_rows(
    src_rank: int, comm_type: str, size_bytes: int, world_size: int
) -> list[dict[str, int]]:
    """Expands one rank's collective into the peer traffic it generates.

    Ring AllReduce sends ``2 * (P-1)/P * B`` bytes to the next rank over
    ``2 * (P-1)`` steps; ring AllGather, ReduceScatter, Broadcast and Reduce
    send ``(P-1)/P * B`` over ``P-1`` steps; AllToAll sends ``B/P`` to every
    other rank. Other collectives generate no rows.
    """
    next_rank = (src_rank + 1) % world_size
    if comm_type == "ALL_REDUCE":
        return [
            {
                "src_rank": src_rank,
                "dst_rank": next_rank,
                "message_count": 2 * (world_size - 1),
                "total_bytes": (2 * (world_size - 1) * size_bytes)
                // world_size,
            }
        ]
    if comm_type in _RING_COLLECTIVES:
        return [
            {
                "src_rank": src_rank,
                "dst_rank": next_rank,
                "message_count": world_size - 1,
                "total_bytes": ((world_size - 1) * size_bytes) // world_size,
            }
        ]
    if comm_type == "ALL_TO_ALL":
        bytes_per_peer = size_bytes // world_size
        return [
            {
                "src_rank": src_rank,
                "dst_rank": dst_rank,
                "message_count": 1,
                "total_bytes": bytes_per_peer,
            }
            for dst_rank in range(world_size)
            if dst_rank != src_rank
        ]
    return []


def _build_traffic_df(nodes: pl.DataFrame, world_size: int) -> pl.DataFrame:
    """Builds the ``traffic`` table (bytes per src/dst rank pair) of nodes.

    Point-to-point sends contribute their explicit (src_rank, dst_rank). With
    more than one rank, collectives also contribute the peer traffic of their
    ring (or AllToAll) algorithm.
    """
    if nodes.is_empty():
        return pl.DataFrame(schema=TRAFFIC_SCHEMA)

    p2p_traffic = (
        nodes.filter(
            (pl.col("op_class") == "comm_send") & (pl.col("dst_rank") >= 0)
        )
        .group_by(
            [
                pl.col("rank").alias("src_rank"),
                pl.col("dst_rank").cast(pl.UInt32),
            ]
        )
        .agg(
            pl.len().cast(pl.UInt64).alias("message_count"),
            pl.col("size_bytes").sum().cast(pl.UInt64).alias("total_bytes"),
        )
    )

    collectives = nodes.filter(
        (pl.col("op_class") == "comm_coll") & (pl.col("size_bytes") > 0)
    )
    if collectives.height == 0 or world_size <= 1:
        if p2p_traffic.height:
            return p2p_traffic.cast(TRAFFIC_SCHEMA)
        return pl.DataFrame(schema=TRAFFIC_SCHEMA)

    collective_rows: list[dict[str, int]] = []
    collective_ops = collectives.select("rank", "comm_type", "size_bytes")
    for row in collective_ops.iter_rows(named=True):
        collective_rows.extend(
            _collective_traffic_rows(
                int(row["rank"]),
                str(row["comm_type"]),
                int(row["size_bytes"]),
                world_size,
            )
        )

    collective_traffic = (
        pl.DataFrame(collective_rows, schema=TRAFFIC_SCHEMA)
        if collective_rows
        else pl.DataFrame(schema=TRAFFIC_SCHEMA)
    )
    return (
        pl.concat([p2p_traffic.cast(TRAFFIC_SCHEMA), collective_traffic])
        .group_by(["src_rank", "dst_rank"])
        .agg(pl.col("message_count").sum(), pl.col("total_bytes").sum())
    )


def _rows_to_frame(
    rows: list[dict[str, object]], frame_schema: pl.Schema
) -> pl.DataFrame:
    """Builds a frame from rows, or an empty frame with the given schema."""
    if rows:
        return pl.DataFrame(rows, schema=frame_schema)
    return pl.DataFrame(schema=frame_schema)


def _write_tables(
    artifact: models.TraceArtifact,
    node_rows: list[dict[str, object]],
    edge_rows: list[dict[str, object]],
    out_dir: pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Writes the ``nodes``, ``edges`` and ``traffic`` Parquet tables.

    The world size used for collective traffic is the larger of the
    artifact's world size and the highest rank seen plus one.
    """
    nodes = _rows_to_frame(node_rows, NODES_SCHEMA)
    edges = _rows_to_frame(edge_rows, EDGES_SCHEMA)
    ranks_seen = (
        int(np.max(nodes["rank"].to_numpy())) + 1 if nodes.height else 1
    )
    world_size = max(artifact.world_size, ranks_seen)
    traffic = _build_traffic_df(nodes, world_size)

    nodes_path = out_dir / "nodes.parquet"
    edges_path = out_dir / "edges.parquet"
    traffic_path = out_dir / "traffic.parquet"
    nodes.write_parquet(nodes_path, compression="zstd")
    edges.write_parquet(edges_path, compression="zstd")
    traffic.write_parquet(traffic_path, compression="zstd")
    return {"nodes": nodes_path, "edges": edges_path, "traffic": traffic_path}


@registry.register_adapter("chakra_et", version="1.1")
class ChakraEtAdapter:
    """Reads Chakra .et protobuf execution traces (one file per rank).

    Attributes:
        format_name: The registered trace format name, ``"chakra_et"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "chakra_et"
    version = "1.1"

    def detect(self, path: pathlib.Path) -> bool:
        """Checks whether a file is a Chakra .et trace.

        Args:
            path: The candidate file.

        Returns:
            True if the file has the .et suffix.
        """
        return path.suffix.lower() == ".et"

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the artifact's .et files to nodes/edges/traffic tables.

        Files are parsed in parallel when there is more than one. Each file's
        rank comes from its name, else from its index in sorted path order.

        Args:
            artifact: The Chakra .et artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"nodes"``, ``"edges"`` and ``"traffic"`` mapped
            to their Parquet paths.
        """
        tasks = list(enumerate(sorted(artifact.paths)))
        all_node_rows: list[dict[str, object]] = []
        all_edge_rows: list[dict[str, object]] = []

        if len(tasks) > 1:
            with futures.ThreadPoolExecutor(
                max_workers=min(8, len(tasks))
            ) as pool:
                for node_rows, edge_rows in pool.map(_parse_single_et, tasks):
                    all_node_rows.extend(node_rows)
                    all_edge_rows.extend(edge_rows)
        else:
            node_rows, edge_rows = _parse_single_et(tasks[0])
            all_node_rows.extend(node_rows)
            all_edge_rows.extend(edge_rows)

        return _write_tables(artifact, all_node_rows, all_edge_rows, out_dir)


def _classify_host_op(name: str) -> tuple[str, str]:
    """Returns the (op_class, comm_type) of a host execution trace op name."""
    lower_name = name.lower()
    for patterns, op_class, comm_type in _HOST_OP_PATTERNS:
        if any(pattern in lower_name for pattern in patterns):
            return op_class, comm_type
    return "compute", ""


def _host_node_row(node: dict[str, Any], rank: int) -> dict[str, object]:
    """Builds the ``nodes`` row of one host execution trace JSON node."""
    name = str(node.get("name", ""))
    op_class, comm_type = _classify_host_op(name)
    duration_us = float(node.get("dur_us", node.get("duration_micros", 1)))
    duration_ns = int(duration_us * 1000)
    size_bytes = int(node.get("comm_bytes", node.get("tensor_bytes", 0)) or 0)
    default_src_rank = rank if op_class == "comm_send" else -1
    default_dst_rank = rank if op_class == "comm_recv" else -1
    src_rank = int(node.get("comm_src", node.get("src_rank", default_src_rank)))
    dst_rank = int(node.get("comm_dst", node.get("dst_rank", default_dst_rank)))
    message_tag = int(node.get("comm_tag", node.get("tag", 0)) or 0)
    return {
        "rank": rank,
        "node_id": int(node["id"]),
        "name": name,
        "op_class": op_class,
        "comm_type": comm_type,
        "duration_ns": max(0, duration_ns),
        "size_bytes": max(0, size_bytes),
        "src_rank": src_rank,
        "dst_rank": dst_rank,
        "message_tag": max(0, message_tag),
        "num_ops": 0,
    }


@registry.register_adapter("chakra_host_et", version="1.1")
class ChakraHostEtAdapter:
    """Reads host execution trace JSON files.

    Op classes and collective types are inferred from the node names.

    Attributes:
        format_name: The registered trace format name, ``"chakra_host_et"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "chakra_host_et"
    version = "1.1"

    def detect(self, path: pathlib.Path) -> bool:
        """Checks whether a file is a host execution trace JSON file.

        Args:
            path: The candidate file.

        Returns:
            True for a .json file whose name mentions chakra, host_et or
            execution_trace, or whose first 4 KiB hold the ``ctrl_deps``,
            ``nodes`` and ``schema`` keys.
        """
        if path.suffix.lower() != ".json":
            return False
        lower_name = path.name.lower()
        if (
            "chakra" in lower_name
            or "host_et" in lower_name
            or "execution_trace" in lower_name
        ):
            return True
        try:
            with path.open("rb") as fh:
                head = fh.read(4096).decode("utf-8", errors="ignore")
            return (
                '"ctrl_deps"' in head
                and '"nodes"' in head
                and '"schema"' in head
            )
        except OSError:
            return False

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the artifact's host ET files to nodes/edges/traffic tables.

        Each file's rank comes from its name, else from its index in sorted
        path order.

        Args:
            artifact: The host execution trace artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"nodes"``, ``"edges"`` and ``"traffic"`` mapped
            to their Parquet paths.
        """
        node_rows: list[dict[str, object]] = []
        edge_rows: list[dict[str, object]] = []
        for file_index, path in enumerate(sorted(artifact.paths)):
            rank = _infer_rank(path, file_index)
            trace = json.loads(path.read_text())
            for node in trace.get("nodes", []):
                node_id = int(node["id"])
                node_rows.append(_host_node_row(node, rank))
                edge_rows.extend(
                    _dependency_rows(
                        rank, node_id, node.get("ctrl_deps", []), "ctrl"
                    )
                )
                edge_rows.extend(
                    _dependency_rows(
                        rank, node_id, node.get("data_deps", []), "data"
                    )
                )

        return _write_tables(artifact, node_rows, edge_rows, out_dir)
