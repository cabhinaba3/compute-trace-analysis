"""Chakra-ET (.et protobuf and host execution trace JSON) adapter.

Both formats are written as three tables: ``nodes`` (one row per execution
trace node), ``edges`` (data and control dependencies) and ``traffic`` (bytes
sent between each pair of ranks, with collectives expanded to the peer traffic
of their usual algorithm).
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import pathlib
from typing import Any

import numpy as np
import polars as pl

from cta import _native as native
from cta import models
from cta import plugins
from cta.ingest import discovery

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


def _collective_traffic(nodes: pl.DataFrame, world_size: int) -> pl.DataFrame:
    """Expands every collective into the peer traffic of its usual algorithm.

    Ring AllReduce sends ``2 * (P-1)/P * B`` bytes to the next rank over
    ``2 * (P-1)`` steps; ring AllGather, ReduceScatter, Broadcast and Reduce
    send ``(P-1)/P * B`` over ``P-1`` steps; AllToAll sends ``B/P`` to every
    other rank. Other collectives generate no rows.

    Collectives are counted per (rank, type, size) first, so the work does not
    grow with the number of collectives, only with the number of distinct
    sizes. The per-collective byte counts are rounded down before they are
    multiplied by the count, as when each collective is expanded on its own.

    Args:
        nodes: The ``nodes`` table.
        world_size: The number of ranks P.

    Returns:
        Traffic rows per (src_rank, dst_rank), not yet summed.
    """
    groups = (
        nodes.filter(
            (pl.col("op_class") == "comm_coll") & (pl.col("size_bytes") > 0)
        )
        .group_by("rank", "comm_type", "size_bytes")
        .len()
        .select(
            pl.col("rank").alias("src_rank"),
            "comm_type",
            pl.col("size_bytes"),
            pl.col("len").cast(pl.UInt64).alias("count"),
        )
    )
    size = pl.col("size_bytes")
    count = pl.col("count")
    ranks = pl.lit(world_size, dtype=pl.UInt64)
    next_rank = ((pl.col("src_rank") + 1) % world_size).cast(pl.UInt32)

    all_reduce = groups.filter(pl.col("comm_type") == "ALL_REDUCE").select(
        "src_rank",
        next_rank.alias("dst_rank"),
        (count * (2 * (world_size - 1))).alias("message_count"),
        (count * ((size * (2 * (world_size - 1))) // ranks)).alias(
            "total_bytes"
        ),
    )
    ring = groups.filter(pl.col("comm_type").is_in(_RING_COLLECTIVES)).select(
        "src_rank",
        next_rank.alias("dst_rank"),
        (count * (world_size - 1)).alias("message_count"),
        (count * ((size * (world_size - 1)) // ranks)).alias("total_bytes"),
    )
    all_to_all = (
        groups.filter(pl.col("comm_type") == "ALL_TO_ALL")
        .join(
            pl.DataFrame(
                {"dst_rank": pl.int_range(world_size, eager=True)}
            ).cast({"dst_rank": pl.UInt32}),
            how="cross",
        )
        .filter(pl.col("dst_rank") != pl.col("src_rank"))
        .select(
            "src_rank",
            "dst_rank",
            count.alias("message_count"),
            (count * (size // ranks)).alias("total_bytes"),
        )
    )
    return pl.concat(
        [all_reduce, ring, all_to_all], how="vertical_relaxed"
    ).cast(TRAFFIC_SCHEMA)


def _build_traffic_df(nodes: pl.DataFrame, world_size: int) -> pl.DataFrame:
    """Builds the ``traffic`` table (bytes per src/dst rank pair) of nodes.

    Point-to-point sends contribute their explicit (src_rank, dst_rank). With
    more than one rank, collectives also contribute the peer traffic of their
    ring (or AllToAll) algorithm. Rows are sorted by (src_rank, dst_rank).
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
        .cast(TRAFFIC_SCHEMA)
    )
    if world_size <= 1:
        return p2p_traffic.sort("src_rank", "dst_rank")
    return (
        pl.concat(
            [p2p_traffic, _collective_traffic(nodes, world_size)],
            how="vertical_relaxed",
        )
        .group_by("src_rank", "dst_rank")
        .agg(pl.col("message_count").sum(), pl.col("total_bytes").sum())
        .sort("src_rank", "dst_rank")
        .cast(TRAFFIC_SCHEMA)
    )


def _rows_to_frame(
    rows: list[dict[str, object]], frame_schema: pl.Schema
) -> pl.DataFrame:
    """Builds a frame from rows, or an empty frame with the given schema."""
    return pl.DataFrame(rows, schema=frame_schema)


def _write_tables(
    artifact: models.TraceArtifact,
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    out_dir: pathlib.Path,
) -> dict[str, pathlib.Path]:
    """Writes the ``nodes``, ``edges`` and ``traffic`` Parquet tables.

    The world size used for collective traffic is the larger of the
    artifact's world size and the highest rank seen plus one.
    """
    ranks_seen = (
        int(np.max(nodes["rank"].to_numpy())) + 1 if nodes.height else 1
    )
    world_size = max(artifact.world_size, ranks_seen)
    traffic = _build_traffic_df(nodes, world_size)

    return discovery.write_tables(
        out_dir,
        {
            "nodes": nodes,
            "edges": edges,
            "traffic": traffic,
        },
    )


# Codes the native reader gives to the op classes and dependency kinds.
_OP_CLASSES = {
    0: "other",
    1: "memory",
    2: "compute",
    3: "comm_send",
    4: "comm_recv",
    5: "comm_coll",
    6: "storage",
    7: "data_op",
}
_DEPENDENCY_KINDS = {0: "data", 1: "ctrl"}


def _decode(codes: np.ndarray, dictionary: list[str]) -> pl.Series:
    """Returns the strings that dictionary codes stand for."""
    return pl.Series(dictionary, dtype=pl.String).gather(codes)


@plugins.register_adapter("chakra_et", version="1.1")
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

        Files are parsed in parallel, one task per file. Each file's rank
        comes from its name, else from its index in sorted path order.

        Args:
            artifact: The Chakra .et artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"nodes"``, ``"edges"`` and ``"traffic"`` mapped
            to their Parquet paths.
        """
        paths = sorted(artifact.paths)
        ranks = [
            discovery.infer_rank(path, index)
            for index, path in enumerate(paths)
        ]
        columns = native.read_chakra_et(
            [str(path) for path in paths],
            ranks,
        )
        nodes = pl.DataFrame(
            {
                "rank": columns["rank"],
                "node_id": columns["node_id"],
                "name": _decode(columns["name"], columns["names"]),
                "op_class": pl.Series(columns["op_class"]).replace_strict(
                    _OP_CLASSES, return_dtype=pl.String
                ),
                "comm_type": _decode(
                    columns["comm_type"], columns["comm_types"]
                ),
                "duration_ns": columns["duration_ns"],
                "size_bytes": columns["size_bytes"],
                "src_rank": columns["src_rank"],
                "dst_rank": columns["dst_rank"],
                "message_tag": columns["message_tag"],
                "num_ops": columns["num_ops"],
            },
            schema=NODES_SCHEMA,
        )
        edges = pl.DataFrame(
            {
                "rank": columns["edge_rank"],
                "src_id": columns["edge_src"],
                "dst_id": columns["edge_dst"],
                "dep_kind": pl.Series(columns["edge_kind"]).replace_strict(
                    _DEPENDENCY_KINDS, return_dtype=pl.String
                ),
            },
            schema=EDGES_SCHEMA,
        )
        return _write_tables(artifact, nodes, edges, out_dir)


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


@plugins.register_adapter("chakra_host_et", version="1.1")
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
        head = discovery.peek_text(path, 4096)
        return (
            '"ctrl_deps"' in head and '"nodes"' in head and '"schema"' in head
        )

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
            rank = discovery.infer_rank(path, file_index)
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

        return _write_tables(
            artifact,
            _rows_to_frame(node_rows, NODES_SCHEMA),
            _rows_to_frame(edge_rows, EDGES_SCHEMA),
            out_dir,
        )
