"""GOAL (.goal text and SPCL txt2bin .bin binary) schedule adapter.

Both encodings are written as the Chakra adapter's ``nodes``, ``edges`` and
``traffic`` tables, so the two formats can be compared with each other and
with Chakra traces.
"""

from __future__ import annotations

import mmap
import pathlib
import struct

import numpy as np
import polars as pl
import pyarrow as pa

from cta import _native as native
from cta import models
from cta import plugins
from cta.adapters import common
from cta.adapters.chakra import reader

GOAL_MAGIC_COOKIE = 4223
GOAL_MAGIC_INVALID = 2342
OPTYPE_SEND = 1
OPTYPE_RECV = 2
OPTYPE_CALC = 3

# The exact 39-byte packed node struct from SPCL LogGOPSim's Parser.hpp.
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


def _is_goal_bin(path: pathlib.Path) -> bool:
    """Checks whether a file starts with a GOAL binary magic cookie."""
    try:
        # The 8-byte magic cookie plus the 6-byte rank map header.
        if path.stat().st_size < 14:
            return False
        with path.open("rb") as fh:
            (cookie,) = struct.unpack("<Q", fh.read(8))
        return cookie in (GOAL_MAGIC_COOKIE, GOAL_MAGIC_INVALID)
    except OSError:
        return False


def _bin_nodes_frame(
    records: np.ndarray, node_ids: np.ndarray, rank: int
) -> pl.DataFrame:
    """Builds the ``nodes`` rows of one rank's packed GOAL node records."""
    num_nodes = len(records)
    op_types = records["op_type"]
    sizes = records["size"]
    peers = records["peer"].astype(np.int32)
    is_send = op_types == OPTYPE_SEND
    is_recv = op_types == OPTYPE_RECV
    is_calc = op_types == OPTYPE_CALC

    op_classes = np.where(
        is_send, "comm_send", np.where(is_recv, "comm_recv", "compute")
    )
    comm_types = np.where(is_send, "SEND", np.where(is_recv, "RECV", ""))
    # A calc node stores its duration in the size field.
    durations_ns = np.where(is_calc, sizes, np.uint64(0))
    sizes_bytes = np.where(~is_calc, sizes, np.uint64(0))
    src_ranks = np.where(is_send, rank, np.where(is_recv, peers, -1))
    dst_ranks = np.where(is_send, peers, np.where(is_recv, rank, -1))

    return pl.DataFrame(
        {
            "rank": np.full(num_nodes, rank, dtype=np.uint32),
            "node_id": node_ids,
            "name": op_classes,
            "op_class": op_classes,
            "comm_type": comm_types,
            "duration_ns": durations_ns,
            "size_bytes": sizes_bytes,
            "src_rank": src_ranks.astype(np.int32),
            "dst_rank": dst_ranks.astype(np.int32),
            "message_tag": np.array(records["tag"], dtype=np.uint32, copy=True),
            "num_ops": np.zeros(num_nodes, dtype=np.uint64),
            "pg_name": [""] * num_nodes,
            "cpu": records["proc"].astype(np.uint16),
            "nic": records["nic"].astype(np.uint16),
        },
        schema=reader.NODES_SCHEMA,
    )


def _bin_edges_frames(
    records: np.ndarray,
    node_ids: np.ndarray,
    appendix: np.ndarray,
    rank: int,
) -> list[pl.DataFrame]:
    """Builds the ``edges`` rows of one rank's packed GOAL node records.

    Each node lists its ``requires`` and ``irequires`` dependencies as a count
    and a start index into the rank's u32 dependency appendix.

    Returns:
        One frame per dependency kind that has at least one edge.
    """
    frames: list[pl.DataFrame] = []
    for count_field, start_field, dep_kind in (
        ("num_deps", "deps_apdx", "requires"),
        ("num_startdeps", "startdeps_apdx", "irequires"),
    ):
        counts = records[count_field]
        nodes_with_deps = np.flatnonzero(counts > 0)
        if len(nodes_with_deps) == 0:
            continue
        dep_counts = counts[nodes_with_deps]
        src_ids = np.repeat(node_ids[nodes_with_deps], dep_counts)
        # A node's dependencies are appendix[start : start + count]: repeat
        # each start count times and add each edge's position within its
        # node (0, 1, ..., count - 1).
        first_edge_of_node = np.cumsum(dep_counts, dtype=np.uint32) - dep_counts
        edge_indices = np.arange(len(src_ids), dtype=np.uint32)
        position_in_node = edge_indices - np.repeat(
            first_edge_of_node, dep_counts
        )
        appendix_starts = records[start_field][nodes_with_deps]
        appendix_indices = (
            np.repeat(appendix_starts, dep_counts) + position_in_node
        )
        dst_ids = appendix[appendix_indices].astype(np.uint64)
        frames.append(
            pl.DataFrame(
                {
                    "rank": np.full(len(src_ids), rank, dtype=np.uint32),
                    "src_id": src_ids,
                    "dst_id": dst_ids,
                    "dep_kind": [dep_kind] * len(src_ids),
                },
                schema=reader.EDGES_SCHEMA,
            )
        )
    return frames


def _decode_goal_mmap(
    mapped_file: mmap.mmap, path: pathlib.Path
) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    """Decodes every rank's schedule from a memory-mapped GOAL binary.

    The file holds an 8-byte magic cookie and then the rank map: a u32 rank
    count, two u8 fields and a [start, end) u64 offset pair per rank, relative
    to the map. A rank's block holds its u32 node and root counts, 4 bytes
    per root, the packed node records and the u32 dependency appendix.

    Args:
        mapped_file: The memory-mapped file contents.
        path: The file's path, for the error message.

    Returns:
        The per-rank ``nodes`` frames and the per-rank, per-kind ``edges``
        frames.

    Raises:
        ValueError: If the file does not start with a GOAL magic cookie.
    """
    node_frames: list[pl.DataFrame] = []
    edge_frames: list[pl.DataFrame] = []
    (cookie,) = struct.unpack_from("<Q", mapped_file, 0)
    if cookie not in (GOAL_MAGIC_COOKIE, GOAL_MAGIC_INVALID):
        raise ValueError(f"Invalid GOAL binary magic cookie {cookie} in {path}")
    map_offset = 8
    num_ranks, _, _ = struct.unpack_from("<IBB", mapped_file, map_offset)
    rank_bounds = np.frombuffer(
        mapped_file, dtype="<u8", count=2 * num_ranks, offset=map_offset + 6
    )

    for rank in range(num_ranks):
        start_rel = int(rank_bounds[2 * rank])
        end_rel = int(rank_bounds[2 * rank + 1])
        if end_rel <= start_rel:
            continue
        rank_offset = map_offset + start_rel
        num_nodes, num_roots = struct.unpack_from(
            "<II", mapped_file, rank_offset
        )
        if num_nodes == 0:
            continue
        nodes_offset = rank_offset + 8 + 4 * num_roots
        records = np.frombuffer(
            mapped_file,
            dtype=GOAL_BIN_NODE_DTYPE,
            count=num_nodes,
            offset=nodes_offset,
        )
        node_ids = np.arange(num_nodes, dtype=np.uint64)
        node_frames.append(_bin_nodes_frame(records, node_ids, rank))

        appendix_offset = (
            nodes_offset + GOAL_BIN_NODE_DTYPE.itemsize * num_nodes
        )
        appendix_bytes = (map_offset + end_rel) - appendix_offset
        if appendix_bytes >= 4:
            appendix = np.frombuffer(
                mapped_file,
                dtype="<u4",
                count=appendix_bytes // 4,
                offset=appendix_offset,
            )
            edge_frames.extend(
                _bin_edges_frames(records, node_ids, appendix, rank)
            )
    return node_frames, edge_frames


def _parse_goal_bin(path: pathlib.Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parses a GOAL binary schedule into ``nodes`` and ``edges`` frames."""
    columns = native.read_goal_bin(str(path))
    return common.chunk_to_graph_frames(
        columns, reader.NODES_SCHEMA, reader.EDGES_SCHEMA
    )


# Codes the native text reader gives to op classes and dependency kinds.
_OP_CLASSES = {0: "compute", 1: "comm_send", 2: "comm_recv"}
_COMM_TYPES = {0: "", 1: "SEND", 2: "RECV"}
_DEPENDENCY_KINDS = {0: "requires", 1: "irequires"}


def _names(name_end: np.ndarray, name_bytes: np.ndarray) -> pl.Series:
    """Returns the op names held as end offsets into one buffer of bytes."""
    if len(name_end) == 0:
        return pl.Series([], dtype=pl.String)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), name_end.astype(np.int64))
    )
    array = pa.Array.from_buffers(
        pa.large_string(),
        len(name_end),
        [None, pa.py_buffer(offsets), pa.py_buffer(name_bytes)],
    )
    return pl.Series(pl.from_arrow(array))


def parse_goal_text(path: pathlib.Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parses a GOAL text schedule into ``nodes`` and ``edges`` frames.

    Nodes are numbered per rank in file order. Labels are scoped to their rank
    block, and dependencies naming an unknown label are dropped. The file is
    cut at rank headers and the pieces are parsed in parallel.

    Args:
        path: The .goal file to read.

    Returns:
        The ``nodes`` and ``edges`` frames.

    Raises:
        ValueError: If the text is not UTF-8, or holds a number too large for
            its column.
    """
    columns = native.read_goal(str(path))
    op_class = pl.Series(columns["op_class"])
    nodes = pl.DataFrame(
        {
            "rank": columns["rank"],
            "node_id": columns["node_id"],
            "name": _names(columns["name_end"], columns["name_bytes"]),
            "op_class": op_class.replace_strict(
                _OP_CLASSES, return_dtype=pl.String
            ),
            "comm_type": op_class.replace_strict(
                _COMM_TYPES, return_dtype=pl.String
            ),
            "duration_ns": columns["duration_ns"],
            "size_bytes": columns["size_bytes"],
            "src_rank": columns["src_rank"],
            "dst_rank": columns["dst_rank"],
            "message_tag": columns["message_tag"],
            "num_ops": np.zeros(len(op_class), dtype=np.uint64),
            "pg_name": pl.repeat("", len(op_class), eager=True),
            "cpu": columns["cpu"],
            "nic": columns["nic"],
        },
        schema=reader.NODES_SCHEMA,
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
        schema=reader.EDGES_SCHEMA,
    )
    return nodes, edges


def _p2p_traffic(nodes: pl.DataFrame) -> pl.DataFrame:
    """Builds the ``traffic`` table from the point-to-point sends of nodes."""
    if not nodes.height:
        return pl.DataFrame(schema=reader.TRAFFIC_SCHEMA)
    return (
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
        .cast(reader.TRAFFIC_SCHEMA)
    )


@plugins.register_adapter("goal", version="1.2")
class GoalAdapter:
    """Reads GOAL schedules, as .goal text or SPCL txt2bin .bin binary.

    Attributes:
        format_name: The registered trace format name, ``"goal"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "goal"
    version = "1.2"

    def detect(self, path: pathlib.Path) -> bool:
        """Checks whether a file is a GOAL schedule.

        Args:
            path: The candidate file.

        Returns:
            True for a .goal file, or a .bin file that starts with a GOAL
            magic cookie.
        """
        suffix = path.suffix.lower()
        if suffix == ".goal":
            return True
        if suffix == ".bin":
            return _is_goal_bin(path)
        return False

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the artifact's schedules to nodes/edges/traffic tables.

        Streams chunks from native C++ readers directly into zstd Parquet row
        groups in bounded RAM. The traffic table counts point-to-point sends
        only.

        Args:
            artifact: The GOAL artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"nodes"``, ``"edges"`` and ``"traffic"`` mapped
            to their Parquet paths.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = out_dir / "nodes.parquet"
        edges_path = out_dir / "edges.parquet"
        traffic_acc = common.OnlineTrafficAccumulator(include_collectives=False)

        with (
            common.StreamingParquetWriter(
                nodes_path, reader.NODES_SCHEMA
            ) as node_writer,
            common.StreamingParquetWriter(
                edges_path, reader.EDGES_SCHEMA
            ) as edge_writer,
        ):
            for path in sorted(artifact.paths):
                stream = (
                    native.open_goal_bin_stream(str(path))
                    if _is_goal_bin(path)
                    else native.open_goal_text_stream(str(path))
                )
                for chunk in stream:
                    nodes_df, edges_df = common.chunk_to_graph_frames(
                        chunk, reader.NODES_SCHEMA, reader.EDGES_SCHEMA
                    )
                    traffic_acc.update(nodes_df)
                    node_writer.write_frame(nodes_df)
                    edge_writer.write_frame(edges_df)

        traffic = traffic_acc.finish(
            reader.TRAFFIC_SCHEMA, min_world_size=artifact.world_size
        )
        traffic_path = out_dir / "traffic.parquet"
        traffic.write_parquet(traffic_path, compression="zstd")
        return {
            "nodes": nodes_path,
            "edges": edges_path,
            "traffic": traffic_path,
        }
