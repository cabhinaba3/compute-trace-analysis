"""GOAL (.goal text and SPCL txt2bin .bin binary) schedule adapter.

Both encodings are written as the Chakra adapter's ``nodes``, ``edges`` and
``traffic`` tables, so the two formats can be compared with each other and
with Chakra traces.
"""

from __future__ import annotations

import mmap
import pathlib
import re
import struct

import numpy as np
import polars as pl

from cta.adapters.chakra import reader
from cta.core import models
from cta.core import registry

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

_BYTES_PER_UNIT = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}
_RANK_HEADER_RE = re.compile(r"^rank\s+(\d+)\s*\{?")
_DEPENDENCY_RE = re.compile(r"^(\S+)\s+(i?requires)\s+(\S+)$")

# Building blocks of the send/recv/calc line patterns.
_LABEL_PREFIX = r"^(?:(?P<label>[^:\s]+):\s*)?"
_SIZE = r"(?P<size>\d+)(?P<unit>[kmgt]?b)?"
_TAG_OPTION = r"(?:\s+tag\s+(?P<tag>\d+))?"
_CPU_NIC_OPTIONS = r"(?:\s+cpu\s+(?P<cpu>\d+))?(?:\s+nic\s+(?P<nic>\d+))?$"

_SEND_RE = re.compile(
    _LABEL_PREFIX
    + r"send\s+"
    + _SIZE
    + r"\s+to\s+(?P<peer>\d+)"
    + _TAG_OPTION
    + _CPU_NIC_OPTIONS,
    re.IGNORECASE,
)
_RECV_RE = re.compile(
    _LABEL_PREFIX
    + r"recv\s+"
    + _SIZE
    + r"\s+from\s+(?P<peer>\d+)"
    + _TAG_OPTION
    + _CPU_NIC_OPTIONS,
    re.IGNORECASE,
)
_CALC_RE = re.compile(
    _LABEL_PREFIX + r"calc\s+(?P<duration>\d+)" + _CPU_NIC_OPTIONS,
    re.IGNORECASE,
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
    with (
        path.open("rb") as fh,
        mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file,
    ):
        node_frames, edge_frames = _decode_goal_mmap(mapped_file, path)
    nodes = (
        pl.concat(node_frames)
        if node_frames
        else pl.DataFrame(schema=reader.NODES_SCHEMA)
    )
    edges = (
        pl.concat(edge_frames)
        if edge_frames
        else pl.DataFrame(schema=reader.EDGES_SCHEMA)
    )
    return nodes, edges


def _strip_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Removes ``/* */``, ``//`` and ``#`` comments from a stripped line.

    Args:
        line: The line, without surrounding whitespace.
        in_block_comment: Whether a ``/*`` comment is open before the line.

    Returns:
        The remaining code (empty if none) and whether a ``/*`` comment is
        still open after the line.
    """
    if in_block_comment:
        if "*/" not in line:
            return "", True
        line = line.split("*/", 1)[1].strip()
        in_block_comment = False
    if "/*" in line:
        before, after = line.split("/*", 1)
        if "*/" in after:
            line = (before + " " + after.split("*/", 1)[1]).strip()
        else:
            line = before.strip()
            in_block_comment = True
    for comment_token in ("//", "#"):
        if comment_token in line:
            line = line.split(comment_token, 1)[0].strip()
    return line, in_block_comment


def _comm_op_row(
    match: re.Match[str], rank: int, node_id: int, op_class: str
) -> dict[str, object]:
    """Builds the ``nodes`` row of a matched send or recv line.

    Args:
        match: The ``_SEND_RE`` or ``_RECV_RE`` match.
        rank: The rank whose block holds the line.
        node_id: The node's index within the rank.
        op_class: ``"comm_send"`` or ``"comm_recv"``.

    Returns:
        The node row.
    """
    is_send = op_class == "comm_send"
    unit = (match.group("unit") or "b").lower()
    size_bytes = int(match.group("size")) * _BYTES_PER_UNIT.get(unit, 1)
    peer = int(match.group("peer"))
    return {
        "rank": rank,
        "node_id": node_id,
        "name": match.group("label") or op_class,
        "op_class": op_class,
        "comm_type": "SEND" if is_send else "RECV",
        "duration_ns": 0,
        "size_bytes": size_bytes,
        "src_rank": rank if is_send else peer,
        "dst_rank": peer if is_send else rank,
        "message_tag": int(match.group("tag") or 0),
        "num_ops": 0,
    }


def _calc_op_row(
    match: re.Match[str], rank: int, node_id: int
) -> dict[str, object]:
    """Builds the ``nodes`` row of a matched calc line."""
    return {
        "rank": rank,
        "node_id": node_id,
        "name": match.group("label") or "compute",
        "op_class": "compute",
        "comm_type": "",
        "duration_ns": int(match.group("duration")),
        "size_bytes": 0,
        "src_rank": -1,
        "dst_rank": -1,
        "message_tag": 0,
        "num_ops": 0,
    }


def _parse_op_line(
    line: str, rank: int, node_id: int
) -> tuple[str | None, dict[str, object]] | None:
    """Parses a send, recv or calc line.

    Args:
        line: The line, without comments.
        rank: The rank whose block holds the line.
        node_id: The id to give the node.

    Returns:
        The op's label (None if unlabelled) and its ``nodes`` row, or None if
        the line is not an op.
    """
    if send_match := _SEND_RE.match(line):
        row = _comm_op_row(send_match, rank, node_id, "comm_send")
        return send_match.group("label"), row
    if recv_match := _RECV_RE.match(line):
        row = _comm_op_row(recv_match, rank, node_id, "comm_recv")
        return recv_match.group("label"), row
    if calc_match := _CALC_RE.match(line):
        return calc_match.group("label"), _calc_op_row(
            calc_match, rank, node_id
        )
    return None


def parse_goal_text(path: pathlib.Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parses a GOAL text schedule into ``nodes`` and ``edges`` frames.

    Nodes are numbered per rank in file order. Labels are scoped to their rank
    block, and dependencies naming an unknown label are dropped.

    Args:
        path: The .goal file to read.

    Returns:
        The ``nodes`` and ``edges`` frames.
    """
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    rank = 0
    next_node_id = 0
    node_ids_by_label: dict[str, int] = {}
    in_block_comment = False

    with path.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            line, in_block_comment = _strip_comments(line, in_block_comment)
            if not line or line in ("{", "}") or line.startswith("num_ranks"):
                if line == "}":
                    node_ids_by_label.clear()
                continue

            rank_match = _RANK_HEADER_RE.match(line)
            if rank_match:
                rank = int(rank_match.group(1))
                next_node_id = 0
                node_ids_by_label.clear()
                continue

            dependency_match = _DEPENDENCY_RE.match(line)
            if dependency_match:
                dst_label, dep_kind, src_label = dependency_match.groups()
                if (
                    src_label in node_ids_by_label
                    and dst_label in node_ids_by_label
                ):
                    edge_rows.append(
                        {
                            "rank": rank,
                            "src_id": node_ids_by_label[src_label],
                            "dst_id": node_ids_by_label[dst_label],
                            "dep_kind": dep_kind,
                        }
                    )
                continue

            parsed_op = _parse_op_line(line, rank, next_node_id)
            if parsed_op is None:
                continue
            label, row = parsed_op
            if label:
                node_ids_by_label[label] = next_node_id
            node_rows.append(row)
            next_node_id += 1

    nodes = (
        pl.DataFrame(node_rows, schema=reader.NODES_SCHEMA)
        if node_rows
        else pl.DataFrame(schema=reader.NODES_SCHEMA)
    )
    edges = (
        pl.DataFrame(edge_rows, schema=reader.EDGES_SCHEMA)
        if edge_rows
        else pl.DataFrame(schema=reader.EDGES_SCHEMA)
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


@registry.register_adapter("goal", version="1.1")
class GoalAdapter:
    """Reads GOAL schedules, as .goal text or SPCL txt2bin .bin binary.

    Attributes:
        format_name: The registered trace format name, ``"goal"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "goal"
    version = "1.1"

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

        The traffic table counts point-to-point sends only.

        Args:
            artifact: The GOAL artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"nodes"``, ``"edges"`` and ``"traffic"`` mapped
            to their Parquet paths.
        """
        node_frames, edge_frames = [], []
        for path in sorted(artifact.paths):
            nodes, edges = (
                _parse_goal_bin(path)
                if _is_goal_bin(path)
                else parse_goal_text(path)
            )
            node_frames.append(nodes)
            edge_frames.append(edges)

        all_nodes = (
            pl.concat(node_frames)
            if node_frames
            else pl.DataFrame(schema=reader.NODES_SCHEMA)
        )
        all_edges = (
            pl.concat(edge_frames)
            if edge_frames
            else pl.DataFrame(schema=reader.EDGES_SCHEMA)
        )
        traffic = _p2p_traffic(all_nodes)

        nodes_path = out_dir / "nodes.parquet"
        edges_path = out_dir / "edges.parquet"
        traffic_path = out_dir / "traffic.parquet"
        all_nodes.write_parquet(nodes_path, compression="zstd")
        all_edges.write_parquet(edges_path, compression="zstd")
        traffic.write_parquet(traffic_path, compression="zstd")
        return {
            "nodes": nodes_path,
            "edges": edges_path,
            "traffic": traffic_path,
        }
