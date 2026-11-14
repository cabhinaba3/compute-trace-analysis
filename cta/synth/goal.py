"""Synthetic LogGOPSim GOAL schedules (text and compiled binary)."""

from __future__ import annotations

import pathlib
import struct

import numpy as np

from cta.adapters.goal import reader

# Byte offsets in the binary jump table count from the end of the 8-byte magic
# cookie, so the first rank blob starts after the 6-byte rank-count header and
# 16 bytes (start and end offset) per rank.
_RANK_COUNT_HEADER_BYTES = 6
_JUMP_TABLE_ENTRY_BYTES = 16

# Appendix offsets written for nodes without requires / irequires successors.
_NO_REQUIRES_APPENDIX = 999
_NO_IREQUIRES_APPENDIX = 0xFFFFFFFF


def write_goal_allreduce(
    path: pathlib.Path,
    *,
    world_size: int = 4,
    compute_ns: int = 1_000_000,
    allreduce_bytes: int = 400_000,
    steps: int = 2,
    algorithm: str = "ring",
) -> pathlib.Path:
    """Generates a valid multi-rank `.goal` text schedule with an AllReduce.

    Every rank runs, per step, a ``calc`` followed by an AllReduce. With
    ``algorithm="ring"`` the AllReduce is a Ring AllReduce; any other value
    gives a direct exchange in which each rank sends the full buffer to every
    peer.

    Args:
        path: Output ``.goal`` file; its parent directory is created if missing.
        world_size: Number of ranks.
        compute_ns: Duration of each ``calc`` operation in nanoseconds.
        allreduce_bytes: AllReduce buffer size in bytes.
        steps: Number of compute + AllReduce steps per rank.
        algorithm: ``"ring"`` for Ring AllReduce; anything else selects the
            direct exchange.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"num_ranks {world_size}", ""]

    for rank in range(world_size):
        lines.append(f"rank {rank} {{")
        previous_label: str | None = None
        label_count = 0

        def next_label() -> str:
            nonlocal label_count
            label_count += 1
            return f"l{label_count}"

        for step in range(steps):
            calc_label = next_label()
            lines.append(f"{calc_label}: calc {compute_ns} cpu 0")
            if previous_label:
                lines.append(f"{calc_label} requires {previous_label}")
            previous_label = calc_label

            if world_size <= 1:
                continue

            if algorithm == "ring":
                # Ring AllReduce: 2 * (P - 1) sequential phases with a chunk
                # size of B / P.
                chunk_bytes = allreduce_bytes // world_size
                next_rank = (rank + 1) % world_size
                previous_rank = (rank - 1 + world_size) % world_size
                for phase in range(2 * (world_size - 1)):
                    tag = step * 100 + phase
                    send_label = next_label()
                    recv_label = next_label()
                    lines.append(
                        f"{send_label}: send {chunk_bytes}b to {next_rank} "
                        f"tag {tag} cpu 0 nic 0"
                    )
                    lines.append(
                        f"{recv_label}: recv {chunk_bytes}b from "
                        f"{previous_rank} tag {tag} cpu 0 nic 0"
                    )
                    lines.append(f"{send_label} requires {previous_label}")
                    lines.append(f"{recv_label} requires {previous_label}")
                    # The next ring phase waits for this phase's recv.
                    previous_label = recv_label
            else:
                # Direct all-to-all exchange: each rank sends the full B bytes
                # to every peer concurrently.
                barrier_label = next_label()
                lines.append(f"{barrier_label}: calc 0 cpu 0")
                for peer in range(world_size):
                    if peer == rank:
                        continue
                    tag = step * 100 + 1
                    send_label = next_label()
                    recv_label = next_label()
                    lines.append(
                        f"{send_label}: send {allreduce_bytes}b to {peer} "
                        f"tag {tag} cpu 0 nic 0"
                    )
                    lines.append(
                        f"{recv_label}: recv {allreduce_bytes}b from {peer} "
                        f"tag {tag} cpu 0 nic 0"
                    )
                    lines.append(f"{send_label} requires {previous_label}")
                    lines.append(f"{recv_label} requires {previous_label}")
                    lines.append(f"{barrier_label} requires {send_label}")
                    lines.append(f"{barrier_label} requires {recv_label}")
                previous_label = barrier_label

        lines.append("}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


def compile_goal_to_bin(
    goal_path: pathlib.Path, bin_path: pathlib.Path
) -> pathlib.Path:
    """Compiles a `.goal` text file into a SPCL LogGOPSim `.bin` schedule.

    The output follows the binary layout of the specification
    (``MAGIC_COOKIE = 4223``): a header with the rank count and a jump table,
    then one blob per rank holding its root ids, node records and dependency
    appendix.

    Args:
        goal_path: The ``.goal`` text schedule to compile.
        bin_path: Output ``.bin`` file; its parent directory is created if
            missing.

    Returns:
        ``bin_path``.
    """
    nodes, edges = reader.parse_goal_text(goal_path)
    num_ranks = int(np.max(nodes["rank"].to_numpy())) + 1 if nodes.height else 1

    rank_blobs: list[bytes] = []
    for rank in range(num_ranks):
        rank_nodes = nodes.filter(nodes["rank"] == rank).sort("node_id")
        rank_edges = edges.filter(edges["rank"] == rank)
        num_nodes = rank_nodes.height
        if num_nodes == 0:
            rank_blobs.append(struct.pack("<II", 0, 0))
            continue

        in_degree = np.zeros(num_nodes, dtype=np.uint32)
        requires_successors: list[list[int]] = [[] for _ in range(num_nodes)]
        irequires_successors: list[list[int]] = [[] for _ in range(num_nodes)]
        for edge in rank_edges.iter_rows(named=True):
            src_id = int(edge["src_id"])
            dst_id = int(edge["dst_id"])
            dep_kind = str(edge["dep_kind"])
            in_degree[dst_id] += 1
            if dep_kind == "irequires":
                irequires_successors[src_id].append(dst_id)
            else:
                requires_successors[src_id].append(dst_id)

        root_ids = np.flatnonzero(in_degree == 0).astype(np.uint32)
        records = np.zeros(num_nodes, dtype=reader.GOAL_BIN_NODE_DTYPE)
        appendix: list[int] = []

        for i, node in enumerate(rank_nodes.iter_rows(named=True)):
            # A structured-array element is a view: writes go to `records`.
            record = records[i]
            op_class = str(node["op_class"])
            record["deps_cnt"] = in_degree[i]
            if op_class == "comm_send":
                record["op_type"] = reader.OPTYPE_SEND
                record["peer"] = int(node["dst_rank"])
                record["size"] = int(node["size_bytes"])
            elif op_class == "comm_recv":
                record["op_type"] = reader.OPTYPE_RECV
                record["peer"] = int(node["src_rank"])
                record["size"] = int(node["size_bytes"])
            else:
                record["op_type"] = reader.OPTYPE_CALC
                record["peer"] = 0
                record["size"] = int(node["duration_ns"])
            record["tag"] = int(node["message_tag"])
            record["proc"] = 0
            record["nic"] = 0

            requires_ids = requires_successors[i]
            record["num_deps"] = len(requires_ids)
            record["deps_apdx"] = (
                len(appendix) if requires_ids else _NO_REQUIRES_APPENDIX
            )
            appendix.extend(requires_ids)

            irequires_ids = irequires_successors[i]
            record["num_startdeps"] = len(irequires_ids)
            record["startdeps_apdx"] = (
                len(appendix) if irequires_ids else _NO_IREQUIRES_APPENDIX
            )
            appendix.extend(irequires_ids)

        appendix_array = np.asarray(appendix, dtype="<u4")
        rank_blob = b"".join(
            [
                struct.pack("<II", num_nodes, len(root_ids)),
                root_ids.astype("<u4").tobytes(),
                records.tobytes(),
                appendix_array.tobytes(),
            ]
        )
        rank_blobs.append(rank_blob)

    jump_table: list[int] = []
    blob_offset = _RANK_COUNT_HEADER_BYTES + _JUMP_TABLE_ENTRY_BYTES * num_ranks
    for rank_blob in rank_blobs:
        jump_table.extend([blob_offset, blob_offset + len(rank_blob)])
        blob_offset += len(rank_blob)

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with bin_path.open("wb") as fh:
        fh.write(
            struct.pack("<QIBB", reader.GOAL_MAGIC_COOKIE, num_ranks, 0, 0)
        )
        fh.write(np.asarray(jump_table, dtype="<u8").tobytes())
        for rank_blob in rank_blobs:
            fh.write(rank_blob)
    return bin_path
