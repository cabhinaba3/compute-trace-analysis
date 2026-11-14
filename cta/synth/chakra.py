"""Synthetic Chakra execution traces (protobuf ET and JSON host ET)."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from cta.adapters.chakra import schema as chakra_schema


def _add_int_attribute(node: Any, name: str, value: int) -> None:
    """Appends an ``int64`` attribute to a Chakra protobuf node."""
    attribute = node.attr.add()
    attribute.name = name
    attribute.int64_val = value


def _append_length_delimited(buffer: bytearray, message_bytes: bytes) -> None:
    """Appends a serialized message preceded by its varint-encoded length."""
    buffer.extend(chakra_schema.encode_varint(len(message_bytes)))
    buffer.extend(message_bytes)


def write_chakra_et_run(
    out_dir: pathlib.Path,
    *,
    world_size: int = 4,
    compute_us: int = 1000,
    allreduce_bytes: int = 400_000,
    steps: int = 2,
    straggler_rank: int | None = None,
    straggler_factor: float = 1.0,
) -> list[pathlib.Path]:
    """Writes `world_size` valid MLCommons Chakra `.et` binary protobuf shards.

    Each shard holds a global-metadata message followed by, per step, one
    compute node and one all-reduce collective node that depends on it. The
    compute node of each step depends on the all-reduce of the previous step.

    Args:
        out_dir: Directory for the shards; created if missing.
        world_size: Number of ranks, one shard per rank.
        compute_us: Duration of each compute node in microseconds.
        allreduce_bytes: Size of each all-reduce; also sets its duration
            (one microsecond per 1000 bytes, at least 10).
        steps: Number of compute + all-reduce pairs per rank.
        straggler_rank: Rank whose compute nodes are slowed down, if any.
        straggler_factor: Multiplier applied to the straggler's compute time.

    Returns:
        The shard paths (``trace.rank<N>.et``), in rank order.
    """
    # Lazy: these attributes load the protobuf module only when .et files are
    # written.
    node_message_type = chakra_schema.NodeMsg
    metadata_message_type = chakra_schema.GlobalMetadataMsg
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[pathlib.Path] = []

    for rank in range(world_size):
        shard_bytes = bytearray()
        metadata = metadata_message_type(version="0.0.4")
        _append_length_delimited(shard_bytes, metadata.SerializeToString())

        node_id = 0
        previous_node_id: int | None = None
        compute_duration_us = (
            int(compute_us * straggler_factor)
            if rank == straggler_rank
            else compute_us
        )

        for step in range(steps):
            compute_node = node_message_type(
                id=node_id,
                name=f"aten::mm_step{step}",
                type=chakra_schema.COMP_NODE,
                duration_micros=compute_duration_us,
            )
            if previous_node_id is not None:
                compute_node.data_deps.append(previous_node_id)
            _add_int_attribute(compute_node, "num_ops", 2 * 1024 * 1024)
            _add_int_attribute(compute_node, "tensor_size", allreduce_bytes)
            _append_length_delimited(
                shard_bytes, compute_node.SerializeToString()
            )
            compute_node_id = node_id
            node_id += 1

            allreduce_node = node_message_type(
                id=node_id,
                name=f"nccl:all_reduce_step{step}",
                type=chakra_schema.COMM_COLL_NODE,
                duration_micros=max(10, allreduce_bytes // 1000),
            )
            allreduce_node.data_deps.append(compute_node_id)
            # A comm_type of 0 is ALL_REDUCE.
            _add_int_attribute(allreduce_node, "comm_type", 0)
            _add_int_attribute(allreduce_node, "comm_size", allreduce_bytes)
            _append_length_delimited(
                shard_bytes, allreduce_node.SerializeToString()
            )
            previous_node_id = node_id
            node_id += 1

        shard_path = out_dir / f"trace.rank{rank}.et"
        shard_path.write_bytes(bytes(shard_bytes))
        shard_paths.append(shard_path)

    return shard_paths


def write_chakra_host_et(
    path: pathlib.Path,
    *,
    steps: int = 2,
    compute_us: float = 500.0,
    comm_bytes: int = 200_000,
) -> pathlib.Path:
    """Writes a Chakra host execution trace in its JSON form.

    Per step it writes an ``aten::linear_<step>`` compute node and a
    ``c10d::all_reduce_<step>`` node that depends on it. Each compute node
    depends on the all-reduce of the previous step.

    Args:
        path: Output JSON file; its parent directory is created if missing.
        steps: Number of compute + all-reduce pairs.
        compute_us: Duration of each compute node in microseconds.
        comm_bytes: Bytes of each all-reduce; its duration is
            ``comm_bytes / 1000`` microseconds.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    node_id = 1
    previous_node_id = None
    for step in range(steps):
        nodes.append(
            {
                "id": node_id,
                "name": f"aten::linear_{step}",
                "ctrl_deps": (
                    [previous_node_id] if previous_node_id is not None else []
                ),
                "dur_us": compute_us,
            }
        )
        compute_node_id = node_id
        node_id += 1
        nodes.append(
            {
                "id": node_id,
                "name": f"c10d::all_reduce_{step}",
                "ctrl_deps": [compute_node_id],
                "dur_us": comm_bytes / 1000.0,
                "comm_bytes": comm_bytes,
            }
        )
        previous_node_id = node_id
        node_id += 1
    path.write_text(
        json.dumps({"schema": "1.0.2-chakra.0.0.4", "nodes": nodes})
    )
    return path
