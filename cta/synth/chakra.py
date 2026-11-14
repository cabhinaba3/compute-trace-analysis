"""Synthetic Chakra execution traces (protobuf ET and JSON host ET)."""

from __future__ import annotations

import pathlib

from cta import _native as native


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
    shard_paths = native.write_chakra_et_run(
        str(out_dir),
        world_size=world_size,
        compute_us=compute_us,
        allreduce_bytes=allreduce_bytes,
        steps=steps,
        straggler_rank=straggler_rank,
        straggler_factor=straggler_factor,
    )
    return [pathlib.Path(shard) for shard in shard_paths]


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
    native.write_chakra_host_et(
        str(path), steps=steps, compute_us=compute_us, comm_bytes=comm_bytes
    )
    return path
