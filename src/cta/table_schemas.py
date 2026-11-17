"""Schemas of the tables that several adapters write and many readers use.

Adapters that see GPU kernels write ``kernels`` and ``steps`` (and, where the
trace has them, ``runtime``, ``osrt``, ``gpu_metrics``, ``runtime_calls`` and
``network_ops``) in these shapes, so aspects and plots work the same whichever
profiler produced the trace. Times are integer nanoseconds.
"""

from __future__ import annotations

import polars as pl

KERNELS = pl.Schema(
    {
        "step": pl.Int32(),
        "scope": pl.String(),
        # Enclosing ranges, "outermost > ... > innermost" (equal to scope for
        # the last level).
        "scope_path": pl.String(),
        "name": pl.String(),
        "demangled_name": pl.String(),
        "canonical_name": pl.String(),
        "op_family": pl.String(),
        "category": pl.String(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
        "bytes": pl.Int64(),
        "stream": pl.Int32(),
        # Per-report/process index; 0 for single-report sources.
        "rank": pl.Int32(),
        # CUDA device id within the report; 0 for single-GPU sources.
        "device": pl.Int32(),
    }
)

STEPS = pl.Schema(
    {
        "step": pl.Int32(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
        "rank": pl.Int32(),
    }
)

_DEFAULTS: dict[str, object] = {"rank": 0, "device": 0, "scope_path": ""}


def conform(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Selects and casts a frame to a schema, filling defaultable columns.

    Missing ``rank`` and ``device`` columns are filled with 0 and a missing
    ``scope_path`` column with "".

    Args:
        df: The frame to conform.
        schema: The target schema.

    Returns:
        The frame with exactly the columns of ``schema``, in its order and
        with its types.

    Raises:
        KeyError: ``df`` lacks a column of ``schema`` that has no default.
    """
    missing = [column for column in schema if column not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.lit(_DEFAULTS[column]).alias(column) for column in missing]
        )
    return df.select(list(schema.keys())).cast(schema)


RUNTIME = pl.Schema(
    {"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()}
)
OSRT = pl.Schema(
    {"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()}
)
GPU_METRICS = pl.Schema(
    {
        "metric_name": pl.String(),
        "timestamp_ns": pl.Int64(),
        "value": pl.Float64(),
        "rank": pl.Int32(),
    }
)

# Category codes the native readers give GPU activities -> category name.
GPU_CATEGORIES = {0: "compute", 1: "comm", 2: "memcpy"}

RUNTIME_CALLS = pl.Schema(
    {
        "rank": pl.Int32(),
        "step": pl.Int32(),
        "name": pl.String(),
        # Thread that made the call: nsys globalTid, or a dense id for Kineto;
        # -1 when unknown.
        "thread": pl.Int64(),
        # Id shared with the GPU work the call launched; -1 when there is none.
        "correlation": pl.Int64(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
    }
)


NETWORK_OPS = pl.Schema(
    {
        "rank": pl.Int32(),
        "device": pl.Int32(),
        "stream": pl.Int32(),
        "step": pl.Int32(),
        # Communication library that issued the call, e.g. "nccl".
        "api": pl.String(),
        # Logical operation, lower case without separators: "allreduce".
        "op": pl.String(),
        # Algorithm and protocol NCCL picked ("RING", "LL"); "" if unknown.
        "algo": pl.String(),
        "proto": pl.String(),
        "dtype": pl.String(),
        # Process group the call ran in; "" if unknown.
        "group": pl.String(),
        "group_size": pl.Int32(),
        # Rank on the other end of a point-to-point call, as the communicator
        # numbers it; -1 for collectives and when the trace does not say.
        "peer": pl.Int32(),
        # Message size in bytes as nccl-tests counts it; 0 when the trace does
        # not record it.
        "bytes": pl.Int64(),
        # True when the call only posts the transfer (MPI_Isend, a UCX
        # submit): its duration says nothing about bandwidth.
        "posted": pl.Boolean(),
        # When the host issued the call (the CUDA launch of the kernel or
        # copy); -1 when the trace does not say. start_ns - launch_ns is how
        # long the GPU work waited before it began.
        "launch_ns": pl.Int64(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
    }
)
