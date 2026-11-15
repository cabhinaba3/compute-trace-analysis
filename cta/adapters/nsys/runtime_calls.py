"""CUDA runtime calls of one rank as ``runtime_calls`` rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from cta.adapters.nsys import network_ops

RUNTIME_CALLS_SCHEMA = pl.Schema(
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


def runtime_calls_frame(
    names: Sequence[str],
    name_code: np.ndarray,
    start_ns: np.ndarray,
    end_ns: np.ndarray,
    thread: np.ndarray,
    correlation: np.ndarray,
    rank: int,
    steps: pl.DataFrame,
) -> pl.DataFrame:
    """Builds a rank's runtime-call table from the reader's columns.

    Args:
        names: The reader's name table.
        name_code: Index into ``names`` for each call.
        start_ns: Call start times.
        end_ns: Call end times.
        thread: Calling thread of each call.
        correlation: Correlation id of each call, -1 if none.
        rank: The rank written to every row.
        steps: The rank's step ranges (``step``, ``start_ns``, ``end_ns``),
            used to number the calls; calls outside every step get -1.

    Returns:
        A frame conforming to ``RUNTIME_CALLS_SCHEMA``, in start order.
    """
    if not len(name_code):
        return pl.DataFrame(schema=RUNTIME_CALLS_SCHEMA)
    order = np.argsort(start_ns, kind="stable")
    start = np.asarray(start_ns)[order]
    end = np.asarray(end_ns)[order]
    name_array = np.array(list(names), dtype=object)[
        np.asarray(name_code)[order]
    ]
    return pl.DataFrame(
        {
            "rank": np.full(len(order), rank, dtype=np.int32),
            "step": network_ops.step_of(start, steps),
            "name": pl.Series(name_array, dtype=pl.String),
            "thread": np.asarray(thread)[order],
            "correlation": np.asarray(correlation)[order],
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
        }
    ).cast(RUNTIME_CALLS_SCHEMA)
