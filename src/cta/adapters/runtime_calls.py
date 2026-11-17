"""CUDA runtime calls of one rank as ``runtime_calls`` rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from cta import table_schemas
from cta.analysis import step_slicing


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
        A frame conforming to ``table_schemas.RUNTIME_CALLS``, in start order.
    """
    if not len(name_code):
        return pl.DataFrame(schema=table_schemas.RUNTIME_CALLS)
    order = np.argsort(start_ns, kind="stable")
    start = np.asarray(start_ns)[order]
    end = np.asarray(end_ns)[order]
    name_array = np.array(list(names), dtype=object)[
        np.asarray(name_code)[order]
    ]
    return pl.DataFrame(
        {
            "rank": np.full(len(order), rank, dtype=np.int32),
            "step": step_slicing.step_of(start, steps),
            "name": pl.Series(name_array, dtype=pl.String),
            "thread": np.asarray(thread)[order],
            "correlation": np.asarray(correlation)[order],
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
        }
    ).cast(table_schemas.RUNTIME_CALLS)
