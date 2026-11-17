"""Lines up two runs of one workload by their compute ops.

Per rank, the k-th compute op of one run is matched with the k-th of the
other when both have as many; otherwise by patience diff on durations
bucketed to a relative tolerance. A phase runs from one matched compute op
to the next; its region is the ops between them in program order.
"""

from __future__ import annotations

import bisect
import collections
from collections.abc import Sequence
import math

import numpy as np
import polars as pl


def anchors(ops: pl.DataFrame) -> pl.DataFrame:
    """A run's compute ops from the trace: ``rank``, ``op``, ``duration_ns``."""
    return (
        ops.filter((pl.col("kind") == "calc") & (pl.col("role") == "original"))
        .select("rank", "op", "duration_ns")
        .sort("rank", "op")
    )


def patience(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int]]:
    """Matched index pairs of two token sequences, in order (patience diff)."""
    out: list[tuple[int, int]] = []
    stack = [(0, len(a), 0, len(b))]
    while stack:
        a0, a1, b0, b1 = stack.pop()
        while a0 < a1 and b0 < b1 and a[a0] == b[b0]:
            out.append((a0, b0))
            a0, b0 = a0 + 1, b0 + 1
        while a0 < a1 and b0 < b1 and a[a1 - 1] == b[b1 - 1]:
            a1, b1 = a1 - 1, b1 - 1
            out.append((a1, b1))
        count_a = collections.Counter(a[a0:a1])
        count_b = collections.Counter(b[b0:b1])
        where_b = {b[j]: j for j in range(b0, b1) if count_b[b[j]] == 1}
        pairs = [
            (i, where_b[a[i]])
            for i in range(a0, a1)
            if count_a[a[i]] == 1 and a[i] in where_b
        ]
        # The longest run of pairs whose b index rises.
        tails: list[int] = []
        ends: list[int] = []
        parent = [-1] * len(pairs)
        for k, (_, j) in enumerate(pairs):
            at = bisect.bisect_left(tails, j)
            tails[at : at + 1] = [j]
            ends[at : at + 1] = [k]
            parent[k] = ends[at - 1] if at else -1
        fixed = []
        k = ends[-1] if ends else -1
        while k >= 0:
            fixed.append(pairs[k])
            k = parent[k]
        last = (a0 - 1, b0 - 1)
        for i, j in reversed(fixed):
            stack.append((last[0] + 1, i, last[1] + 1, j))
            out.append((i, j))
            last = (i, j)
        if fixed:
            stack.append((last[0] + 1, a1, last[1] + 1, b1))
    return sorted(out)


def match(
    base: pl.DataFrame, other: pl.DataFrame, *, tolerance: float = 0.01
) -> pl.DataFrame:
    """Matches anchors: ``rank``, ``phase``, ``op_base``, ``op_other``."""
    step = math.log1p(tolerance)

    def tokens(durations: np.ndarray) -> list[int]:
        return [
            -1 if d <= 0 else round(math.log(d) / step)
            for d in durations.tolist()
        ]

    frames = []
    by_rank = other.partition_by("rank", as_dict=True)
    for (rank,), mine in base.partition_by("rank", as_dict=True).items():
        if (rank,) not in by_rank:
            continue
        theirs = by_rank[(rank,)]
        a, b = mine["op"].to_numpy(), theirs["op"].to_numpy()
        if len(a) != len(b):
            pairs = patience(
                tokens(mine["duration_ns"].to_numpy()),
                tokens(theirs["duration_ns"].to_numpy()),
            )
            a, b = a[[i for i, _ in pairs]], b[[j for _, j in pairs]]
        frames.append(
            pl.DataFrame(
                {
                    "rank": rank,
                    "phase": np.arange(len(a)),
                    "op_base": a,
                    "op_other": b,
                }
            )
        )
    schema = pl.Schema(
        dict.fromkeys(("rank", "phase", "op_base", "op_other"), pl.UInt32())
    )
    return (
        pl.concat([f.cast(schema) for f in frames])
        if frames
        else pl.DataFrame(schema=schema)
    )


def phases(
    ops: pl.DataFrame, ranks: pl.DataFrame, starts: pl.DataFrame
) -> pl.DataFrame:
    """Measures a run's phases from their first ops (rank, phase, op).

    Returns ``rank``, ``phase``, ``anchor`` (its name), ``start_ns``,
    ``span_ns`` (to the next phase or the rank's finish), ``comm_ns`` (the
    span less the anchor's own time), and the region's sent ``bytes`` and
    ``collectives``.
    """
    timed = (
        starts.join(ops.select("op", "name", "start_ns", "end_ns"), on="op")
        .join(ranks.select("rank", "finish_ns"), on="rank")
        .sort("rank", "phase")
        .with_columns(
            pl.col("start_ns")
            .shift(-1)
            .over("rank")
            .fill_null(pl.col("finish_ns"))
            .alias("next_ns")
        )
    )
    region = (
        ops.select("op", "rank", "kind", "role", "bytes", "comm_type")
        .join_asof(
            starts.select("rank", "phase", pl.col("op").alias("first")).sort(
                "first"
            ),
            left_on="op",
            right_on="first",
            by="rank",
            check_sortedness=False,  # both sides are sorted by op
        )
        .drop_nulls("phase")
        .group_by("rank", "phase")
        .agg(
            pl.col("bytes").filter(pl.col("kind") == "send").sum(),
            pl.col("comm_type")
            .filter(
                (pl.col("kind") == "collective") | (pl.col("role") == "entry")
            )
            .unique(maintain_order=True)
            .str.join(",")
            .alias("collectives"),
        )
    )
    return timed.join(region, on=["rank", "phase"], how="left").select(
        "rank",
        "phase",
        pl.col("name").alias("anchor"),
        "start_ns",
        (pl.col("next_ns") - pl.col("start_ns")).alias("span_ns"),
        (pl.col("next_ns") - pl.col("end_ns")).alias("comm_ns"),
        pl.col("bytes").fill_null(0),
        pl.col("collectives").fill_null(""),
    )
