"""Matches the compute ops of two runs and cuts each rank's run into phases.

Two simulations of one workload (a Chakra trace and the GOAL schedule made
from it, two GOAL schedules, a run before and after a change) share their
computation even when their communication differs, so compute ops are the
anchors. Per rank, when both runs have the same number of compute ops the
k-th is matched with the k-th; otherwise the two sequences are aligned by
patience diff on their durations (bucketed to a relative ``tolerance``),
which matches the ops whose durations agree in order and leaves the rest
unmatched.

A phase is the stretch of a rank from one matched anchor to the next: the
anchor's compute and the communication and waiting that follow it until the
next anchor starts. Its region is the ops between the two anchors in program
order, whose sends and collectives are the phase's communication.
"""

from __future__ import annotations

import bisect
import collections
from collections.abc import Sequence
import math

import numpy as np
import polars as pl


def anchors(ops: pl.DataFrame) -> pl.DataFrame:
    """Returns a run's anchors: compute ops that come from the trace.

    Columns: ``rank``, ``op``, ``k`` (index within the rank), ``node_id``,
    ``name``, ``start_ns``, ``end_ns``, ``duration_ns``.
    """
    return (
        ops.filter((pl.col("kind") == "calc") & (pl.col("role") == "original"))
        .select(
            "rank", "op", "node_id", "name", "start_ns", "end_ns", "duration_ns"
        )
        .sort("rank", "op")
        .with_columns(
            pl.int_range(pl.len()).over("rank").cast(pl.UInt32).alias("k")
        )
    )


def _tokens(durations: np.ndarray, tolerance: float) -> list[int]:
    """Buckets durations so that ones within ``tolerance`` mostly share one."""
    step = math.log1p(max(tolerance, 1e-9))
    return [
        -1 if d <= 0 else int(round(math.log(float(d)) / step))
        for d in durations
    ]


def _longest_increasing(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The longest run of pairs (sorted by first) whose second items rise."""
    tails: list[int] = []
    tail_index: list[int] = []
    parent = [-1] * len(pairs)
    for i, (_, j) in enumerate(pairs):
        at = bisect.bisect_left(tails, j)
        if at == len(tails):
            tails.append(j)
            tail_index.append(i)
        else:
            tails[at] = j
            tail_index[at] = i
        parent[i] = tail_index[at - 1] if at > 0 else -1
    out: list[tuple[int, int]] = []
    i = tail_index[-1] if tail_index else -1
    while i >= 0:
        out.append(pairs[i])
        i = parent[i]
    return out[::-1]


def patience(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int]]:
    """Returns matched (i, j) index pairs of two token sequences, in order.

    Patience diff: common prefixes and suffixes match; in between, tokens
    that occur once in each side and keep their order become fixed points,
    and the stretches between them are aligned the same way.
    """
    out: list[tuple[int, int]] = []
    stack = [(0, len(a), 0, len(b))]
    while stack:
        a0, a1, b0, b1 = stack.pop()
        while a0 < a1 and b0 < b1 and a[a0] == b[b0]:
            out.append((a0, b0))
            a0 += 1
            b0 += 1
        while a0 < a1 and b0 < b1 and a[a1 - 1] == b[b1 - 1]:
            a1 -= 1
            b1 -= 1
            out.append((a1, b1))
        if a0 >= a1 or b0 >= b1:
            continue
        count_a = collections.Counter(a[a0:a1])
        count_b = collections.Counter(b[b0:b1])
        where_b = {b[j]: j for j in range(b0, b1) if count_b[b[j]] == 1}
        pairs = [
            (i, where_b[a[i]])
            for i in range(a0, a1)
            if count_a[a[i]] == 1 and a[i] in where_b
        ]
        fixed = _longest_increasing(pairs)
        if not fixed:
            continue
        last_a, last_b = a0 - 1, b0 - 1
        for i, j in fixed:
            stack.append((last_a + 1, i, last_b + 1, j))
            out.append((i, j))
            last_a, last_b = i, j
        stack.append((last_a + 1, a1, last_b + 1, b1))
    return sorted(out)


def match(
    base: pl.DataFrame, other: pl.DataFrame, *, tolerance: float = 0.01
) -> pl.DataFrame:
    """Matches two runs' anchors rank by rank.

    Args:
        base: ``anchors`` of the baseline run.
        other: ``anchors`` of the other run.
        tolerance: Relative difference under which two durations count as
            equal when the counts differ and the sequences must be aligned.

    Returns:
        ``rank``, ``phase`` (index of the match within the rank), ``op_base``,
        ``op_other`` and ``positional`` (whether the counts were equal).
    """
    frames = []
    ranks = sorted(set(base["rank"].to_list()) & set(other["rank"].to_list()))
    base_by_rank = base.partition_by("rank", as_dict=True)
    other_by_rank = other.partition_by("rank", as_dict=True)
    for rank in ranks:
        a = base_by_rank[(rank,)]
        b = other_by_rank[(rank,)]
        ops_a = a["op"].to_numpy()
        ops_b = b["op"].to_numpy()
        if len(ops_a) == len(ops_b):
            pairs_a, pairs_b, positional = ops_a, ops_b, True
        else:
            pairs = patience(
                _tokens(a["duration_ns"].to_numpy(), tolerance),
                _tokens(b["duration_ns"].to_numpy(), tolerance),
            )
            pairs_a = ops_a[[i for i, _ in pairs]] if pairs else ops_a[:0]
            pairs_b = ops_b[[j for _, j in pairs]] if pairs else ops_b[:0]
            positional = False
        frames.append(
            pl.DataFrame(
                {
                    "rank": np.full(len(pairs_a), rank, dtype=np.uint32),
                    "phase": np.arange(len(pairs_a), dtype=np.uint32),
                    "op_base": pairs_a.astype(np.uint32),
                    "op_other": pairs_b.astype(np.uint32),
                    "positional": np.full(len(pairs_a), positional),
                }
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "rank": pl.UInt32,
                "phase": pl.UInt32,
                "op_base": pl.UInt32,
                "op_other": pl.UInt32,
                "positional": pl.Boolean,
            }
        )
    return pl.concat(frames)


def phases(
    ops: pl.DataFrame, ranks: pl.DataFrame, anchor_ops: pl.DataFrame
) -> pl.DataFrame:
    """Measures one run's phases, given the ops that start them.

    Args:
        ops: The run's op table.
        ranks: The run's rank table (for the end of each rank's last phase).
        anchor_ops: ``rank``, ``phase`` and ``op`` of each phase's anchor.

    Returns:
        ``rank``, ``phase``, ``anchor`` (name), ``start_ns``, ``span_ns``
        (to the next anchor's start, or the rank's finish), ``compute_ns``
        (the anchor's own time), ``comm_ns`` (the rest), ``bytes`` and
        ``messages`` sent in the region, and ``collectives`` (the region's
        collective types).
    """
    starts = (
        anchor_ops.join(
            ops.select("op", "name", "start_ns", "end_ns"), on="op", how="left"
        )
        .sort("rank", "phase")
        .join(ranks.select("rank", "finish_ns"), on="rank", how="left")
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
        .sort("op")
        .join_asof(
            anchor_ops.select(
                "rank", "phase", pl.col("op").alias("anchor_op")
            ).sort("anchor_op"),
            left_on="op",
            right_on="anchor_op",
            by="rank",
            strategy="backward",
            check_sortedness=False,  # both sides are sorted by op
        )
        .filter(pl.col("phase").is_not_null())
        .group_by("rank", "phase")
        .agg(
            pl.col("bytes")
            .filter(pl.col("kind") == "send")
            .sum()
            .alias("bytes"),
            (pl.col("kind") == "send").sum().alias("messages"),
            pl.col("comm_type")
            .filter(
                (pl.col("kind") == "collective") | (pl.col("role") == "entry")
            )
            .unique(maintain_order=True)
            .str.join(",")
            .alias("collectives"),
        )
    )
    return (
        starts.join(region, on=["rank", "phase"], how="left")
        .select(
            "rank",
            "phase",
            pl.col("name").alias("anchor"),
            "start_ns",
            (pl.col("next_ns") - pl.col("start_ns")).alias("span_ns"),
            (pl.col("end_ns") - pl.col("start_ns")).alias("compute_ns"),
            pl.col("bytes").fill_null(0),
            pl.col("messages").fill_null(0),
            pl.col("collectives").fill_null(""),
        )
        .with_columns(
            (pl.col("span_ns") - pl.col("compute_ns")).alias("comm_ns")
        )
    )
