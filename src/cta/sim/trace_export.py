"""A simulated timeline as Chrome trace JSON (Perfetto, chrome://tracing).

Each rank is a process with a track per CPU, one for messages and one for
collectives; arrows join each send to its receiver. The critical path is one
more process.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import pathlib
from typing import Any

import polars as pl

from cta.sim import result as result_lib

_MESSAGES, _COLLECTIVES, _PATH = 1000, 1001, 1 << 30


def chrome_events(
    result: result_lib.SimResult,
    *,
    ranks: Iterable[int] | None = None,
    max_ops: int = 1_000_000,
) -> list[dict[str, Any]]:
    """The run's trace events (at most ``max_ops`` ops, the earliest)."""
    if result.ops.is_empty():
        raise ValueError("the result kept no op table")
    us = 1e-3
    ops = result.ops.filter(
        pl.col("end_ns").is_not_null() & (pl.col("kind") != "nop")
    )
    messages = result.messages
    if ranks is not None:
        keep = list(ranks)
        ops = ops.filter(pl.col("rank").is_in(keep))
        messages = messages.filter(
            pl.col("src").is_in(keep) & pl.col("dst").is_in(keep)
        )
    ops = ops.sort("start_ns").head(max_ops)
    kind = pl.col("kind")
    events = ops.select(
        ph=pl.lit("X"),
        name=pl.when(pl.col("role") == "original")
        .then("name")
        .otherwise(pl.format("{}:{}", "name", kind)),
        cat=kind,
        pid="rank",
        tid=pl.when(kind == "calc")
        .then("cpu")
        .when(kind == "collective")
        .then(_COLLECTIVES)
        .otherwise(_MESSAGES),
        ts=pl.col("start_ns") * us,
        dur=(pl.col("end_ns") - pl.col("start_ns")) * us,
        args=pl.struct("op", "node_id", "bytes", "peer", "origin"),
    ).to_dicts()
    for row in messages.head(max_ops).iter_rows(named=True):
        flow = {
            "name": "message",
            "cat": "message",
            "id": row["send"],
            "tid": _MESSAGES,
        }
        events.append(
            flow | {"ph": "s", "pid": row["src"], "ts": row["start_ns"] * us}
        )
        events.append(
            flow
            | {
                "ph": "f",
                "bp": "e",
                "pid": row["dst"],
                "ts": row["done_ns"] * us,
            }
        )
    events += result.critical_path.select(
        ph=pl.lit("X"),
        name="segment",
        cat=pl.lit("critical_path"),
        pid=pl.lit(_PATH),
        tid=pl.lit(0),
        ts=pl.col("start_ns") * us,
        dur=pl.col("duration_ns") * us,
        args=pl.struct("op", "rank", "name"),
    ).to_dicts()
    names = [(_PATH, None, "critical path")] + [
        (rank, tid, label)
        for rank in ops["rank"].unique().to_list()
        for tid, label in (
            (None, f"rank {rank}"),
            (_MESSAGES, "messages"),
            (_COLLECTIVES, "collectives"),
        )
    ]
    for pid, tid, label in names:
        meta = {
            "ph": "M",
            "pid": pid,
            "name": "process_name" if tid is None else "thread_name",
            "args": {"name": label},
        }
        events.append(meta if tid is None else meta | {"tid": tid})
    return events


def write_chrome_trace(
    result: result_lib.SimResult, path: str | pathlib.Path, **options: Any
) -> int:
    """Writes ``chrome_events`` to ``path``; returns how many."""
    events = chrome_events(result, **options)
    pathlib.Path(path).write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "ns"})
    )
    return len(events)
