"""Writes a simulated timeline as Chrome trace JSON (Perfetto, chrome://tracing).

Each rank is a process. Its calcs run on one track per CPU, its sends and
recvs on a "messages" track and its collectives (analytical level) on a
"collectives" track; arrows join each message's send to its receiver. The
critical path is one more process, its segments coloured by category.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
import pathlib
from typing import Any

import polars as pl

from cta.sim import result as result_lib

_MESSAGE_TRACK = 1000
_COLLECTIVE_TRACK = 1001
_CRITICAL_PID = 1 << 30


def chrome_events(
    result: result_lib.SimResult,
    *,
    ranks: Iterable[int] | None = None,
    max_ops: int = 1_000_000,
    flows: bool = True,
) -> list[dict[str, Any]]:
    """Returns the trace events of a run.

    Args:
        result: A run that kept its op table.
        ranks: Only these ranks; None keeps all.
        max_ops: Keep at most this many ops (the earliest).
        flows: Whether to draw an arrow per message.

    Raises:
        ValueError: The result has no op table.
    """
    if result.ops.is_empty():
        raise ValueError(
            "the result kept no op table; rerun with keep_ops=True"
        )
    ops = result.ops.filter(
        pl.col("start_ns").is_not_null()
        & pl.col("end_ns").is_not_null()
        & (pl.col("kind") != "nop")
    )
    keep = None if ranks is None else sorted(set(ranks))
    if keep is not None:
        ops = ops.filter(pl.col("rank").is_in(keep))
    ops = ops.sort("start_ns").head(max_ops)

    events: list[dict[str, Any]] = []
    for rank in sorted(set(ops["rank"].to_list())):
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": rank,
                "args": {"name": f"rank {rank}"},
            }
        )
        events.append(
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": rank,
                "args": {"sort_index": rank},
            }
        )
        for tid, name in (
            (_MESSAGE_TRACK, "messages"),
            (_COLLECTIVE_TRACK, "collectives"),
        ):
            events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": rank,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
    for row in ops.iter_rows(named=True):
        kind = row["kind"]
        tid = (
            int(row["cpu"])
            if kind == "calc"
            else _COLLECTIVE_TRACK
            if kind == "collective"
            else _MESSAGE_TRACK
        )
        args: dict[str, Any] = {"op": row["op"], "node_id": row["node_id"]}
        if kind != "calc":
            args.update({"bytes": row["bytes"], "peer": row["peer"]})
        if row["role"] != "original":
            args["lowered_from"] = row["origin"]
        events.append(
            {
                "ph": "X",
                "name": row["name"]
                if row["role"] == "original"
                else f"{row['name']}:{kind}",
                "cat": kind,
                "pid": row["rank"],
                "tid": tid,
                "ts": row["start_ns"] / 1000.0,
                "dur": (row["end_ns"] - row["start_ns"]) / 1000.0,
                "args": args,
            }
        )
    if flows and not result.messages.is_empty():
        messages = result.messages.filter(pl.col("done_ns").is_not_null())
        if keep is not None:
            messages = messages.filter(
                pl.col("src").is_in(keep)
                & pl.col("dst").cast(pl.UInt32).is_in(keep)
            )
        for row in messages.head(max_ops).iter_rows(named=True):
            common = {
                "name": "message",
                "cat": "message",
                "id": int(row["send"]),
            }
            events.append(
                {
                    **common,
                    "ph": "s",
                    "pid": row["src"],
                    "tid": _MESSAGE_TRACK,
                    "ts": row["start_ns"] / 1000.0,
                }
            )
            events.append(
                {
                    **common,
                    "ph": "f",
                    "bp": "e",
                    "pid": int(row["dst"]),
                    "tid": _MESSAGE_TRACK,
                    "ts": row["done_ns"] / 1000.0,
                }
            )
    events.append(
        {
            "ph": "M",
            "name": "process_name",
            "pid": _CRITICAL_PID,
            "args": {"name": "critical path"},
        }
    )
    for row in result.critical_path.iter_rows(named=True):
        events.append(
            {
                "ph": "X",
                "name": row["segment"],
                "cat": "critical_path",
                "pid": _CRITICAL_PID,
                "tid": 0,
                "ts": row["start_ns"] / 1000.0,
                "dur": row["duration_ns"] / 1000.0,
                "args": {
                    "op": row["op"],
                    "rank": row["rank"],
                    "name": row["name"],
                },
            }
        )
    return events


def write_chrome_trace(
    result: result_lib.SimResult,
    path: str | pathlib.Path,
    **options: Any,
) -> int:
    """Writes the run's trace events to ``path``; returns how many.

    Accepts the options of ``chrome_events``.
    """
    events = chrome_events(result, **options)
    pathlib.Path(path).write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "ns"})
    )
    return len(events)
