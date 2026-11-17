"""The result of one simulation run, as Polars tables.

Times are integer nanoseconds from the start of the run; an op that never
reached a state (the run deadlocked, or stopped early) has null there.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, cast

import numpy as np
import polars as pl

from cta.sim import config as config_lib
from cta.sim import graph as graph_lib

# Critical-path categories, in the engine's numbering (sim::Segment).
SEGMENTS = (
    "compute",
    "noise",
    "overhead",
    "latency",
    "transfer",
    "contention",
    "collective",
    "idle",
)
# The categories that are communication.
COMM_SEGMENTS = ("overhead", "latency", "transfer", "contention", "collective")
LINK_KINDS = (
    "nic_up",
    "nic_down",
    "intra_up",
    "intra_down",
    "fabric",
    "global",
)
ROLES = ("original", "entry", "step", "exit")
LEVELS = ("p2p", "lowered", "analytical")

_NEVER = np.iinfo(np.int64).max
_NONE = np.iinfo(np.uint32).max


def _times(values: np.ndarray) -> pl.Expr:
    """Int64 times with the engine's "never" as null."""
    series = pl.Series(values, dtype=pl.Int64)
    return pl.when(series == _NEVER).then(None).otherwise(series)


def _nullable_ids(values: np.ndarray) -> pl.Expr:
    """Op ids with the engine's "none" as null."""
    series = pl.Series(values, dtype=pl.UInt32)
    return pl.when(series == _NONE).then(None).otherwise(series)


def _codes(values: np.ndarray, names: tuple[str, ...]) -> pl.Series:
    """Names of integer codes."""
    return (
        pl.Series(values)
        .cast(pl.Int64)
        .replace_strict(dict(enumerate(names)), return_dtype=pl.String)
    )


@dataclasses.dataclass(slots=True)
class SimResult:
    """Everything one run produced.

    Attributes:
        label: Name of the simulated graph.
        trace_format: Format the graph came from.
        level: ``p2p`` (no collectives), ``lowered`` or ``analytical``.
        config: The settings of the run.
        ops: One row per simulated op (lowered ops included): ``op``,
            ``rank``, ``kind``, ``role``, ``origin`` (the trace op it came
            from), ``action``, ``node_id``, ``name``, ``comm_type``,
            ``cpu``, ``nic``, ``bytes``, ``peer``, ``duration_ns``,
            ``ready_ns``, ``start_ns``, ``end_ns``, ``noise_ns``,
            ``matched``. Empty when the run kept no op table.
        ranks: One row per rank: ``finish_ns``, ``compute_ns``,
            ``noise_ns``, ``overhead_ns`` (CPU time on messages), ``nic_ns``
            (LogGOPS NIC gaps), ``wait_ns`` (ready but queued for a CPU or
            NIC), ``sync_ns`` (waiting for collective peers), ``idle_ns``,
            ``utilization``, ``bytes_sent``, ``bytes_received``,
            ``messages``.
        critical_path: The critical path from 0 to the makespan as
            consecutive segments: ``seq``, ``segment`` (SEGMENTS), ``op``,
            ``rank``, ``kind``, ``name``, ``node_id``, ``start_ns``,
            ``end_ns``, ``duration_ns``.
        messages: One row per message: ``send`` op, ``src``, ``dst``,
            ``bytes``, ``collective`` (whether lowering made it),
            ``start_ns``, ``arrive_ns``, ``done_ns``, ``latency_ns``.
        links: Flow model only: ``link``, ``kind``, ``src``, ``dst``,
            ``capacity_gbs``, ``latency_ns``, ``bytes``, ``busy_ns``,
            ``utilization`` (bytes over capacity times makespan),
            ``max_flows``.
        collectives: One row per collective op and rank: ``origin``,
            ``rank``, ``node_id``, ``name``, ``comm_type``, ``pg_name``,
            ``bytes``, ``ready_ns``, ``start_ns``, ``end_ns``,
            ``duration_ns``.
        issues: Problems found building or running the graph
            (graph.ISSUES_SCHEMA).
        stats: Engine counters (events, wall time, ...).
        lowering: Lowering counters, empty unless level is ``lowered``.
    """

    label: str
    trace_format: str
    level: str
    config: config_lib.SimConfig
    ops: pl.DataFrame
    ranks: pl.DataFrame
    critical_path: pl.DataFrame
    messages: pl.DataFrame
    links: pl.DataFrame
    collectives: pl.DataFrame
    issues: pl.DataFrame
    stats: dict[str, Any]
    lowering: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def name(self) -> str:
        """The run's display name: the label, with the level if relevant."""
        return (
            self.label
            if self.level == "p2p"
            else f"{self.label} [{self.level}]"
        )

    @property
    def makespan_ns(self) -> int:
        """When the last op finished."""
        return int(self.stats["makespan_ns"])

    @property
    def completed(self) -> bool:
        """Whether every op finished."""
        return int(self.stats["finished_ops"]) == int(self.stats["ops"])

    def breakdown(self) -> pl.DataFrame:
        """Returns critical-path time per segment: segment, ns and pct."""
        totals = dict(
            self.critical_path.group_by("segment")
            .agg(pl.col("duration_ns").sum())
            .iter_rows()
        )
        span = max(self.makespan_ns, 1)
        return pl.DataFrame(
            {
                "segment": list(SEGMENTS),
                "ns": [int(totals.get(s, 0)) for s in SEGMENTS],
                "pct": [100.0 * totals.get(s, 0) / span for s in SEGMENTS],
            }
        )

    def summary(self) -> dict[str, float]:
        """Returns the run's headline numbers, flat, for tables and gates."""
        out: dict[str, float] = {
            "makespan_ns": float(self.makespan_ns),
            "makespan_ms": self.makespan_ns / 1e6,
        }
        comm = 0.0
        for row in self.breakdown().iter_rows(named=True):
            out[f"cp_{row['segment']}_ns"] = float(row["ns"])
            out[f"cp_{row['segment']}_pct"] = float(row["pct"])
            if row["segment"] in COMM_SEGMENTS:
                comm += row["ns"]
        out["cp_comm_ns"] = comm
        out["cp_comm_pct"] = 100.0 * comm / max(self.makespan_ns, 1)
        ranks = self.ranks
        if ranks.height:
            finish = ranks["finish_ns"].cast(pl.Float64)
            mean = float(cast(Any, finish.mean()) or 0.0)
            out["rank_finish_min_ns"] = float(cast(Any, finish.min()) or 0.0)
            out["rank_finish_max_ns"] = float(cast(Any, finish.max()) or 0.0)
            out["rank_finish_spread_ns"] = (
                out["rank_finish_max_ns"] - out["rank_finish_min_ns"]
            )
            out["rank_finish_cv"] = (
                float(cast(Any, finish.std()) or 0.0) / mean
                if mean > 0
                else 0.0
            )
            out["mean_utilization"] = float(
                cast(Any, ranks["utilization"].mean()) or 0.0
            )
            for column in (
                "compute_ns",
                "overhead_ns",
                "wait_ns",
                "sync_ns",
                "noise_ns",
                "bytes_sent",
                "messages",
            ):
                out[f"total_{column}"] = float(ranks[column].sum())
        for key in ("ops", "finished_ops", "events", "wall_seconds"):
            out[key] = float(self.stats[key])
        return out

    def write(
        self, out_dir: str | pathlib.Path, *, csv: bool = False
    ) -> list[pathlib.Path]:
        """Writes every non-empty table (Parquet, or CSV) and summary.json.

        Returns:
            The files written.
        """
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[pathlib.Path] = []
        for name in (
            "ops",
            "ranks",
            "critical_path",
            "messages",
            "links",
            "collectives",
            "issues",
        ):
            table: pl.DataFrame = getattr(self, name)
            if table.is_empty():
                continue
            path = out / f"{name}.{'csv' if csv else 'parquet'}"
            if csv:
                table.write_csv(path)
            else:
                table.write_parquet(path)
            written.append(path)
        summary_path = out / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "label": self.label,
                    "level": self.level,
                    "trace_format": self.trace_format,
                    "summary": self.summary(),
                    "stats": self.stats,
                    "lowering": self.lowering,
                    "config": self.config.to_dict(),
                },
                indent=2,
                default=str,
            )
        )
        written.append(summary_path)
        return written


def from_native(
    graph: graph_lib.SimGraph,
    settings: config_lib.SimConfig,
    level: str,
    run: dict[str, np.ndarray],
    raw: dict[str, Any],
    lowering: dict[str, int],
    *,
    keep_ops: bool = True,
) -> SimResult:
    """Builds a SimResult from the engine's output.

    Args:
        graph: The graph before lowering.
        settings: The run's config.
        level: The run's level.
        run: The arrays the engine ran (lowered, if it was).
        raw: What ``cta._native.simulate`` returned.
        lowering: Lowering counters.
        keep_ops: Whether to build the per-op and per-message tables.
    """
    stats = dict(raw["stats"])
    n = len(run["kind"])
    origin = run.get("origin", np.arange(n, dtype=np.uint32))
    nodes = graph.nodes
    names = nodes["name"]
    node_ids = nodes["node_id"]

    ranks = _ranks_table(raw)
    critical_path = _critical_path_table(raw, run, origin, names, node_ids)
    links = _links_table(raw, stats["makespan_ns"])

    ops = pl.DataFrame()
    messages = pl.DataFrame()
    collectives = pl.DataFrame()
    if keep_ops:
        ops = _ops_table(run, raw, origin, nodes)
        messages = _messages_table(raw, run, origin)
        collectives = _collectives_table(ops, level)
    issues = pl.concat(
        [graph.issues, _run_issues(raw, stats, lowering)], how="vertical"
    )
    return SimResult(
        label=graph.label,
        trace_format=graph.trace_format,
        level=level,
        config=settings,
        ops=ops,
        ranks=ranks,
        critical_path=critical_path,
        messages=messages,
        links=links,
        collectives=collectives,
        issues=issues,
        stats=stats,
        lowering=dict(lowering),
    )


def _ranks_table(raw: dict[str, Any]) -> pl.DataFrame:
    ranks = pl.DataFrame(
        {
            "rank": np.arange(len(raw["rank_finish_ns"]), dtype=np.uint32),
            "finish_ns": raw["rank_finish_ns"],
            "compute_ns": raw["rank_compute_ns"],
            "noise_ns": raw["rank_noise_ns"],
            "overhead_ns": raw["rank_overhead_ns"],
            "nic_ns": raw["rank_nic_ns"],
            "wait_ns": raw["rank_wait_ns"],
            "sync_ns": raw["rank_sync_ns"],
            "bytes_sent": raw["rank_bytes_sent"],
            "bytes_received": raw["rank_bytes_received"],
            "messages": raw["rank_messages"],
        }
    )
    busy = pl.col("compute_ns") + pl.col("noise_ns") + pl.col("overhead_ns")
    return ranks.with_columns(
        (pl.col("finish_ns") - busy).clip(lower_bound=0).alias("idle_ns"),
        pl.when(pl.col("finish_ns") > 0)
        .then((busy / pl.col("finish_ns")).clip(upper_bound=1.0))
        .otherwise(0.0)
        .alias("utilization"),
    )


def _critical_path_table(
    raw: dict[str, Any],
    run: dict[str, np.ndarray],
    origin: np.ndarray,
    names: pl.Series,
    node_ids: pl.Series,
) -> pl.DataFrame:
    ops = raw["cp_op"].astype(np.int64)
    source = origin[ops] if len(ops) else np.zeros(0, dtype=np.int64)
    return pl.DataFrame(
        {
            "seq": np.arange(len(ops), dtype=np.uint32),
            "segment": _codes(raw["cp_segment"], SEGMENTS),
            "op": raw["cp_op"],
            "rank": raw["cp_rank"],
            "kind": _codes(run["kind"][ops], graph_lib.KIND_NAMES)
            if len(ops)
            else pl.Series([], dtype=pl.String),
            "name": names.gather(source)
            if len(ops)
            else pl.Series([], dtype=pl.String),
            "node_id": node_ids.gather(source)
            if len(ops)
            else pl.Series([], dtype=pl.UInt64),
            "start_ns": raw["cp_start_ns"],
            "end_ns": raw["cp_end_ns"],
        }
    ).with_columns((pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"))


def _links_table(raw: dict[str, Any], makespan_ns: int) -> pl.DataFrame:
    if len(raw["link_kind"]) == 0:
        return pl.DataFrame()
    span = max(int(makespan_ns), 1)
    return pl.DataFrame(
        {
            "link": np.arange(len(raw["link_kind"]), dtype=np.uint32),
            "kind": _codes(raw["link_kind"], LINK_KINDS),
            "src": raw["link_from"],
            "dst": raw["link_to"],
            "capacity_gbs": raw["link_capacity"],
            "latency_ns": raw["link_latency_ns"],
            "bytes": raw["link_bytes"],
            "busy_ns": raw["link_busy_ns"],
            "max_flows": raw["link_max_flows"],
        }
    ).with_columns(
        (pl.col("bytes") / (pl.col("capacity_gbs") * span)).alias("utilization")
    )


def _ops_table(
    run: dict[str, np.ndarray],
    raw: dict[str, Any],
    origin: np.ndarray,
    nodes: pl.DataFrame,
) -> pl.DataFrame:
    n = len(run["kind"])
    role = run.get("role", np.zeros(n, dtype=np.uint8))
    action = run.get("action", np.full(n, _NONE, dtype=np.uint32))
    ops = pl.DataFrame(
        {
            "op": np.arange(n, dtype=np.uint32),
            "rank": run["rank"],
            "kind": _codes(run["kind"], graph_lib.KIND_NAMES),
            "role": _codes(role, ROLES),
            "origin": origin.astype(np.uint32),
            "cpu": run["cpu"],
            "nic": run["nic"],
            "bytes": run["bytes"],
            "peer": run["peer"],
            "duration_ns": run["duration"],
            "noise_ns": raw["noise_ns"],
        }
    ).with_columns(
        _nullable_ids(action).alias("action"),
        _times(raw["ready_ns"]).alias("ready_ns"),
        _times(raw["start_ns"]).alias("start_ns"),
        _times(raw["end_ns"]).alias("end_ns"),
        _nullable_ids(raw["matched"]).alias("matched"),
    )
    info = nodes.select(
        pl.col("op").alias("origin"), "node_id", "name", "comm_type", "pg_name"
    )
    return ops.join(info, on="origin", how="left", maintain_order="left")


def _messages_table(
    raw: dict[str, Any], run: dict[str, np.ndarray], origin: np.ndarray
) -> pl.DataFrame:
    sends = raw["msg_send"].astype(np.int64)
    n = len(run["kind"])
    role = run.get("role", np.zeros(n, dtype=np.uint8))
    start = (
        raw["start_ns"][sends] if len(sends) else np.zeros(0, dtype=np.int64)
    )
    return pl.DataFrame(
        {
            "send": raw["msg_send"],
            "src": run["rank"][sends],
            "dst": run["peer"][sends],
            "bytes": run["bytes"][sends],
            "collective": role[sends] == 2,
            "origin": origin[sends].astype(np.uint32),
            "start_ns": start,
            "arrive_ns": raw["msg_arrive_ns"],
            "done_ns": raw["msg_done_ns"],
        }
    ).with_columns((pl.col("done_ns") - pl.col("start_ns")).alias("latency_ns"))


def _collectives_table(ops: pl.DataFrame, level: str) -> pl.DataFrame:
    columns = ["origin", "rank", "node_id", "name", "comm_type", "pg_name"]
    if level == "lowered":
        entries = ops.filter(pl.col("role") == "entry").select(
            *columns, "ready_ns", "start_ns"
        )
        exits = ops.filter(pl.col("role") == "exit").select("origin", "end_ns")
        sizes = (
            ops.filter(pl.col("role") == "step")
            .group_by("origin")
            .agg(
                pl.col("bytes")
                .filter(pl.col("kind") == "send")
                .sum()
                .alias("wire_bytes")
            )
        )
        table = entries.join(exits, on="origin", how="left").join(
            sizes, on="origin", how="left"
        )
    else:
        table = ops.filter(pl.col("kind") == "collective").select(
            *columns,
            "ready_ns",
            "start_ns",
            "end_ns",
            pl.lit(None, pl.UInt64).alias("wire_bytes"),
        )
    return table.with_columns(
        (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns")
    ).sort("origin")


def _run_issues(
    raw: dict[str, Any], stats: dict[str, Any], lowering: dict[str, int]
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []

    def add(severity: str, check: str, count: int, detail: str) -> None:
        if count:
            rows.append(
                {
                    "severity": severity,
                    "check": check,
                    "count": int(count),
                    "detail": detail,
                }
            )

    unfinished = int(stats["ops"]) - int(stats["finished_ops"])
    add(
        "error",
        "unfinished_ops",
        unfinished,
        "ops that never finished (deadlock: unmatched messages, incomplete "
        "collectives or a dependency cycle)",
    )
    add(
        "error",
        "dependency_cycle",
        int(stats["unordered_ops"]),
        "ops on or behind a cycle",
    )
    add(
        "error",
        "unmatched_recvs",
        len(raw["unmatched_recvs"]),
        "recvs no message reached",
    )
    add(
        "error",
        "unmatched_sends",
        len(raw["unmatched_sends"]),
        "messages no recv took",
    )
    add(
        "error",
        "incomplete_collectives",
        len(raw["incomplete_instances"]),
        "collective instances some member never reached",
    )
    add(
        "warning",
        "algorithm_fallback",
        int(lowering.get("fallbacks", 0)) + int(stats["collective_fallbacks"]),
        "collectives whose algorithm does not support them ran the type's "
        "default algorithm",
    )
    if stats.get("stopped_early"):
        add("warning", "stopped_early", 1, "the run hit max_events")
    return pl.DataFrame(rows, schema=graph_lib.ISSUES_SCHEMA)
