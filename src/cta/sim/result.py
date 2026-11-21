"""One simulation run's results as Polars tables (times in ns; null: never)."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any, cast

import numpy as np
import polars as pl

from cta.sim import config as config_lib
from cta.sim import graph as graph_lib

# Critical-path categories, in the engine's numbering.
SEGMENTS = (
    "compute", "noise", "overhead", "latency", "transfer", "contention",
    "collective", "idle",
)  # fmt: skip
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
_NEVER = np.iinfo(np.int64).max
_NONE = np.iinfo(np.uint32).max


@dataclasses.dataclass(slots=True)
class SimResult:
    """Tables of one run.

    ``ops`` (empty unless kept): one row per simulated op, lowered ones
    included, with the trace op it came from (``origin``). ``ranks``: where
    each rank's time went. ``critical_path``: consecutive SEGMENTS from 0 to
    the makespan. ``messages``, ``links`` (flow model) and ``collectives``:
    per message, link and collective op. ``issues``: graph.ISSUES_SCHEMA.
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
        """The label, with the level when there is one."""
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
        return bool(self.stats["finished_ops"] == self.stats["ops"])

    def breakdown(self) -> pl.DataFrame:
        """Critical-path time per segment: ``segment``, ``ns``, ``pct``."""
        totals = dict(
            self.critical_path.group_by("segment")
            .agg(pl.col("duration_ns").sum())
            .iter_rows()
        )
        ns = [int(totals.get(s, 0)) for s in SEGMENTS]
        span = max(self.makespan_ns, 1)
        return pl.DataFrame(
            {"segment": SEGMENTS, "ns": ns, "pct": [100 * x / span for x in ns]}
        )

    def summary(self) -> dict[str, float]:
        """Headline numbers, flat."""
        span = max(self.makespan_ns, 1)
        out = {
            "makespan_ns": float(self.makespan_ns),
            "makespan_ms": span / 1e6,
        }
        for segment, ns, pct in self.breakdown().iter_rows():
            out[f"cp_{segment}_ns"], out[f"cp_{segment}_pct"] = float(ns), pct
        out["cp_comm_ns"] = sum(out[f"cp_{s}_ns"] for s in COMM_SEGMENTS)
        out["cp_comm_pct"] = 100 * out["cp_comm_ns"] / span
        finish = self.ranks["finish_ns"].cast(pl.Float64)
        mean = cast(float, finish.mean() or 0.0)
        out["rank_finish_spread_ns"] = cast(float, finish.max() or 0) - cast(
            float, finish.min() or 0
        )
        out["rank_finish_cv"] = (
            cast(float, finish.std() or 0.0) / mean if mean else 0.0
        )
        out["mean_utilization"] = cast(
            float, self.ranks["utilization"].mean() or 0.0
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
            out[f"total_{column}"] = float(self.ranks[column].sum())
        for key in ("ops", "finished_ops", "events", "wall_seconds"):
            out[key] = float(self.stats[key])
        return out

    def tables(self) -> dict[str, pl.DataFrame]:
        """Every table by name."""
        return {
            name: getattr(self, name)
            for name in (
                "ops",
                "ranks",
                "critical_path",
                "messages",
                "links",
                "collectives",
                "issues",
            )
        }

    def write(
        self, out_dir: str | pathlib.Path, *, csv: bool = False
    ) -> list[pathlib.Path]:
        """Writes the non-empty tables and summary.json; returns the files."""
        written = write_tables(self.tables(), out_dir, csv=csv)
        path = pathlib.Path(out_dir) / "summary.json"
        summary = {
            "run": self.name,
            "summary": self.summary(),
            "stats": self.stats,
            "lowering": self.lowering,
            "config": dict(self.config),
        }
        path.write_text(json.dumps(summary, indent=2, default=str))
        return [*written, path]


def write_tables(
    tables: Mapping[str, pl.DataFrame],
    out_dir: str | pathlib.Path,
    *,
    csv: bool = False,
) -> list[pathlib.Path]:
    """Writes the non-empty tables as Parquet (or CSV); returns the files."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        if not table.is_empty():
            path = out / f"{name}.{'csv' if csv else 'parquet'}"
            table.write_csv(path) if csv else table.write_parquet(path)
            written.append(path)
    return written


def _table(raw: Mapping[str, Any], prefix: str) -> pl.DataFrame:
    return pl.DataFrame(
        {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    )


def _names(codes: np.ndarray, names: tuple[str, ...]) -> pl.Series:
    return (
        pl.Series(codes)
        .cast(pl.Int64)
        .replace_strict(dict(enumerate(names)), return_dtype=pl.String)
    )


def _nullable(table: pl.DataFrame) -> pl.DataFrame:
    """Turns the engine's "never" and "none" markers into nulls."""
    return table.with_columns(
        [
            pl.when(pl.col(c) != _NEVER).then(c).alias(c)
            for c in table.columns
            if c.endswith("_ns")
        ]
        + [
            pl.when(pl.col(c) != _NONE).then(c).alias(c)
            for c in ("matched", "action")
            if c in table.columns
        ]
    )


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
    """Builds the tables of the engine's output for the arrays ``run``."""
    n = len(run["kind"])
    origin = run.get("origin", np.arange(n, dtype=np.uint32))
    busy = pl.col("compute_ns") + pl.col("noise_ns") + pl.col("overhead_ns")
    ranks = (
        _table(raw, "rank_")
        .with_row_index("rank")
        .with_columns(
            (pl.col("finish_ns") - busy).clip(lower_bound=0).alias("idle_ns"),
            (busy / pl.col("finish_ns"))
            .fill_nan(0.0)
            .clip(upper_bound=1.0)
            .alias("utilization"),
        )
    )
    on_path = origin[raw["cp_op"]]
    critical_path = _table(raw, "cp_").with_columns(
        segment=_names(raw["cp_segment"], SEGMENTS),
        kind=_names(run["kind"][raw["cp_op"]], graph_lib.KIND_NAMES),
        name=graph.nodes["name"].gather(on_path),
        node_id=graph.nodes["node_id"].gather(on_path),
        duration_ns=pl.col("end_ns") - pl.col("start_ns"),
    )
    links = _table(raw, "link_").with_row_index("link")
    if links.height:
        span = max(int(raw["stats"]["makespan_ns"]), 1)
        links = links.with_columns(
            kind=_names(raw["link_kind"], LINK_KINDS),
            utilization=pl.col("bytes") / (pl.col("capacity_gbs") * span),
        )
    ops = messages = collectives = pl.DataFrame()
    if keep_ops:
        role = run.get("role", np.zeros(n, dtype=np.uint8))
        ops = _nullable(
            pl.DataFrame(
                {
                    "rank": run["rank"],
                    "kind": _names(run["kind"], graph_lib.KIND_NAMES),
                    "role": _names(role, ROLES),
                    "origin": origin,
                    "action": run.get(
                        "action", np.full(n, _NONE, dtype=np.uint32)
                    ),
                    **{c: run[c] for c in ("cpu", "nic", "bytes", "peer")},
                    "duration_ns": run["duration"],
                    **_table(raw, "op_").to_dict(),
                }
            ).with_row_index("op")
        ).join(
            graph.nodes.select(
                pl.col("op").alias("origin"),
                "node_id",
                "name",
                "comm_type",
                "pg_name",
            ),
            on="origin",
            how="left",
            maintain_order="left",
        )
        sends = raw["msg_send"]
        messages = (
            _table(raw, "msg_")
            .with_columns(
                src=run["rank"][sends],
                dst=run["peer"][sends],
                bytes=run["bytes"][sends],
                collective=role[sends] == ROLES.index("step"),
                start_ns=raw["op_start_ns"][sends],
            )
            .with_columns(latency_ns=pl.col("done_ns") - pl.col("start_ns"))
        )
        collectives = _collectives(ops, level)
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
        issues=pl.concat([graph.issues, _run_issues(raw, lowering)]),
        stats=dict(raw["stats"]),
        lowering=lowering,
    )


def _collectives(ops: pl.DataFrame, level: str) -> pl.DataFrame:
    """One row per collective op: its span, and the bytes its lowering sent."""
    columns = [
        "origin",
        "rank",
        "node_id",
        "name",
        "comm_type",
        "pg_name",
        "ready_ns",
        "start_ns",
    ]
    if level == "lowered":
        steps = ops.filter(
            (pl.col("role") == "step") & (pl.col("kind") == "send")
        )
        table = (
            ops.filter(pl.col("role") == "entry")
            .select(columns)
            .join(
                ops.filter(pl.col("role") == "exit").select("origin", "end_ns"),
                on="origin",
            )
            .join(
                steps.group_by("origin").agg(
                    pl.col("bytes").sum().alias("wire_bytes")
                ),
                on="origin",
                how="left",
            )
        )
    else:
        table = ops.filter(pl.col("kind") == "collective").select(
            *columns, "end_ns"
        )
    return table.with_columns(duration_ns=pl.col("end_ns") - pl.col("start_ns"))


def _run_issues(
    raw: Mapping[str, Any], lowering: Mapping[str, int]
) -> pl.DataFrame:
    stats = raw["stats"]
    found = [
        (
            "error",
            "unfinished_ops",
            stats["ops"] - stats["finished_ops"],
            "never finished (deadlock or cycle)",
        ),
        (
            "error",
            "dependency_cycle",
            stats["unordered_ops"],
            "ops on or behind a cycle",
        ),
        (
            "error",
            "unmatched_recvs",
            len(raw["left_recvs"]),
            "recvs no message reached",
        ),
        (
            "error",
            "unmatched_sends",
            len(raw["left_sends"]),
            "messages no recv took",
        ),
        (
            "error",
            "incomplete_collectives",
            len(raw["left_collectives"]),
            "instances a member never reached",
        ),
        (
            "warning",
            "algorithm_fallback",
            lowering.get("fallbacks", 0) + stats["collective_fallbacks"],
            "collectives ran their type's default algorithm",
        ),
        (
            "warning",
            "stopped_early",
            int(stats["stopped_early"]),
            "hit max_events",
        ),
    ]
    return pl.DataFrame(
        [graph_lib.issue(*row) for row in found if row[2]],
        schema=graph_lib.ISSUES_SCHEMA,
    )
