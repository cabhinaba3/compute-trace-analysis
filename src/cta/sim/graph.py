"""Builds the engine's graph from ingested GOAL or Chakra tables.

Ops are sorted by (rank, node id), so each rank keeps its trace order. A
collective without a ``pg_name`` spans every rank; one with a ``pg_name``
takes its members from the groups file ({pg_name: [ranks]}, ASTRA-sim's
format) or else from the ranks that run it, in rank order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any, cast

import numpy as np
import polars as pl

from cta import plugins
from cta.ingest import cache
from cta.ingest import discovery
from cta.sim import config as config_lib

GRAPH_FORMATS = ("goal", "chakra", "chakra_host")
KIND_CALC, KIND_SEND, KIND_RECV, KIND_COLLECTIVE, KIND_NOP = range(5)
KIND_NAMES = ("calc", "send", "recv", "collective", "nop")
ISSUES_SCHEMA = pl.Schema(
    {
        "severity": pl.String(),  # error, warning or info
        "check": pl.String(),
        "count": pl.Int64(),
        "detail": pl.String(),
    }
)


@dataclasses.dataclass(slots=True)
class SimGraph:
    """A graph ready for the engine.

    ``nodes`` has one row per op (``op``, ``rank``, ``node_id``, ``name``,
    ``comm_type``, ``pg_name``, ``kind``, ``recorded_ns``, ``size_bytes``,
    ``num_ops``); ``columns`` are the engine's arrays; group g's members are
    ``group_members[group_offsets[g]:group_offsets[g + 1]]``.
    """

    label: str
    trace_format: str
    world_size: int
    nodes: pl.DataFrame
    columns: dict[str, np.ndarray]
    group_names: tuple[str, ...]
    group_offsets: np.ndarray
    group_members: np.ndarray
    issues: pl.DataFrame

    @property
    def has_collectives(self) -> bool:
        """Whether any op is a collective."""
        return bool(np.any(self.columns["kind"] == KIND_COLLECTIVE))

    def group(self, name: str) -> list[int]:
        """The members of a group."""
        g = self.group_names.index(name)
        start, end = self.group_offsets[g], self.group_offsets[g + 1]
        return self.group_members[start:end].tolist()


def read_groups(path: str | pathlib.Path) -> dict[str, list[int]]:
    """Reads a JSON object of group name -> list of ranks."""
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, dict) or not all(
        isinstance(v, list) and all(isinstance(r, int) for r in v)
        for v in data.values()
    ):
        raise ValueError(f"{path}: expected an object of name -> list of ranks")
    return {str(k): v for k, v in data.items()}


def load_graph(
    source: str | pathlib.Path,
    *,
    trace_format: str | None = None,
    label: str | None = None,
    groups: Mapping[str, Sequence[int]] | str | pathlib.Path | None = None,
    world_size: int = 0,
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> SimGraph:
    """Ingests a GOAL or Chakra trace through the cache and builds its graph.

    Raises:
        ValueError: The source holds no GOAL or Chakra trace, or several and
            ``trace_format`` does not pick one.
    """
    bundle = discovery.resolve_bundle(source)
    found = [f for f in GRAPH_FORMATS if f in bundle.artifacts]
    if trace_format in found:
        found = [trace_format]
    if len(found) != 1:
        raise ValueError(
            f"{source}: expected one of {', '.join(GRAPH_FORMATS)} "
            f"(found {', '.join(sorted(bundle.formats)) or 'nothing'})"
        )
    artifact = bundle.artifacts[found[0]]
    tables = cache.CacheStore(pathlib.Path(cache_dir)).get_or_ingest(
        artifact, plugins.ADAPTERS[found[0]]
    )
    return from_tables(
        pl.read_parquet(tables["nodes"]),
        pl.read_parquet(tables["edges"]),
        trace_format=found[0],
        label=label or bundle.label,
        groups=groups,
        world_size=max(world_size, artifact.world_size),
    )


def issue(severity: str, check: str, count: int, detail: str) -> dict[str, Any]:
    """A row of ISSUES_SCHEMA."""
    return {
        "severity": severity,
        "check": check,
        "count": count,
        "detail": detail,
    }


def from_tables(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    trace_format: str,
    label: str,
    groups: Mapping[str, Sequence[int]] | str | pathlib.Path | None = None,
    world_size: int = 0,
) -> SimGraph:
    """Builds a graph from ``nodes`` and ``edges`` tables.

    Sends and recvs without a peer become empty compute, collectives of an
    unknown type compute of their recorded length, and edges to missing
    nodes are dropped; ``issues`` reports each.

    Raises:
        ValueError: A rank has two nodes with one id, or runs a collective of
            a group the groups file does not give it.
    """
    for column, value in (("pg_name", ""), ("cpu", 0), ("nic", 0)):
        if column not in nodes.columns:
            nodes = nodes.with_columns(pl.lit(value).alias(column))
    nodes = nodes.sort("rank", "node_id").with_row_index("op")
    if nodes.select(pl.struct("rank", "node_id").is_duplicated().any()).item():
        raise ValueError(f"{label}: two nodes of one rank share a node id")
    op_class = pl.col("op_class")
    p2p = op_class.is_in(["comm_send", "comm_recv"])
    nodes = nodes.with_columns(
        pl.col("comm_type")
        .replace_strict(
            {t: i for i, t in enumerate(config_lib.COLLECTIVE_TYPES)},
            default=-1,
            return_dtype=pl.Int8,
        )
        .alias("coll_type"),
        pl.when(op_class == "comm_send")
        .then(pl.col("dst_rank"))
        .when(op_class == "comm_recv")
        .then(pl.col("src_rank"))
        .otherwise(-1)
        .cast(pl.Int32)
        .alias("peer"),
    ).with_columns(
        pl.when(p2p & (pl.col("peer") >= 0))
        .then(
            pl.when(op_class == "comm_send")
            .then(KIND_SEND)
            .otherwise(KIND_RECV)
        )
        .when((op_class == "comm_coll") & (pl.col("coll_type") >= 0))
        .then(KIND_COLLECTIVE)
        .otherwise(KIND_CALC)
        .cast(pl.UInt8)
        .alias("kind"),
        pl.when(p2p & (pl.col("peer") < 0))
        .then(0)
        .otherwise(pl.col("duration_ns"))
        .cast(pl.Int64)
        .alias("recorded_ns"),
    )
    issues = []
    no_peer = nodes.filter(p2p & (pl.col("peer") < 0)).height
    if no_peer:
        issues.append(
            issue(
                "warning", "p2p_without_peer", no_peer, "run as empty compute"
            )
        )
    unknown = nodes.filter(
        (op_class == "comm_coll") & (pl.col("coll_type") < 0)
    )
    if unknown.height:
        types = ", ".join(sorted(set(unknown["comm_type"])))
        issues.append(
            issue(
                "warning",
                "unknown_collective_type",
                unknown.height,
                f"{types} run as compute",
            )
        )

    max_rank = cast(Any, nodes["rank"].max()) or 0
    max_peer = cast(Any, nodes["peer"].max()) or 0
    world = max(world_size, max_rank + 1, max_peer + 1, 1)
    if isinstance(groups, str | pathlib.Path):
        groups = read_groups(groups)
    names, offsets, members, group_issues = _groups(
        nodes, groups or {}, world, label
    )
    issues += group_issues
    index = {name: i for i, name in enumerate(names)}
    nodes = nodes.with_columns(
        pl.when(pl.col("kind") == KIND_COLLECTIVE)
        .then(
            pl.col("pg_name").replace_strict(
                index, default=-1, return_dtype=pl.Int32
            )
        )
        .otherwise(-1)
        .alias("coll_group")
    )

    ids = nodes.select("rank", "node_id", "op")
    joined = edges.join(
        ids.rename({"node_id": "src_id", "op": "src"}), on=["rank", "src_id"]
    ).join(
        ids.rename({"node_id": "dst_id", "op": "dst"}), on=["rank", "dst_id"]
    )
    loops = joined.filter(pl.col("src") == pl.col("dst"))
    joined = joined.filter(pl.col("src") != pl.col("dst"))
    dropped = edges.height - joined.height
    if dropped:
        issues.append(
            issue(
                "warning",
                "dropped_edges",
                dropped,
                f"to missing nodes or ({loops.height}) themselves",
            )
        )
    issues += _message_issues(nodes)

    def array(
        column: str | pl.Expr, frame: pl.DataFrame, dtype: Any
    ) -> np.ndarray:
        return frame.select(column).to_series().to_numpy().astype(dtype)

    columns = {
        "kind": array("kind", nodes, np.uint8),
        "rank": array("rank", nodes, np.uint32),
        "cpu": array("cpu", nodes, np.uint16),
        "nic": array("nic", nodes, np.uint16),
        "duration": array("recorded_ns", nodes, np.int64),
        "bytes": array("size_bytes", nodes, np.uint64),
        "peer": array("peer", nodes, np.int32),
        "tag": array("message_tag", nodes, np.uint64),
        "coll_type": array("coll_type", nodes, np.int8),
        "coll_group": array("coll_group", nodes, np.int32),
        "coll_root": np.zeros(nodes.height, dtype=np.int32),
        "edge_src": array("src", joined, np.uint32),
        "edge_dst": array("dst", joined, np.uint32),
        "edge_kind": array(pl.col("dep_kind") == "irequires", joined, np.uint8),
    }
    return SimGraph(
        label=label,
        trace_format=trace_format,
        world_size=world,
        nodes=nodes.select(
            "op",
            "rank",
            "node_id",
            "name",
            "comm_type",
            "pg_name",
            "kind",
            "recorded_ns",
            "size_bytes",
            "num_ops",
        ),  # fmt: skip
        columns=columns,
        group_names=tuple(names),
        group_offsets=np.asarray(offsets, dtype=np.uint32),
        group_members=np.asarray(members, dtype=np.uint32),
        issues=pl.DataFrame(issues, schema=ISSUES_SCHEMA),
    )


def _groups(
    nodes: pl.DataFrame,
    explicit: Mapping[str, Sequence[int]],
    world: int,
    label: str,
) -> tuple[list[str], list[int], list[int], list[dict[str, Any]]]:
    collectives = nodes.filter(pl.col("kind") == KIND_COLLECTIVE)
    users = dict(
        collectives.group_by("pg_name")
        .agg(pl.col("rank").unique().sort())
        .iter_rows()
    )
    names, offsets, members, inferred = sorted(users), [0], [], []
    for name in names:
        if name == "":
            ranks = list(range(world))
        elif name in explicit:
            ranks = list(explicit[name])
            if outsiders := sorted(set(users[name]) - set(ranks)):
                raise ValueError(
                    f"{label}: ranks {outsiders[:8]} run collectives of group "
                    f"{name!r}, which the groups file does not give them"
                )
        else:
            ranks = users[name]
            inferred.append(name)
        members += ranks
        offsets.append(len(members))
    issues = []
    if inferred:
        issues.append(
            issue(
                "info",
                "groups_inferred",
                len(inferred),
                "members are the ranks using " + ", ".join(inferred[:6]),
            )
        )
    counts = (
        collectives.group_by("pg_name", "rank")
        .len()
        .group_by("pg_name")
        .agg(pl.col("len").n_unique().alias("counts"))
        .filter(pl.col("counts") > 1)
    )
    if counts.height:
        issues.append(
            issue(
                "error",
                "collective_count_mismatch",
                counts.height,
                "members of a group run different numbers of collectives",
            )
        )
    return names, offsets, members, issues


def _message_issues(nodes: pl.DataFrame) -> list[dict[str, Any]]:
    """Reports (src, dst, tag) channels whose sends and recvs do not pair up."""
    send = pl.col("kind") == KIND_SEND
    balance = (
        nodes.filter(pl.col("kind").is_in([KIND_SEND, KIND_RECV]))
        .group_by(
            pl.when(send)
            .then("rank")
            .otherwise("peer")
            .cast(pl.Int64)
            .alias("src"),
            pl.when(send)
            .then("peer")
            .otherwise("rank")
            .cast(pl.Int64)
            .alias("dst"),
            "message_tag",
        )
        .agg(send.sum().alias("sends"), (~send).sum().alias("recvs"))
        .filter(pl.col("sends") != pl.col("recvs"))
    )
    if balance.is_empty():
        return []
    unpaired = int((balance["sends"] - balance["recvs"]).abs().sum())
    return [
        issue(
            "error",
            "unpaired_messages",
            unpaired,
            f"in {balance.height} channels",
        )
    ]
