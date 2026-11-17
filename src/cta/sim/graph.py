"""Builds the simulator's graph from ingested GOAL or Chakra tables.

The GOAL and Chakra adapters both write ``nodes`` and ``edges`` tables (see
``cta.adapters.chakra.reader``); this module turns them into the flat arrays
the native engine runs, one row per op across all ranks, sorted by
(rank, node id) so each rank's ops keep their trace order.

Collective groups: a collective without a ``pg_name`` spans every rank. One
with a ``pg_name`` takes its members from the groups file when it names that
group ({pg_name: [ranks]}, ASTRA-sim's format), else from the ranks whose
traces run collectives of that group, in rank order.
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

# Op kind codes of the native engine.
KIND_CALC, KIND_SEND, KIND_RECV, KIND_COLLECTIVE, KIND_NOP = range(5)
KIND_NAMES = ("calc", "send", "recv", "collective", "nop")

ISSUES_SCHEMA = pl.Schema(
    {
        "severity": pl.String(),  # "error", "warning" or "info"
        "check": pl.String(),
        "count": pl.Int64(),
        "detail": pl.String(),
    }
)


@dataclasses.dataclass(slots=True)
class SimGraph:
    """An execution graph ready for the engine.

    Attributes:
        label: Name of the graph in results.
        trace_format: ``goal``, ``chakra`` or ``chakra_host``.
        world_size: Number of ranks.
        nodes: One row per op: ``op`` (its index), ``rank``, ``node_id``,
            ``name``, ``op_class``, ``comm_type``, ``pg_name``, ``kind``
            (KIND_NAMES), ``recorded_ns``, ``size_bytes``, ``num_ops``.
        columns: The engine's op and edge arrays (see native/src/sim/graph.hpp).
        group_names: Name of each collective group ("" is all ranks).
        group_offsets: Group g's members are
            ``group_members[group_offsets[g]:group_offsets[g + 1]]``.
        group_members: Ranks of every group, in group order.
        issues: What building the graph found wrong or assumed
            (ISSUES_SCHEMA).
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
    def num_ops(self) -> int:
        """Number of ops."""
        return int(self.nodes.height)

    @property
    def num_edges(self) -> int:
        """Number of dependency edges."""
        return int(len(self.columns["edge_src"]))

    @property
    def has_collectives(self) -> bool:
        """Whether any op is a collective."""
        return bool(np.any(self.columns["kind"] == KIND_COLLECTIVE))

    def group(self, name: str) -> list[int]:
        """Returns the members of the group called ``name``."""
        index = self.group_names.index(name)
        start, end = self.group_offsets[index], self.group_offsets[index + 1]
        return [int(r) for r in self.group_members[start:end]]


def read_groups(path: str | pathlib.Path) -> dict[str, list[int]]:
    """Reads a groups file: a JSON object of group name -> list of ranks.

    Raises:
        ValueError: The file is not such an object.
    """
    data = json.loads(pathlib.Path(path).read_text())
    if not isinstance(data, dict) or not all(
        isinstance(v, list) and all(isinstance(r, int) for r in v)
        for v in data.values()
    ):
        raise ValueError(
            f"{path}: expected a JSON object of group name -> list of ranks"
        )
    return {str(k): [int(r) for r in v] for k, v in data.items()}


def load_graph(
    source: str | pathlib.Path,
    *,
    trace_format: str | None = None,
    label: str | None = None,
    groups: Mapping[str, Sequence[int]] | str | pathlib.Path | None = None,
    world_size: int = 0,
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> SimGraph:
    """Ingests a GOAL or Chakra trace (through the cache) and builds its graph.

    Args:
        source: A trace file, directory, glob or run bundle.
        trace_format: Which format to use when the source holds several.
        label: Name of the graph; defaults to the bundle's label.
        groups: Collective groups, or a JSON file of them (read_groups).
        world_size: Number of ranks, if more than the trace shows.
        cache_dir: Directory of the Parquet ingest cache.

    Returns:
        The graph.

    Raises:
        ValueError: The source holds no GOAL or Chakra trace, or several and
            ``trace_format`` does not pick one.
    """
    bundle = discovery.resolve_bundle(source)
    found = [f for f in GRAPH_FORMATS if f in bundle.artifacts]
    if trace_format is not None:
        if trace_format not in found:
            raise ValueError(
                f"{source} has no {trace_format} trace (found: "
                f"{', '.join(sorted(bundle.formats)) or 'nothing'})"
            )
        chosen = trace_format
    elif len(found) == 1:
        chosen = found[0]
    elif not found:
        raise ValueError(
            f"{source} holds no GOAL or Chakra trace (found: "
            f"{', '.join(sorted(bundle.formats)) or 'nothing'})"
        )
    else:
        raise ValueError(
            f"{source} holds {' and '.join(found)} traces; pick one with "
            "trace_format"
        )
    artifact = bundle.artifacts[chosen]
    store = cache.CacheStore(pathlib.Path(cache_dir))
    tables = store.get_or_ingest(artifact, plugins.ADAPTERS[chosen])
    return from_tables(
        pl.read_parquet(tables["nodes"]),
        pl.read_parquet(tables["edges"]),
        trace_format=chosen,
        label=label or bundle.label,
        groups=groups,
        world_size=max(world_size, artifact.world_size),
    )


def _issue(
    severity: str, check: str, count: int, detail: str
) -> dict[str, object]:
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

    Point-to-point ops without a peer and collectives of unknown type become
    computation (the latter with their recorded duration); edges to missing
    nodes are dropped. Each is reported in ``issues``.

    Raises:
        ValueError: Two nodes of one rank share an id, or a rank runs a
            collective of a group the groups file says it is not in.
    """
    issues: list[dict[str, object]] = []
    for column, value in (("pg_name", ""), ("cpu", 0), ("nic", 0)):
        if column not in nodes.columns:
            nodes = nodes.with_columns(pl.lit(value).alias(column))
    nodes = nodes.sort("rank", "node_id").with_row_index("op")
    if nodes.select(pl.struct("rank", "node_id").is_duplicated().any()).item():
        raise ValueError(f"{label}: two nodes of one rank share a node id")

    coll_code = (
        pl.col("comm_type")
        .replace_strict(
            {name: i for i, name in enumerate(config_lib.COLLECTIVE_TYPES)},
            default=-1,
            return_dtype=pl.Int8,
        )
        .alias("coll_code")
    )
    peer = (
        pl.when(pl.col("op_class") == "comm_send")
        .then(pl.col("dst_rank"))
        .when(pl.col("op_class") == "comm_recv")
        .then(pl.col("src_rank"))
        .otherwise(-1)
        .cast(pl.Int32)
        .alias("peer")
    )
    nodes = nodes.with_columns(coll_code, peer)
    is_p2p = pl.col("op_class").is_in(["comm_send", "comm_recv"])
    nodes = nodes.with_columns(
        pl.when(is_p2p & (pl.col("peer") >= 0))
        .then(
            pl.when(pl.col("op_class") == "comm_send")
            .then(KIND_SEND)
            .otherwise(KIND_RECV)
        )
        .when((pl.col("op_class") == "comm_coll") & (pl.col("coll_code") >= 0))
        .then(KIND_COLLECTIVE)
        .otherwise(KIND_CALC)
        .cast(pl.UInt8)
        .alias("kind"),
        pl.col("duration_ns").cast(pl.Int64).alias("recorded_ns"),
    )
    no_peer = nodes.filter(is_p2p & (pl.col("peer") < 0)).height
    if no_peer:
        issues.append(
            _issue(
                "warning",
                "p2p_without_peer",
                no_peer,
                "sends/recvs with no peer rank run as zero-length compute",
            )
        )
    unknown = nodes.filter(
        (pl.col("op_class") == "comm_coll") & (pl.col("coll_code") < 0)
    )
    if unknown.height:
        types = ", ".join(sorted(set(unknown["comm_type"].to_list())))
        issues.append(
            _issue(
                "warning",
                "unknown_collective_type",
                unknown.height,
                f"collectives of type {types} run as compute with their "
                "recorded duration",
            )
        )
    # A send or recv that became compute has no duration of its own.
    nodes = nodes.with_columns(
        pl.when(is_p2p & (pl.col("kind") == KIND_CALC))
        .then(0)
        .otherwise(pl.col("recorded_ns"))
        .alias("recorded_ns")
    )

    max_rank = int(cast(Any, nodes["rank"].max()) or 0) if nodes.height else 0
    max_peer = int(cast(Any, nodes["peer"].max()) or 0) if nodes.height else 0
    world = max(world_size, max_rank + 1, max_peer + 1, 1)

    explicit = (
        read_groups(groups)
        if isinstance(groups, str | pathlib.Path)
        else dict(groups or {})
    )
    names, offsets, members, group_of, group_issues = _resolve_groups(
        nodes, explicit, world, label
    )
    issues.extend(group_issues)
    nodes = nodes.with_columns(group_of.alias("coll_group"))

    ids = nodes.select("rank", "node_id", "op")
    joined = edges.join(
        ids.rename({"node_id": "src_id", "op": "src"}),
        on=["rank", "src_id"],
        how="inner",
    ).join(
        ids.rename({"node_id": "dst_id", "op": "dst"}),
        on=["rank", "dst_id"],
        how="inner",
    )
    dangling = edges.height - joined.height
    if dangling:
        issues.append(
            _issue(
                "warning",
                "dangling_edges",
                dangling,
                "dependencies on nodes the trace does not hold were dropped",
            )
        )
    loops = joined.filter(pl.col("src") == pl.col("dst")).height
    if loops:
        issues.append(_issue("warning", "self_dependencies", loops, "dropped"))
        joined = joined.filter(pl.col("src") != pl.col("dst"))
    issues.extend(_message_issues(nodes))

    columns: dict[str, np.ndarray] = {
        "kind": nodes["kind"].to_numpy().astype(np.uint8),
        "rank": nodes["rank"].to_numpy().astype(np.uint32),
        "cpu": nodes["cpu"].to_numpy().astype(np.uint16),
        "nic": nodes["nic"].to_numpy().astype(np.uint16),
        "duration": nodes["recorded_ns"].to_numpy().astype(np.int64),
        "bytes": nodes["size_bytes"].to_numpy().astype(np.uint64),
        "peer": nodes["peer"].to_numpy().astype(np.int32),
        "tag": nodes["message_tag"].to_numpy().astype(np.uint64),
        "coll_type": nodes["coll_code"].to_numpy().astype(np.int8),
        "coll_group": nodes["coll_group"].to_numpy().astype(np.int32),
        "coll_root": np.zeros(nodes.height, dtype=np.int32),
        "edge_src": joined["src"].to_numpy().astype(np.uint32),
        "edge_dst": joined["dst"].to_numpy().astype(np.uint32),
        "edge_kind": (joined["dep_kind"] == "irequires")
        .to_numpy()
        .astype(np.uint8),
    }
    kept = nodes.select(
        "op",
        "rank",
        "node_id",
        "name",
        "op_class",
        "comm_type",
        "pg_name",
        pl.col("kind"),
        "recorded_ns",
        "size_bytes",
        "num_ops",
        "peer",
        "cpu",
        "nic",
    )
    return SimGraph(
        label=label,
        trace_format=trace_format,
        world_size=world,
        nodes=kept,
        columns=columns,
        group_names=tuple(names),
        group_offsets=np.asarray(offsets, dtype=np.uint32),
        group_members=np.asarray(members, dtype=np.uint32),
        issues=pl.DataFrame(issues, schema=ISSUES_SCHEMA),
    )


def _resolve_groups(
    nodes: pl.DataFrame,
    explicit: Mapping[str, Sequence[int]],
    world: int,
    label: str,
) -> tuple[list[str], list[int], list[int], pl.Series, list[dict[str, object]]]:
    """Gives every collective a group; returns the groups and their members.

    Returns:
        Group names, offsets, members, each op's group index (-1 for ops
        that are not collectives) and the issues found.
    """
    issues: list[dict[str, object]] = []
    collectives = nodes.filter(pl.col("kind") == KIND_COLLECTIVE)
    used = sorted(set(collectives["pg_name"].to_list()))
    names: list[str] = []
    offsets = [0]
    members: list[int] = []
    inferred = []
    for name in used:
        users = sorted(
            set(collectives.filter(pl.col("pg_name") == name)["rank"].to_list())
        )
        if name == "":
            ranks = list(range(world))
        elif name in explicit:
            ranks = [int(r) for r in explicit[name]]
            outsiders = sorted(set(users) - set(ranks))
            if outsiders:
                raise ValueError(
                    f"{label}: ranks {outsiders[:8]} run collectives of group "
                    f"{name!r}, which the groups file does not give them"
                )
        else:
            ranks = users
            inferred.append(name)
        names.append(name)
        members.extend(ranks)
        offsets.append(len(members))
    if inferred:
        issues.append(
            _issue(
                "info",
                "groups_inferred",
                len(inferred),
                "members of groups not in the groups file are the ranks "
                "that use them: "
                + ", ".join(inferred[:6])
                + (" ..." if len(inferred) > 6 else ""),
            )
        )
    index = {name: i for i, name in enumerate(names)}
    group_of = (
        pl.when(pl.col("kind") == KIND_COLLECTIVE)
        .then(
            pl.col("pg_name").replace_strict(
                index, default=-1, return_dtype=pl.Int32
            )
        )
        .otherwise(-1)
        .cast(pl.Int32)
    )
    counts = (
        collectives.group_by("pg_name", "rank")
        .len()
        .group_by("pg_name")
        .agg(
            pl.col("len").min().alias("least"),
            pl.col("len").max().alias("most"),
        )
        .filter(pl.col("least") != pl.col("most"))
    )
    if counts.height:
        issues.append(
            _issue(
                "error",
                "collective_count_mismatch",
                counts.height,
                "members of groups "
                + ", ".join(repr(n) for n in counts["pg_name"].to_list()[:6])
                + " run different numbers of collectives; some instances "
                "can never complete",
            )
        )
    return names, offsets, members, nodes.select(group_of).to_series(), issues


def _message_issues(nodes: pl.DataFrame) -> list[dict[str, object]]:
    """Reports point-to-point channels whose sends and recvs do not pair up."""
    p2p = nodes.filter(pl.col("kind").is_in([KIND_SEND, KIND_RECV]))
    if p2p.is_empty():
        return []
    channels = p2p.with_columns(
        pl.when(pl.col("kind") == KIND_SEND)
        .then(pl.col("rank").cast(pl.Int64))
        .otherwise(pl.col("peer").cast(pl.Int64))
        .alias("src"),
        pl.when(pl.col("kind") == KIND_SEND)
        .then(pl.col("peer").cast(pl.Int64))
        .otherwise(pl.col("rank").cast(pl.Int64))
        .alias("dst"),
    )
    balance = (
        channels.group_by("src", "dst", "message_tag")
        .agg(
            (pl.col("kind") == KIND_SEND).sum().alias("sends"),
            (pl.col("kind") == KIND_RECV).sum().alias("recvs"),
        )
        .filter(pl.col("sends") != pl.col("recvs"))
    )
    if balance.is_empty():
        return []
    missing = int((balance["sends"] - balance["recvs"]).abs().sum())
    return [
        _issue(
            "error",
            "unpaired_messages",
            missing,
            f"{balance.height} (src, dst, tag) channels have more sends than "
            "recvs or the reverse; those ops never finish",
        )
    ]
