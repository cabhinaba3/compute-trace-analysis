"""Collective algorithms: who sends what to whom, and how long it takes.

The schedules come from the native builder that lowering uses, so what this
module shows is exactly what a lowered collective runs. ``compare_algorithms``
sets the algorithms side by side for one collective: messages, bytes on the
wire, the busiest member's bytes, and the analytical (contention-free) time
under a config's network model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import polars as pl

from cta import _native as native
from cta.sim import config as config_lib

ACTION_KINDS = ("send", "recv", "reduce")


def type_code(coll_type: str) -> int:
    """Returns Chakra's number for a collective type name (case-insensitive).

    Raises:
        ValueError: An unknown type.
    """
    name = coll_type.upper().replace("-", "_")
    if name not in config_lib.COLLECTIVE_TYPES:
        raise ValueError(
            f"unknown collective type {coll_type!r}; use one of "
            f"{', '.join(config_lib.COLLECTIVE_TYPES)}"
        )
    return config_lib.COLLECTIVE_TYPES.index(name)


def supported(
    coll_type: str, algorithm: str, size: int, node_size: int = 1
) -> bool:
    """Whether ``algorithm`` can run ``coll_type`` on ``size`` members."""
    return bool(
        native.algorithm_supported(
            type_code(coll_type), algorithm, size, node_size
        )
    )


def schedule(
    coll_type: str,
    algorithm: str,
    size: int,
    nbytes: int,
    *,
    root: int = 0,
    node_size: int = 1,
    chunks: int = 1,
    reduce: bool = False,
) -> pl.DataFrame:
    """Returns every member's actions, one row each.

    Columns: ``member``, ``action``, ``kind`` (send, recv, reduce), ``peer``,
    ``bytes``, ``seq``, ``output`` and ``deps`` (the member's actions this
    one waits for).

    Raises:
        ValueError: The algorithm does not support the collective.
    """
    raw = native.collective_schedule(
        type_code(coll_type),
        algorithm,
        size,
        nbytes,
        root,
        node_size,
        chunks,
        reduce,
    )
    actions = pl.DataFrame(
        {
            "member": raw["member"],
            "action": raw["action"],
            "kind": pl.Series(raw["kind"])
            .cast(pl.Int64)
            .replace_strict(
                dict(enumerate(ACTION_KINDS)), return_dtype=pl.String
            ),
            "peer": raw["peer"],
            "bytes": raw["bytes"],
            "seq": raw["seq"],
            "output": pl.Series(raw["output"]).cast(pl.Boolean),
        }
    )
    deps = (
        pl.DataFrame(
            {
                "member": raw["dep_member"],
                "action": raw["dep_action"],
                "dep": raw["dep_on"],
            }
        )
        .group_by("member", "action")
        .agg(pl.col("dep").sort().alias("deps"))
    )
    return actions.join(deps, on=["member", "action"], how="left").with_columns(
        pl.col("deps").fill_null(pl.lit([], dtype=pl.List(pl.UInt32)))
    )


def times(
    coll_type: str,
    nbytes: int,
    members: int | Sequence[int],
    settings: config_lib.SimConfig | None = None,
    *,
    root: int = 0,
) -> list[int]:
    """Returns each member's analytical finish time in ns, starting at 0.

    The algorithm is the config's for the type (falling back to the type's
    default when it does not support the group).

    Args:
        coll_type: The collective type.
        nbytes: Its size (Chakra's comm_size).
        members: The group's ranks, or a count (ranks 0..n-1).
        settings: The config; None uses the defaults.
        root: Position of the root, for rooted collectives.
    """
    settings = settings or config_lib.SimConfig()
    ranks = list(range(members)) if isinstance(members, int) else list(members)
    params = settings.to_native(max(ranks, default=0) + 1)
    return [
        int(t)
        for t in native.collective_times(
            type_code(coll_type), nbytes, ranks, params, root
        )
    ]


def compare_algorithms(
    coll_type: str,
    size: int,
    nbytes: int,
    settings: config_lib.SimConfig | None = None,
    *,
    algorithms: Sequence[str] | None = None,
    node_size: int = 1,
) -> pl.DataFrame:
    """Sets algorithms side by side for one collective.

    Returns:
        One row per algorithm: ``algorithm``, ``supported``, ``steps`` (the
        longest chain of a member's actions), ``messages``, ``wire_bytes``,
        ``max_member_bytes`` (most bytes one member sends) and
        ``time_ns`` (slowest member, analytical); unsupported algorithms
        have nulls.
    """
    settings = settings or config_lib.SimConfig()
    if node_size > 1 and settings.ranks_per_node <= 1:
        settings = settings.with_overrides({"ranks_per_node": node_size})
    type_name = config_lib.COLLECTIVE_TYPES[type_code(coll_type)]
    rows = []
    for algorithm in algorithms or config_lib.ALGORITHMS:
        row: dict[str, object] = {"algorithm": algorithm}
        if not supported(type_name, algorithm, size, node_size):
            row["supported"] = False
            rows.append(row)
            continue
        actions = schedule(
            type_name, algorithm, size, nbytes, node_size=node_size
        )
        sends = actions.filter(pl.col("kind") == "send")
        row["supported"] = True
        row["steps"] = _longest_chain(actions)
        row["messages"] = sends.height
        row["wire_bytes"] = int(sends["bytes"].sum())
        per_member = sends.group_by("member").agg(pl.col("bytes").sum())
        row["max_member_bytes"] = int(cast(Any, per_member["bytes"].max()) or 0)
        chosen = settings.with_overrides(
            {f"collectives.algorithms.{type_name}": algorithm}
        )
        row["time_ns"] = max(times(type_name, nbytes, size, chosen), default=0)
        rows.append(row)
    schema = {
        "algorithm": pl.String,
        "supported": pl.Boolean,
        "steps": pl.Int64,
        "messages": pl.Int64,
        "wire_bytes": pl.Int64,
        "max_member_bytes": pl.Int64,
        "time_ns": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def _longest_chain(actions: pl.DataFrame) -> int:
    """The most actions any member must run one after another."""
    longest = 0
    for _, member in actions.group_by("member"):
        depth: dict[int, int] = {}
        for action, deps in member.select("action", "deps").iter_rows():
            depth[action] = 1 + max((depth[d] for d in deps), default=0)
        longest = max(longest, max(depth.values(), default=0))
    return longest
