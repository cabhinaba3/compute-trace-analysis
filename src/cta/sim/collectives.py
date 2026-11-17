"""Collective algorithms as lowering runs them: schedules, analytical times."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import polars as pl

from cta import _native as native
from cta.sim import config as config_lib

ACTION_KINDS = ("send", "recv", "reduce")


def type_code(coll_type: str) -> int:
    """Chakra's number of a collective type name."""
    name = coll_type.upper().replace("-", "_")
    if name not in config_lib.COLLECTIVE_TYPES:
        raise ValueError(
            f"unknown collective type {coll_type!r}; use one of "
            + ", ".join(config_lib.COLLECTIVE_TYPES)
        )
    return config_lib.COLLECTIVE_TYPES.index(name)


def supported(
    coll_type: str, algorithm: str, size: int, node_size: int = 1
) -> bool:
    """Whether ``algorithm`` runs ``coll_type`` on ``size`` members."""
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
    all_to_all_per_peer: bool = False,
) -> pl.DataFrame:
    """Every member's actions, one row each.

    Columns: ``member``, ``action``, ``kind``, ``peer``, ``bytes``, ``seq``,
    ``output`` and ``deps`` (the member's actions it waits for).
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
        all_to_all_per_peer,
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
    return (
        pl.DataFrame(
            {k: raw[k] for k in ("member", "action", "peer", "bytes", "seq")}
        )
        .with_columns(
            kind=pl.Series(raw["kind"])
            .cast(pl.Int64)
            .replace_strict(dict(enumerate(ACTION_KINDS))),
            output=pl.Series(raw["output"]).cast(pl.Boolean),
        )
        .join(deps, on=["member", "action"], how="left", maintain_order="left")
        .with_columns(pl.col("deps").fill_null([]))
    )


def times(
    coll_type: str,
    nbytes: int,
    members: int | Sequence[int],
    settings: config_lib.SimConfig | None = None,
    *,
    root: int = 0,
) -> list[int]:
    """Each member's contention-free finish time in ns.

    ``members`` are ranks, or a count of ranks 0..n-1; the algorithm is the
    config's for the type.
    """
    settings = settings or config_lib.SimConfig()
    ranks = list(range(members)) if isinstance(members, int) else list(members)
    params = settings.to_native(max(ranks, default=0) + 1)
    return list(
        native.collective_times(
            type_code(coll_type), nbytes, ranks, params, root
        )
    )


def compare_algorithms(
    coll_type: str,
    size: int,
    nbytes: int,
    settings: config_lib.SimConfig | None = None,
    *,
    algorithms: Sequence[str] | None = None,
    node_size: int = 1,
) -> pl.DataFrame:
    """Sets the algorithms side by side for one collective.

    Columns: ``supported``, ``messages``, ``wire_bytes``,
    ``max_member_bytes`` (the busiest member's) and ``time_ns`` (the slowest
    member's, contention-free).
    """
    settings = settings or config_lib.SimConfig()
    if node_size > 1 and settings["ranks_per_node"] <= 1:
        settings = settings.with_overrides({"ranks_per_node": node_size})
    name = config_lib.COLLECTIVE_TYPES[type_code(coll_type)]
    rows = []
    for algorithm in algorithms or config_lib.ALGORITHMS:
        if not supported(name, algorithm, size, node_size):
            rows.append({"algorithm": algorithm, "supported": False})
            continue
        per_peer = settings["collectives.all_to_all_size"] == "per_peer"
        sends = schedule(
            name,
            algorithm,
            size,
            nbytes,
            node_size=node_size,
            all_to_all_per_peer=per_peer,
        ).filter(pl.col("kind") == "send")
        chosen = settings.with_overrides(
            {f"collectives.algorithms.{name}": algorithm}
        )
        rows.append(
            {
                "algorithm": algorithm,
                "supported": True,
                "messages": sends.height,
                "wire_bytes": int(sends["bytes"].sum()),
                "max_member_bytes": int(
                    cast(
                        Any,
                        sends.group_by("member")
                        .agg(pl.col("bytes").sum())["bytes"]
                        .max(),
                    )
                    or 0
                ),
                "time_ns": max(times(name, nbytes, size, chosen), default=0),
            }
        )
    schema = {"algorithm": pl.String, "supported": pl.Boolean} | dict.fromkeys(
        ("messages", "wire_bytes", "max_member_bytes", "time_ns"), pl.Int64
    )
    return pl.DataFrame(rows, schema=schema)
