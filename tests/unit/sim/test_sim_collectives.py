"""Tests of the collective algorithms: schedules, bytes and analytical times."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import polars as pl
import pytest

from cta import _native as native
from cta.sim import collectives
from cta.sim import config as config_lib

SIZES = (2, 3, 4, 5, 8)
B = 1_000_003
CASES = [
    (coll, algorithm, size, node_size)
    for coll, algorithm, size in itertools.product(
        config_lib.COLLECTIVE_TYPES, config_lib.ALGORITHMS, SIZES
    )
    for node_size in [2 if algorithm == "hierarchical" and size % 2 == 0 else 1]
    if collectives.supported(coll, algorithm, size, node_size)
]
ALPHA_BETA = config_lib.SimConfig().with_overrides({"model": "alpha_beta"})


def chunk(size: int) -> int:
    return B // size + (B % size > 0)


@pytest.mark.parametrize(("coll", "algorithm", "size", "node_size"), CASES)
def test_schedules_pair_up_and_complete(
    coll: str, algorithm: str, size: int, node_size: int, graph_builder: Any
) -> None:
    actions = collectives.schedule(
        coll, algorithm, size, B, node_size=node_size, chunks=3, reduce=True
    )
    sends = actions.filter(pl.col("kind") == "send").select(
        src="member", dst="peer", seq="seq", bytes="bytes"
    )
    recvs = actions.filter(pl.col("kind") == "recv").select(
        src="peer", dst="member", seq="seq", bytes="bytes"
    )
    assert sends.sort(pl.all()).equals(recvs.sort(pl.all()))
    assert all(
        d < a
        for a, deps in actions.select("action", "deps").iter_rows()
        for d in deps
    )
    # Lowered on a network, every collective finishes.
    graph = graph_builder(
        [
            {
                "kind": 3,
                "rank": r,
                "coll_type": config_lib.COLLECTIVE_TYPES.index(coll),
                "coll_group": 0,
                "bytes": 100_000,
            }
            for r in range(size)
        ]
    )
    groups = (
        np.array([0, size], dtype=np.uint32),
        np.arange(size, dtype=np.uint32),
    )
    params = (
        config_lib.SimConfig()
        .with_overrides(
            {
                f"collectives.algorithms.{coll}": algorithm,
                "ranks_per_node": node_size,
                "collectives.chunks": 2,
            }
        )
        .to_native(size)
    )
    lowered, stats = native.lower_collectives(graph, *groups, params)
    assert (stats["collectives"], stats["fallbacks"]) == (size, 0)
    raw = native.simulate(lowered, *groups, params)
    assert raw["stats"]["finished_ops"] == raw["stats"]["ops"]


@pytest.mark.parametrize(
    ("coll", "algorithm", "total"),
    [
        ("ALL_REDUCE", "ring", lambda p: 2 * (p - 1) * B),
        ("ALL_REDUCE", "direct", lambda p: p * (p - 1) * B),
        ("REDUCE_SCATTER", "ring", lambda p: (p - 1) * B),
        ("ALL_GATHER", "ring", lambda p: (p - 1) * B),
        ("ALL_GATHER", "direct", lambda p: (p - 1) * B),
        ("ALL_TO_ALL", "direct", lambda p: (p - 1) * B),
        ("ALL_TO_ALL", "ring", lambda p: (p - 1) * B),
        ("BROADCAST", "binomial_tree", lambda p: (p - 1) * B),
        ("REDUCE", "binomial_tree", lambda p: (p - 1) * B),
        ("GATHER", "direct", lambda p: B - chunk(p)),
        ("SCATTER", "direct", lambda p: B - chunk(p)),
        ("BARRIER", "recursive_doubling", lambda p: 0),
    ],
)
@pytest.mark.parametrize("size", SIZES)
def test_bytes_on_the_wire(
    coll: str, algorithm: str, total: Any, size: int
) -> None:
    sends = collectives.schedule(coll, algorithm, size, B).filter(
        pl.col("kind") == "send"
    )
    assert sends["bytes"].sum() == total(size)


def test_binomial_gather_brings_each_chunk_to_the_root_once() -> None:
    actions = collectives.schedule("GATHER", "binomial_tree", 5, B)
    at_root = actions.filter(
        (pl.col("member") == 0) & (pl.col("kind") == "recv")
    )
    assert at_root["bytes"].sum() == B - chunk(5)


def test_all_to_all_sizes_can_be_per_peer() -> None:
    sends = collectives.schedule(
        "ALL_TO_ALL", "direct", 4, 1000, all_to_all_per_peer=True
    ).filter(pl.col("kind") == "send")
    assert sends.height == 12 and (sends["bytes"] == 1000).all()


@pytest.mark.parametrize(
    ("coll", "algorithm", "expected"),
    [
        ("ALL_REDUCE", "ring", [6 * 11500] * 4),  # 2(P-1) steps of 1000 B
        (
            "ALL_REDUCE",
            "halving_doubling",
            [2 * 17500 + 2 * 11500] * 4,
        ),  # 2000, 1000, 1000, 2000 B
        (
            "ALL_REDUCE",
            "recursive_doubling",
            [2 * 29500] * 4,
        ),  # log2 P steps of 4000 B
        ("BROADCAST", "binomial_tree", [0, 29500, 29500, 59000]),
        ("ALL_TO_ALL", "direct", [11500] * 4),
    ],
)
def test_alpha_beta_times_by_hand(
    coll: str, algorithm: str, expected: list[int]
) -> None:
    settings = ALPHA_BETA.with_overrides(
        {f"collectives.algorithms.{coll}": algorithm}
    )
    assert collectives.times(coll, 4000, 4, settings) == expected


@pytest.mark.parametrize(
    "algorithm",
    ["ring", "halving_doubling", "double_binary_tree", "binomial_tree"],
)
def test_lowered_matches_analytical_without_contention(
    algorithm: str, graph_builder: Any
) -> None:
    graph = graph_builder(
        [
            {
                "kind": 3,
                "rank": r,
                "coll_type": 0,
                "coll_group": 0,
                "bytes": 600_000,
            }
            for r in range(6)
        ]
    )
    groups = (np.array([0, 6], dtype=np.uint32), np.arange(6, dtype=np.uint32))
    params = ALPHA_BETA.with_overrides(
        {"collectives.algorithms.ALL_REDUCE": algorithm}
    ).to_native(6)
    lowered, _ = native.lower_collectives(graph, *groups, params)
    assert (
        native.simulate(lowered, *groups, params)["stats"]["makespan_ns"]
        == native.simulate(graph, *groups, params)["stats"]["makespan_ns"]
    )


def test_unsupported_algorithms_fall_back() -> None:
    assert not collectives.supported("ALL_GATHER", "recursive_doubling", 6)
    with pytest.raises(ValueError, match="does not support"):
        collectives.schedule("ALL_GATHER", "recursive_doubling", 6, 6000)
    rd = ALPHA_BETA.with_overrides(
        {"collectives.algorithms.ALL_GATHER": "recursive_doubling"}
    )
    assert collectives.times("ALL_GATHER", 6000, 6, rd) == collectives.times(
        "ALL_GATHER", 6000, 6, ALPHA_BETA
    )


def test_compare_algorithms() -> None:
    table = collectives.compare_algorithms(
        "ALL_REDUCE", 8, 1 << 20, node_size=4
    )
    assert table["algorithm"].to_list() == list(config_lib.ALGORITHMS)
    ring = table.row(0, named=True)
    assert (ring["messages"], ring["wire_bytes"]) == (8 * 14, 14 << 20)
    assert (
        table.filter(pl.col("algorithm") == "hierarchical")["time_ns"].item()
        > 0
    )
    with pytest.raises(ValueError, match="unknown collective type"):
        collectives.type_code("ALLREDUCE_X")
