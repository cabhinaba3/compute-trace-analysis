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

TYPES = config_lib.COLLECTIVE_TYPES
SIZES = (2, 3, 4, 5, 8)


def supported_cases() -> list[tuple[str, str, int, int]]:
    cases = []
    for coll, algorithm, size in itertools.product(
        TYPES, config_lib.ALGORITHMS, SIZES
    ):
        node_size = 2 if algorithm == "hierarchical" and size % 2 == 0 else 1
        if collectives.supported(coll, algorithm, size, node_size):
            cases.append((coll, algorithm, size, node_size))
    return cases


CASES = supported_cases()


def test_every_type_has_algorithms_for_every_size() -> None:
    for coll, size in itertools.product(TYPES, SIZES):
        default = config_lib.DEFAULT_ALGORITHMS[coll]
        assert collectives.supported(coll, default, size), (coll, size)


@pytest.mark.parametrize(("coll", "algorithm", "size", "node_size"), CASES)
def test_every_send_pairs_with_a_recv_of_the_same_size(
    coll: str, algorithm: str, size: int, node_size: int
) -> None:
    actions = collectives.schedule(
        coll,
        algorithm,
        size,
        1_000_003,
        node_size=node_size,
        chunks=3,
        reduce=True,
    )
    sends = actions.filter(pl.col("kind") == "send").select(
        pl.col("member").alias("src"),
        pl.col("peer").alias("dst"),
        "seq",
        "bytes",
    )
    recvs = actions.filter(pl.col("kind") == "recv").select(
        pl.col("peer").alias("src"),
        pl.col("member").alias("dst"),
        "seq",
        "bytes",
    )
    assert sends.height == recvs.height
    paired = sends.join(
        recvs, on=["src", "dst", "seq"], how="full", suffix="_recv"
    )
    assert paired.filter(
        pl.col("bytes").is_null() | pl.col("bytes_recv").is_null()
    ).is_empty()
    assert (paired["bytes"] == paired["bytes_recv"]).all()
    # Dependencies point backwards, so every schedule is acyclic.
    assert all(
        d < a
        for a, deps in actions.select("action", "deps").iter_rows()
        for d in deps
    )


B = 1_000_003


@pytest.mark.parametrize(
    ("coll", "algorithm", "total"),
    [
        ("ALL_REDUCE", "ring", lambda p: 2 * (p - 1) * B),
        ("REDUCE_SCATTER", "ring", lambda p: (p - 1) * B),
        ("ALL_GATHER", "ring", lambda p: (p - 1) * B),
        ("ALL_GATHER", "direct", lambda p: (p - 1) * B),
        ("ALL_TO_ALL", "direct", lambda p: (p - 1) * B),
        ("ALL_TO_ALL", "ring", lambda p: (p - 1) * B),
        ("ALL_REDUCE", "direct", lambda p: p * (p - 1) * B),
        ("BROADCAST", "binomial_tree", lambda p: (p - 1) * B),
        ("BROADCAST", "direct", lambda p: (p - 1) * B),
        ("REDUCE", "binomial_tree", lambda p: (p - 1) * B),
        ("GATHER", "direct", lambda p: B - (B // p + (B % p > 0))),
        ("SCATTER", "direct", lambda p: B - (B // p + (B % p > 0))),
        ("BARRIER", "recursive_doubling", lambda p: 0),
    ],
)
@pytest.mark.parametrize("size", SIZES)
def test_bytes_on_the_wire(
    coll: str, algorithm: str, total: Any, size: int
) -> None:
    actions = collectives.schedule(coll, algorithm, size, B)
    sent = int(actions.filter(pl.col("kind") == "send")["bytes"].sum())
    assert sent == total(size)


@pytest.mark.parametrize(("coll", "algorithm", "size", "node_size"), CASES)
def test_lowered_collective_always_completes(
    coll: str, algorithm: str, size: int, node_size: int, graph_builder: Any
) -> None:
    ops = [
        {
            "kind": 3,
            "rank": r,
            "coll_type": TYPES.index(coll),
            "coll_group": 0,
            "bytes": 100_000,
        }
        for r in range(size)
    ]
    graph = graph_builder(ops)
    offsets = np.array([0, size], dtype=np.uint32)
    members = np.arange(size, dtype=np.uint32)
    settings = config_lib.SimConfig().with_overrides(
        {
            f"collectives.algorithms.{coll}": algorithm,
            "ranks_per_node": node_size,
            "collectives.chunks": 2,
        }
    )
    params = settings.to_native(size)
    lowered, stats = native.lower_collectives(graph, offsets, members, params)
    assert stats["collectives"] == size
    assert stats["fallbacks"] == 0
    raw = native.simulate(lowered, offsets, members, params)
    assert raw["stats"]["finished_ops"] == raw["stats"]["ops"]


AB = config_lib.SimConfig(model="alpha_beta").with_overrides(
    {"alpha_beta.alpha": 5500, "alpha_beta.beta": 6}
)


@pytest.mark.parametrize(
    ("coll", "algorithm", "expected"),
    [
        # 2(P-1) steps of a 1000-byte chunk.
        ("ALL_REDUCE", "ring", [6 * (5500 + 6000)] * 4),
        # Halving: 2000 then 1000 bytes; doubling: 1000 then 2000.
        (
            "ALL_REDUCE",
            "halving_doubling",
            [2 * (5500 + 12000) + 2 * (5500 + 6000)] * 4,
        ),
        # log2 P exchanges of the whole buffer.
        ("ALL_REDUCE", "recursive_doubling", [2 * (5500 + 24000)] * 4),
        # The root's sends finish at once; ranks 1 and 2 get the buffer in one
        # hop, rank 3 in two.
        ("BROADCAST", "binomial_tree", [0, 29500, 29500, 59000]),
        # Every pair at once, 1000 bytes each.
        ("ALL_TO_ALL", "direct", [11500] * 4),
    ],
)
def test_alpha_beta_times_by_hand(
    coll: str, algorithm: str, expected: list[int]
) -> None:
    settings = AB.with_overrides({f"collectives.algorithms.{coll}": algorithm})
    assert collectives.times(coll, 4000, 4, settings) == expected


@pytest.mark.parametrize(
    "algorithm",
    ["ring", "halving_doubling", "double_binary_tree", "binomial_tree"],
)
def test_lowered_and_analytical_agree_without_contention(
    algorithm: str, graph_builder: Any
) -> None:
    # Alpha-beta shares nothing, so a lone collective lowered takes exactly
    # its analytical time.
    size = 6
    ops = [
        {
            "kind": 3,
            "rank": r,
            "coll_type": 0,
            "coll_group": 0,
            "bytes": 600_000,
        }
        for r in range(size)
    ]
    graph = graph_builder(ops)
    offsets = np.array([0, size], dtype=np.uint32)
    members = np.arange(size, dtype=np.uint32)
    params = AB.with_overrides(
        {"collectives.algorithms.ALL_REDUCE": algorithm}
    ).to_native(size)
    analytical = native.simulate(graph, offsets, members, params)["stats"][
        "makespan_ns"
    ]
    lowered, _ = native.lower_collectives(graph, offsets, members, params)
    assert (
        native.simulate(lowered, offsets, members, params)["stats"][
            "makespan_ns"
        ]
        == analytical
    )


@pytest.mark.parametrize("size", SIZES)
def test_binomial_gather_brings_every_chunk_to_the_root_once(size: int) -> None:
    actions = collectives.schedule("GATHER", "binomial_tree", size, B)
    at_root = actions.filter(
        (pl.col("member") == 0) & (pl.col("kind") == "recv")
    )
    assert int(at_root["bytes"].sum()) == B - (B // size + (B % size > 0))


def test_unsupported_algorithms_fall_back_to_the_default() -> None:
    # Recursive-doubling all-gather needs a power of two.
    assert not collectives.supported("ALL_GATHER", "recursive_doubling", 6)
    settings = AB.with_overrides(
        {"collectives.algorithms.ALL_GATHER": "recursive_doubling"}
    )
    ring = AB.with_overrides({"collectives.algorithms.ALL_GATHER": "ring"})
    assert collectives.times(
        "ALL_GATHER", 6000, 6, settings
    ) == collectives.times("ALL_GATHER", 6000, 6, ring)
    with pytest.raises(ValueError, match="does not support"):
        collectives.schedule("ALL_GATHER", "recursive_doubling", 6, 6000)


def test_hierarchical_needs_whole_nodes() -> None:
    assert collectives.supported("ALL_REDUCE", "hierarchical", 8, 4)
    assert not collectives.supported("ALL_REDUCE", "hierarchical", 6, 4)
    table = collectives.compare_algorithms(
        "ALL_REDUCE", 8, 1 << 20, node_size=4
    )
    hier = table.filter(pl.col("algorithm") == "hierarchical").row(
        0, named=True
    )
    assert hier["supported"] and hier["time_ns"] > 0


def test_compare_algorithms_reports_every_algorithm() -> None:
    table = collectives.compare_algorithms("ALL_REDUCE", 8, 1 << 20)
    assert table["algorithm"].to_list() == list(config_lib.ALGORITHMS)
    ring = table.filter(pl.col("algorithm") == "ring").row(0, named=True)
    assert ring["messages"] == 8 * 14
    assert ring["steps"] == 14
    assert ring["wire_bytes"] == 14 * (1 << 20)


def test_double_binary_tree_uses_two_complementary_trees() -> None:
    actions = collectives.schedule("ALL_REDUCE", "double_binary_tree", 8, 8000)
    # Each tree carries half the buffer up and down: every member but the
    # roots sends 4000 bytes up and receives 4000 down per tree.
    sent = (
        actions.filter(pl.col("kind") == "send")
        .group_by("member")
        .agg(pl.col("bytes").sum())
    )
    assert int(sent["bytes"].sum()) == 2 * 2 * 7 * 4000
    with pytest.raises(ValueError, match="unknown collective type"):
        collectives.type_code("ALLREDUCE_X")
