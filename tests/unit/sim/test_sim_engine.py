"""Tests of the native engine, against the Python oracle and by hand."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from cta import _native as native
from cta.sim import result

CALC, SEND, RECV, COLL, NOP = range(5)
NO_GROUPS = (np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32))
NEVER = np.iinfo(np.int64).max


def run(
    graph: dict[str, np.ndarray],
    params: dict[str, Any],
    groups: Any = NO_GROUPS,
) -> dict[str, Any]:
    return native.simulate(graph, groups[0], groups[1], params)


def times(raw: dict[str, Any], name: str = "op_end_ns") -> list[int | None]:
    return [None if t == NEVER else int(t) for t in raw[name]]


def segments(raw: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for code, start, end in zip(
        raw["cp_segment"], raw["cp_start_ns"], raw["cp_end_ns"], strict=True
    ):
        out[result.SEGMENTS[code]] = out.get(result.SEGMENTS[code], 0) + int(
            end - start
        )
    return out


def random_ops(
    seed: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
    """Random calcs and matched messages on 4 ranks.

    Some messages exceed S, tags repeat, ops use two CPUs and NICs, and each
    op depends on earlier ops of its rank.
    """
    rng = random.Random(seed)
    per_rank: list[list[dict[str, Any]]] = [
        [
            {
                "kind": CALC,
                "rank": r,
                "duration": rng.randrange(5000),
                "cpu": rng.randrange(2),
            }
            for _ in range(6)
        ]
        for r in range(4)
    ]
    for _ in range(10):
        src, dst = rng.sample(range(4), 2)
        common = {
            "bytes": rng.choice([0, 1, 4096, 70000, 200000]),
            "tag": rng.randrange(3),
            "cpu": rng.randrange(2),
            "nic": rng.randrange(2),
        }
        per_rank[src].append({"kind": SEND, "rank": src, "peer": dst, **common})
        per_rank[dst].append({"kind": RECV, "rank": dst, "peer": src, **common})
    for ops in per_rank:
        rng.shuffle(ops)
        ops.append({"kind": NOP, "rank": ops[0]["rank"]})
    ops = [op for rank_ops in per_rank for op in rank_ops]
    edges: list[tuple[int, int, int]] = []
    first: dict[int, int] = {}
    for i, op in enumerate(ops):
        start = first.setdefault(op["rank"], i)
        edges += [
            (rng.randrange(start, i), i, int(rng.random() < 0.2))
            for _ in range(rng.randrange(3))
            if i > start
        ]
    return ops, edges


@pytest.mark.parametrize(
    "params",
    [
        {"model": "loggops"},
        {
            "model": "loggops",
            "loggops.L": 700,
            "loggops.o": 120,
            "loggops.g": 80,
            "loggops.G": 0.37,
            "loggops.O": 0.05,
            "loggops.S": 8192,
        },
        {
            "model": "loggops",
            "ranks_per_node": 2,
            "loggops.intra_L": 100,
            "loggops.intra_G": 0.5,
            "noise.nic_scale": [1.0, 0.5, 1.0, 2.0],
        },
        {
            "model": "alpha_beta",
            "alpha_beta.alpha": 900,
            "alpha_beta.beta": 0.25,
        },
        {
            "model": "alpha_beta",
            "ranks_per_node": 2,
            "alpha_beta.intra_alpha": 50,
        },
    ],
)
@pytest.mark.parametrize("seed", range(6))
def test_engine_matches_oracle(
    params: dict[str, Any], seed: int, oracle: Any, graph_builder: Any
) -> None:
    graph = graph_builder(*random_ops(seed))
    raw = run(graph, params)
    expected = oracle(graph, *NO_GROUPS, params).run()
    for name in ("op_ready_ns", "op_start_ns", "op_end_ns"):
        assert times(raw, name) == expected[name]


def test_recorded_collectives_match_oracle(
    oracle: Any, graph_builder: Any
) -> None:
    ops: list[dict[str, Any]] = []
    edges: list[tuple[int, ...]] = []
    for r in range(4):
        edges += [(len(ops), len(ops) + 1), (len(ops) + 1, len(ops) + 2, 1)]
        ops += [
            {"kind": CALC, "rank": r, "duration": 1000 * (r + 1)},
            {
                "kind": COLL,
                "rank": r,
                "coll_type": 0,
                "coll_group": 0,
                "duration": 500 + 100 * r,
            },
            {
                "kind": COLL,
                "rank": r,
                "coll_type": 2,
                "coll_group": 1 + r % 2,
                "duration": 300 + r,
            },
        ]
    groups = (
        np.array([0, 4, 6, 8], dtype=np.uint32),
        np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=np.uint32),
    )
    graph = graph_builder(ops, edges)
    params = {"collectives.cost": "recorded"}
    raw = run(graph, params, groups)
    assert times(raw) == oracle(graph, *groups, params).run()["op_end_ns"]
    # The all-reduce starts with its last member (4 us), for the shortest time.
    assert times(raw)[1::3] == [4500] * 4


def message(nbytes: int, graph_builder: Any) -> dict[str, np.ndarray]:
    return graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": nbytes},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": nbytes},
        ]
    )


def test_loggops_message_by_hand(graph_builder: Any) -> None:
    raw = run(message(1000, graph_builder), {})
    # The send takes o = 1.5 us; the first byte arrives at o + L = 4 us, then
    # the receiver takes o + (s-1)G = 1500 + 999 * 6.
    assert times(raw) == [1500, 11494]
    assert segments(raw) == {
        "overhead": 3000,
        "latency": 2500,
        "transfer": 5994,
    }


def test_alpha_beta_message_by_hand(graph_builder: Any) -> None:
    raw = run(message(1000, graph_builder), {"model": "alpha_beta"})
    assert times(raw) == [0, 5500 + 6000]


def test_nic_gap_serialises_sends(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000},
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000, "tag": 1},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000, "tag": 1},
        ]
    )
    assert times(run(graph, {}), "op_start_ns")[:2] == [0, 1000 + 5994]


def test_rendezvous_send_finishes_when_matched(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 100_000},
            {"kind": CALC, "rank": 1, "duration": 10_000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 100_000},
        ],
        [(1, 2)],
    )
    # The receiver's CPU is busy until 10 us, when it processes the message.
    assert times(run(graph, {})) == [10_000, 10_000, 10_000 + 1500 + 99_999 * 6]


def flow(**extra: Any) -> dict[str, Any]:
    return {
        "model": "flow",
        "loggops.o": 0,
        "topology.link_bandwidth": 1.0,
        "topology.link_latency": 100,
        "world_size": 4,
        **extra,
    }


def test_flow_shares_links_fairly(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 2, "bytes": 1000},
            {"kind": SEND, "rank": 1, "peer": 2, "bytes": 1000},
            {"kind": RECV, "rank": 2, "peer": 0, "bytes": 1000},
            {"kind": RECV, "rank": 2, "peer": 1, "bytes": 1000},
        ]
    )
    raw = run(graph, flow())
    # Both share rank 2's down-link at 0.5 B/ns, then two links of latency.
    assert times(raw)[2:] == [2200, 2200]
    assert segments(raw) == {
        "transfer": 1000,
        "contention": 1000,
        "latency": 200,
    }
    assert times(run(message(1000, graph_builder), flow()))[1] == 1200


@pytest.mark.parametrize(
    ("extra", "dst", "links", "latency"),
    [
        ({"topology.kind": "ring"}, 2, 4, 100),  # nic, two hops, nic
        ({"topology.kind": "torus", "topology.dims": [2, 2]}, 3, 4, 100),
        (
            {
                "topology.kind": "fat_tree",
                "topology.hosts_per_leaf": 2,
                "topology.spines": 2,
            },
            3,
            4,
            100,
        ),
        (
            {
                "topology.kind": "fat_tree",
                "topology.hosts_per_leaf": 2,
                "topology.spines": 2,
            },
            1,
            2,
            100,
        ),
        (
            {
                "topology.kind": "dragonfly",
                "topology.hosts_per_router": 1,
                "topology.routers_per_group": 2,
            },
            3,
            4,
            100,
        ),
        (
            {
                "ranks_per_node": 2,
                "topology.intra_bandwidth": 10.0,
                "topology.intra_latency": 10,
            },
            1,
            2,
            10,
        ),
    ],
)
def test_flow_routes(
    extra: dict[str, Any],
    dst: int,
    links: int,
    latency: int,
    graph_builder: Any,
) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": dst},
            {"kind": RECV, "rank": dst, "peer": 0},
        ]
    )
    assert times(run(graph, flow(**extra)))[1] == links * latency


def test_dependencies_and_cpus(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": CALC, "duration": 1000},
            {
                "kind": CALC,
                "duration": 300,
                "cpu": 1,
            },  # irequires op 0: starts with it
            {"kind": CALC, "duration": 300},  # requires op 0
            {"kind": CALC, "duration": 100},  # independent, same CPU as op 0
        ],
        [(0, 1, 1), (0, 2)],
    )
    assert times(run(graph, {}), "op_start_ns") == [0, 0, 1100, 1000]


def test_deadlocks_and_cycles_are_reported(graph_builder: Any) -> None:
    raw = run(
        graph_builder(
            [{"kind": RECV, "rank": 1, "peer": 0}, {"kind": CALC, "rank": 1}],
            [(0, 1)],
        ),
        {},
    )
    assert times(raw) == [None, None]
    assert list(raw["left_recvs"]) == [0]
    cycle = run(
        graph_builder([{"kind": CALC}, {"kind": CALC}], [(0, 1), (1, 0)]), {}
    )
    assert cycle["stats"]["unordered_ops"] == 2


@pytest.mark.parametrize("model", ["loggops", "alpha_beta", "flow"])
def test_critical_path_tiles_the_run(model: str, graph_builder: Any) -> None:
    raw = run(graph_builder(*random_ops(7)), {"model": model, "loggops.o": 50})
    starts, ends = list(raw["cp_start_ns"]), list(raw["cp_end_ns"])
    assert starts[0] == 0 and ends[-1] == raw["stats"]["makespan_ns"]
    assert ends[:-1] == starts[1:]
    assert "idle" not in segments(raw)


def test_os_noise(graph_builder: Any) -> None:
    graph = graph_builder(
        [{"kind": CALC, "rank": r % 2, "duration": 100_000} for r in range(20)]
    )
    noisy = {"noise.frequency": 1e5, "noise.duration": 2000.0, "seed": 3}
    first = run(graph, noisy)
    assert (
        first["stats"]["makespan_ns"]
        == run(graph, noisy)["stats"]["makespan_ns"]
    )
    assert (
        first["stats"]["makespan_ns"] > run(graph, {})["stats"]["makespan_ns"]
    )
    assert (
        run(graph, {**noisy, "seed": 4})["stats"]["makespan_ns"]
        != first["stats"]["makespan_ns"]
    )


def test_bad_input_is_rejected(graph_builder: Any) -> None:
    graph = graph_builder([{"kind": CALC}])
    with pytest.raises(ValueError, match="unknown simulation parameter"):
        run(graph, {"modle": "loggops"})
    with pytest.raises(ValueError, match="model cannot be"):
        run(graph, {"model": "packet"})
    with pytest.raises(ValueError, match="has no peer"):
        run(graph_builder([{"kind": SEND}]), {})
    with pytest.raises(ValueError, match="'rank' must be"):
        run(dict(graph, rank=graph["rank"].astype(np.int64)), {})
