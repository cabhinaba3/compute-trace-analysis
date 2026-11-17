"""Tests of the native engine: against the Python oracle and by hand."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from cta import _native as native
from cta.sim import config as config_lib
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


def times(raw: dict[str, Any], name: str) -> list[int | None]:
    return [None if t == NEVER else int(t) for t in raw[name]]


def segment_name(code: int) -> str:
    return result.SEGMENTS[code]


def random_graph(
    seed: int, ranks: int, ops_per_rank: int, messages: int
) -> list[dict[str, Any]]:
    """Random calcs and matched sends and recvs, deps on earlier ops only.

    Some messages exceed S; ops use two CPUs and NICs; tags repeat.
    """
    rng = random.Random(seed)
    per_rank: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
    for r in range(ranks):
        for _ in range(ops_per_rank):
            per_rank[r].append(
                {
                    "kind": CALC,
                    "rank": r,
                    "duration": rng.randrange(0, 5000),
                    "cpu": rng.randrange(2),
                }
            )
    for _ in range(messages):
        src, dst = rng.sample(range(ranks), 2)
        size = rng.choice([0, 1, 100, 4096, 70000, 200000])
        tag = rng.randrange(3)
        nic = rng.randrange(2)
        cpu = rng.randrange(2)
        per_rank[src].append(
            {
                "kind": SEND,
                "rank": src,
                "peer": dst,
                "bytes": size,
                "tag": tag,
                "cpu": cpu,
                "nic": nic,
            }
        )
        per_rank[dst].append(
            {
                "kind": RECV,
                "rank": dst,
                "peer": src,
                "bytes": size,
                "tag": tag,
                "cpu": cpu,
                "nic": nic,
            }
        )
    for r in range(ranks):
        rng.shuffle(per_rank[r])
        if rng.random() < 0.5:
            per_rank[r].append({"kind": NOP, "rank": r})
    return [op for ops in per_rank for op in ops]


def random_edges(
    ops: list[dict[str, Any]], seed: int
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    edges = []
    first_of_rank: dict[int, int] = {}
    for i, op in enumerate(ops):
        first = first_of_rank.setdefault(op["rank"], i)
        for _ in range(rng.randrange(3)):
            if i > first:
                edges.append(
                    (rng.randrange(first, i), i, 1 if rng.random() < 0.2 else 0)
                )
    return edges


PARAM_SETS = [
    {"model": "loggops"},
    {"model": "loggops", "rendezvous": "rts_cts"},
    {
        "model": "loggops",
        "L": 700,
        "o": 120,
        "g": 80,
        "G": 0.37,
        "O": 0.05,
        "S": 8192,
    },
    {
        "model": "loggops",
        "ranks_per_node": 2,
        "intra_L": 100,
        "intra_G": 0.5,
        "nic_scale": [1.0, 0.5, 1.0, 2.0],
    },
    {"model": "alpha_beta", "alpha": 900, "beta": 0.25},
    {"model": "alpha_beta", "ranks_per_node": 2, "intra_alpha": 50},
]


@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("seed", range(6))
def test_engine_matches_oracle_on_random_graphs(
    params: dict[str, Any], seed: int, oracle: Any, graph_builder: Any
) -> None:
    ops = random_graph(seed, ranks=4, ops_per_rank=6, messages=10)
    graph = graph_builder(ops, random_edges(ops, seed + 100))
    raw = run(graph, params)
    expected = oracle(graph, *NO_GROUPS, params).run()
    for name in ("ready_ns", "start_ns", "end_ns"):
        assert times(raw, name) == expected[name], name


def test_engine_matches_oracle_on_recorded_collectives(
    oracle: Any, graph_builder: Any
) -> None:
    # Two groups; the k-th collective of a group on each member is one instance.
    ops: list[dict[str, Any]] = []
    edges: list[tuple[int, ...]] = []
    for r in range(4):
        base = len(ops)
        ops.append({"kind": CALC, "rank": r, "duration": 1000 * (r + 1)})
        ops.append(
            {
                "kind": COLL,
                "rank": r,
                "coll_type": 0,
                "coll_group": 0,
                "duration": 500 + 100 * r,
                "bytes": 64,
            }
        )
        ops.append(
            {
                "kind": COLL,
                "rank": r,
                "coll_type": 2,
                "coll_group": 1 + r % 2,
                "duration": 300 + r,
                "bytes": 64,
            }
        )
        edges += [(base, base + 1), (base + 1, base + 2, 1)]
    groups = (
        np.array([0, 4, 6, 8], dtype=np.uint32),
        np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=np.uint32),
    )
    graph = graph_builder(ops, edges)
    params = {"model": "loggops", "collective_cost": "recorded"}
    raw = run(graph, params, groups)
    expected = oracle(graph, *groups, params).run()
    for name in ("ready_ns", "start_ns", "end_ns"):
        assert times(raw, name) == expected[name]
    # The all-reduce starts when the slowest member (rank 3, 4 us) is ready
    # and lasts the shortest recorded time.
    assert [times(raw, "start_ns")[1 + 3 * r] for r in range(4)] == [4000] * 4
    assert [times(raw, "end_ns")[1 + 3 * r] for r in range(4)] == [4500] * 4


def test_loggops_single_eager_message_by_hand(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000},
        ]
    )
    raw = run(graph, {"model": "loggops"})
    # Send: o = 1500 on the CPU. First byte at o + L = 4000; the receiver
    # then spends o + (s-1)G = 1500 + 999 * 6.
    assert times(raw, "end_ns") == [1500, 4000 + 1500 + 5994]
    segments = [segment_name(s) for s in raw["cp_segment"]]
    assert segments == ["overhead", "latency", "overhead", "transfer"]
    assert list(raw["cp_start_ns"]) == [0, 1500, 4000, 5500]
    assert list(raw["cp_end_ns"]) == [1500, 4000, 5500, 11494]


def test_loggops_nic_gap_serialises_back_to_back_sends(
    graph_builder: Any,
) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000},
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000, "tag": 1},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000, "tag": 1},
        ]
    )
    raw = run(graph, {"model": "loggops"})
    # The NIC is busy g + (s-1)G = 1000 + 5994 after the first send.
    assert times(raw, "start_ns")[:2] == [0, 6994]


def test_rendezvous_send_finishes_when_matched(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 100_000},
            {"kind": CALC, "rank": 1, "duration": 10_000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 100_000},
        ],
        [(1, 2)],
    )
    raw = run(graph, {"model": "loggops"})
    end = times(raw, "end_ns")
    # The receiver's CPU is busy until 10 us, so the message is processed
    # then; the sync-rendezvous send completes with the match.
    assert end[0] == 10_000
    assert end[2] == 10_000 + 1500 + 99_999 * 6


def test_rts_cts_adds_a_round_trip_before_the_data(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 100_000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 100_000},
        ]
    )
    raw = run(graph, {"model": "loggops", "rendezvous": "rts_cts"})
    # RTS: o + L = 4000; CTS: o + L = 8000; data sent at 8000, its CPU free
    # at 9500 (the send completes), first byte at 12000, processed by
    # 12000 + 1500 + 99999 * 6.
    assert times(raw, "end_ns") == [9500, 12000 + 1500 + 599_994]


def test_alpha_beta_by_hand(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000},
        ]
    )
    raw = run(graph, {"model": "alpha_beta", "alpha": 5500, "beta": 6})
    assert times(raw, "end_ns") == [0, 5500 + 6000]


def flow_params(**extra: Any) -> dict[str, Any]:
    return {
        "model": "flow",
        "o": 0,
        "O": 0,
        "topology": "switch",
        "link_bandwidth": 1.0,
        "link_latency": 100,
        **extra,
    }


def test_flow_shares_a_link_max_min_fairly(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 2, "bytes": 1000},
            {"kind": SEND, "rank": 1, "peer": 2, "bytes": 1000},
            {"kind": RECV, "rank": 2, "peer": 0, "bytes": 1000},
            {"kind": RECV, "rank": 2, "peer": 1, "bytes": 1000},
        ]
    )
    raw = run(graph, flow_params())
    # Both flows share rank 2's down-link at 0.5 B/ns: 2000 ns, plus two
    # links of 100 ns latency.
    assert times(raw, "end_ns")[2:] == [2200, 2200]
    alone = run(
        graph_builder(
            [
                {"kind": SEND, "rank": 0, "peer": 2, "bytes": 1000},
                {"kind": RECV, "rank": 2, "peer": 0, "bytes": 1000},
            ]
        ),
        flow_params(),
    )
    assert times(alone, "end_ns")[1] == 1200
    segments = {
        segment_name(s): int(e - b)
        for s, b, e in zip(
            raw["cp_segment"], raw["cp_start_ns"], raw["cp_end_ns"], strict=True
        )
    }
    assert segments == {"transfer": 1000, "contention": 1000, "latency": 200}
    assert int(raw["link_max_flows"].max()) == 2


def test_flow_unshared_links_run_at_full_speed(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": 1, "bytes": 1000},
            {"kind": SEND, "rank": 2, "peer": 3, "bytes": 1000},
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 1000},
            {"kind": RECV, "rank": 3, "peer": 2, "bytes": 1000},
        ]
    )
    raw = run(graph, flow_params())
    assert times(raw, "end_ns")[2:] == [1200, 1200]


@pytest.mark.parametrize(
    ("topology", "extra", "dst", "links"),
    [
        ("ring", {}, 2, 4),  # nic, two ring hops, nic
        ("torus", {"dims": [2, 2]}, 3, 4),  # one hop per dimension
        (
            "fat_tree",
            {"hosts_per_leaf": 2, "spines": 2},
            3,
            4,
        ),  # nic, up, down, nic
        ("fat_tree", {"hosts_per_leaf": 2, "spines": 2}, 1, 2),  # same leaf
        (
            "dragonfly",
            {"hosts_per_router": 1, "routers_per_group": 2},
            3,
            4,
        ),  # nic, global, local, nic
        (
            "switch",
            {"ranks_per_node": 2, "intra_bandwidth": 10.0, "intra_latency": 10},
            1,
            2,
        ),
    ],
)
def test_flow_routes_through_the_topology(
    graph_builder: Any,
    topology: str,
    extra: dict[str, Any],
    dst: int,
    links: int,
) -> None:
    graph = graph_builder(
        [
            {"kind": SEND, "rank": 0, "peer": dst, "bytes": 0},
            {"kind": RECV, "rank": dst, "peer": 0, "bytes": 0},
        ]
    )
    raw = run(
        graph, {**flow_params(topology=topology), "world_size": 4, **extra}
    )
    latency = 10 if "intra_latency" in extra else 100
    assert times(raw, "end_ns")[1] == links * latency


def test_irequires_starts_with_its_dependency(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": CALC, "rank": 0, "duration": 1000, "cpu": 0},
            {"kind": CALC, "rank": 0, "duration": 300, "cpu": 1},
            {"kind": CALC, "rank": 0, "duration": 300, "cpu": 0},
        ],
        [(0, 1, 1), (0, 2, 0)],
    )
    raw = run(graph, {"model": "loggops"})
    assert times(raw, "start_ns") == [0, 0, 1000]


def test_a_cpu_runs_one_calc_at_a_time(graph_builder: Any) -> None:
    graph = graph_builder(
        [{"kind": CALC, "rank": 0, "duration": 100}] * 3
        + [{"kind": CALC, "rank": 0, "duration": 100, "cpu": 1}]
    )
    raw = run(graph, {"model": "loggops"})
    assert times(raw, "start_ns") == [0, 100, 200, 0]


def test_unmatched_messages_leave_ops_unfinished(graph_builder: Any) -> None:
    graph = graph_builder(
        [
            {"kind": RECV, "rank": 1, "peer": 0, "bytes": 8},
            {"kind": CALC, "rank": 1, "duration": 5},
        ],
        [(0, 1)],
    )
    raw = run(graph, {"model": "loggops"})
    assert times(raw, "end_ns") == [None, None]
    assert list(raw["unmatched_recvs"]) == [0]
    assert raw["stats"]["finished_ops"] == 0


def test_dependency_cycles_are_counted(graph_builder: Any) -> None:
    graph = graph_builder([{"kind": CALC}, {"kind": CALC}], [(0, 1), (1, 0)])
    raw = run(graph, {"model": "loggops"})
    assert raw["stats"]["unordered_ops"] == 2


@pytest.mark.parametrize("model", ["loggops", "alpha_beta", "flow"])
def test_critical_path_tiles_the_run(model: str, graph_builder: Any) -> None:
    ops = random_graph(7, ranks=4, ops_per_rank=8, messages=12)
    raw = run(
        graph_builder(ops, random_edges(ops, 8)),
        {"model": model, "o": 50, "L": 200},
    )
    starts, ends = list(raw["cp_start_ns"]), list(raw["cp_end_ns"])
    assert starts[0] == 0
    assert ends[-1] == raw["stats"]["makespan_ns"]
    assert all(a == b for a, b in zip(ends[:-1], starts[1:], strict=True))
    assert "idle" not in [segment_name(s) for s in raw["cp_segment"]]


def test_os_noise_is_reproducible_and_only_slows_down(
    graph_builder: Any,
) -> None:
    graph = graph_builder(
        [{"kind": CALC, "rank": r % 2, "duration": 100_000} for r in range(20)]
    )
    quiet = run(graph, {"model": "loggops"})["stats"]["makespan_ns"]
    noisy = {
        "model": "loggops",
        "noise_frequency": 1e-4,
        "noise_duration": 2000.0,
        "seed": 3,
    }
    first = run(graph, noisy)
    assert (
        first["stats"]["makespan_ns"]
        == run(graph, noisy)["stats"]["makespan_ns"]
    )
    assert first["stats"]["makespan_ns"] > quiet
    assert int(first["noise_ns"].sum()) > 0
    other = run(graph, {**noisy, "seed": 4})
    assert other["stats"]["makespan_ns"] != first["stats"]["makespan_ns"]


def test_unknown_parameters_and_bad_graphs_are_rejected(
    graph_builder: Any,
) -> None:
    graph = graph_builder([{"kind": CALC}])
    with pytest.raises(ValueError, match="unknown simulation parameter"):
        run(graph, {"modle": "loggops"})
    with pytest.raises(ValueError, match="unknown network model"):
        run(graph, {"model": "packet"})
    bad = graph_builder([{"kind": SEND, "rank": 0}])
    with pytest.raises(ValueError, match="has no peer"):
        run(bad, {})
    wrong_dtype = dict(graph, rank=graph["rank"].astype(np.int64))
    with pytest.raises(ValueError, match="'rank' must be"):
        run(wrong_dtype, {})


def test_default_parameters_are_loggopsim_defaults() -> None:
    settings = config_lib.SimConfig()
    params = settings.to_native(1)
    assert (
        params["L"],
        params["o"],
        params["g"],
        params["G"],
        params["S"],
    ) == (2500, 1500, 1000, 6, 65535)
