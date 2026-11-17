"""Tests of the DAG solver (`cta.analysis.dag`, backed by native `solve_dag`).

The expected numbers for the random graphs came from the Python solver this
code replaced, with one change: where predecessors tie on finish time and
depth it takes the lowest vertex id, the rule the native solver uses so that
the critical path does not depend on the order of the edge rows.
"""

from __future__ import annotations

from collections.abc import Sequence
import random
import warnings

import polars as pl
import pytest

from cta import _native as native
from cta.analysis import dag

_NOMINAL_COLLECTIVE_NS = (
    500.0 + 2 * 150.0
)  # latency + two overheads, before bytes


def _frames(
    ops: Sequence[tuple[object, ...]], edges: Sequence[tuple[int, int, int]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds the node and edge tables.

    Each op is (rank, node_id, op_class, duration_ns, size_bytes) and, for a
    send or receive, (src_rank, dst_rank, message_tag) after that.
    """
    columns = ("rank", "node_id", "op_class", "duration_ns", "size_bytes")
    channel = ("src_rank", "dst_rank", "message_tag")
    rows = [
        dict(zip(columns, op[:5], strict=True))
        | dict(
            zip(channel, op[5:] if len(op) > 5 else (-1, -1, 0), strict=True)
        )
        for op in ops
    ]
    nodes = pl.DataFrame(
        rows,
        schema={
            "rank": pl.UInt32,
            "node_id": pl.UInt64,
            "op_class": pl.String,
            "duration_ns": pl.Float64,
            "size_bytes": pl.UInt64,
            "src_rank": pl.Int32,
            "dst_rank": pl.Int32,
            "message_tag": pl.UInt32,
        },
    )
    edge_table = pl.DataFrame(
        edges,
        schema={"rank": pl.UInt32, "src_id": pl.UInt64, "dst_id": pl.UInt64},
        orient="row",
    )
    return nodes, edge_table


def _solve(
    ops: Sequence[tuple[object, ...]], edges: Sequence[tuple[int, int, int]]
) -> dag.DagStats:
    return dag.solve_distributed_dag(*_frames(ops, edges))


def test_diamond() -> None:
    stats = _solve(
        [
            (0, 1, "compute", 5, 0),
            (0, 2, "compute", 3, 0),
            (0, 3, "compute", 4, 0),
            (0, 4, "compute", 2, 0),
        ],
        [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 4)],
    )
    assert (stats.num_nodes, stats.num_edges, stats.dag_depth) == (4, 4, 3)
    assert stats.critical_path_ns == 11  # 5 + 4 + 2, through node 3
    assert stats.cp_compute_ns == 11
    assert stats.cp_exposed_comm_ns == 0
    assert stats.mean_slack_ns == pytest.approx(0.25)  # node 2 has 1 ns
    assert stats.zero_slack_ratio == pytest.approx(0.75)


def test_repeated_edges_count_once() -> None:
    stats = _solve(
        [(0, 1, "compute", 5, 0), (0, 2, "compute", 3, 0)], [(0, 1, 2)] * 5
    )
    assert stats.num_edges == 1
    assert stats.critical_path_ns == 8


def test_a_graph_without_edges_or_with_one_node() -> None:
    two = _solve([(0, 1, "compute", 5, 0), (1, 1, "compute", 9, 0)], [])
    assert (two.num_edges, two.dag_depth, two.critical_path_ns) == (0, 1, 9)
    assert two.zero_slack_ratio == pytest.approx(
        0.5
    )  # node 1 has 4 ns of slack
    one = _solve([(0, 1, "compute", 5, 0)], [])
    assert (one.num_nodes, one.critical_path_ns) == (1, 5)
    assert dag.solve_distributed_dag(*_frames([], [])).num_nodes == 0


def test_edges_to_unknown_nodes_are_ignored() -> None:
    stats = _solve(
        [(0, 1, "compute", 5, 0), (0, 2, "compute", 5, 0)],
        [(0, 1, 2), (0, 1, 99), (3, 1, 2)],
    )
    assert stats.num_edges == 1


def test_a_tie_goes_to_the_lowest_vertex_and_not_to_the_edge_order() -> None:
    ops = [
        (0, 1, "compute", 10, 0),
        (0, 2, "comm_coll", 10, 64),
        (0, 3, "compute", 1, 0),
    ]
    for edges in ([(0, 1, 3), (0, 2, 3)], [(0, 2, 3), (0, 1, 3)]):
        stats = _solve(ops, edges)
        assert stats.critical_path_ns == 11
        # Vertex 0 (compute) beats vertex 1 (comm) as the parent of node 3.
        assert (stats.cp_compute_ns, stats.cp_exposed_comm_ns) == (11, 0)
        assert stats.cp_comm_bytes == 0


def test_comm_without_a_duration_gets_the_nominal_cost() -> None:
    stats = _solve(
        [(0, 1, "comm_coll", 0, 4096), (0, 2, "compute", 5, 0)],
        [(0, 1, 2)],
    )
    nominal = _NOMINAL_COLLECTIVE_NS + 4096 * 0.02
    assert stats.cp_exposed_comm_ns == pytest.approx(nominal)
    assert stats.critical_path_ns == pytest.approx(nominal + 5)
    assert stats.cp_comm_bytes == 4096


def test_a_send_and_receive_split_the_nominal_cost() -> None:
    stats = _solve(
        [
            (0, 1, "comm_send", 0, 1000, 0, 1, 7),
            (1, 1, "comm_recv", 0, 1000, 0, 1, 7),
        ],
        [],
    )
    half = 0.5 * (_NOMINAL_COLLECTIVE_NS + 1000 * 0.02)
    assert stats.num_edges == 1  # the send -> recv link
    assert stats.critical_path_ns == pytest.approx(2 * half)


def test_the_kth_send_pairs_with_the_kth_receive_of_a_channel() -> None:
    ops = [
        (0, 1, "comm_send", 5, 8, 0, 1, 3),
        (0, 2, "comm_send", 5, 8, 0, 1, 3),
        (1, 1, "comm_recv", 5, 8, 0, 1, 3),
        (1, 2, "comm_recv", 5, 8, 0, 1, 3),
        (1, 3, "compute", 2, 0),
    ]
    stats = _solve(ops, [(0, 1, 2), (1, 1, 2), (1, 2, 3)])
    assert stats.num_edges == 5
    assert stats.dag_depth == 4  # send 1 -> send 2 -> recv 2 -> compute


@pytest.mark.parametrize(
    "edges", [[(0, 1, 2), (0, 2, 1), (0, 2, 3)], [(0, 1, 1), (0, 1, 2)]]
)
def test_a_cycle_warns_and_still_gives_numbers(
    edges: list[tuple[int, int, int]],
) -> None:
    ops = [
        (0, 1, "compute", 10, 0),
        (0, 2, "compute", 10, 0),
        (0, 3, "compute", 1, 0),
    ]
    with pytest.warns(RuntimeWarning, match="cycle"):
        stats = _solve(ops, edges)
    assert stats.num_nodes == 3
    assert stats.critical_path_ns >= 10


def test_a_graph_without_a_cycle_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _solve([(0, 1, "compute", 1, 0), (0, 2, "compute", 1, 0)], [(0, 1, 2)])


def random_graph(
    seed: int, n: int, ranks: int, integer: bool
) -> tuple[list[tuple[object, ...]], list[tuple[int, int, int]]]:
    """Returns a random layered graph; integer durations make many ties."""
    rng = random.Random(seed)
    ops: list[tuple[object, ...]] = []
    ids: list[list[int]] = [[] for _ in range(ranks)]
    for _ in range(n):
        rank = rng.randrange(ranks)
        node_id = len(ids[rank]) + 1
        ids[rank].append(node_id)
        kind = rng.choice(
            ["compute", "compute", "comm_coll", "comm_send", "comm_recv"]
        )
        duration = (
            float(rng.randrange(1, 6) * 10)
            if integer
            else rng.uniform(1.0, 100.0)
        )
        if kind == "comm_coll" and rng.random() < 0.3:
            duration = 0.0
        size = rng.randrange(0, 5000)
        tag = rng.randrange(0, 3)
        if kind == "comm_send":
            ops.append(
                (
                    rank,
                    node_id,
                    kind,
                    duration,
                    size,
                    rank,
                    (rank + 1) % ranks,
                    tag,
                )
            )
        elif kind == "comm_recv":
            ops.append((rank, node_id, kind, duration, size, -1, rank, tag))
        else:
            ops.append((rank, node_id, kind, duration, size))
    edges = []
    for rank in range(ranks):
        for j in range(1, len(ids[rank])):
            for k in range(j):
                if rng.random() < 3.0 / j:
                    edges.append((rank, ids[rank][k], ids[rank][j]))
    return ops, edges


# (seed, nodes, ranks, integer durations) -> the numbers in DagStats order.
RANDOM_CASES: dict[tuple[int, int, int, bool], tuple[float, ...]] = {
    (1, 2, 1, True): (2, 1, 2, 50.0, 0.0, 50.0, 5198, 0.0, 1.0),
    (2, 30, 3, True): (
        30,
        78,
        9,
        917.4,
        80.0,
        837.4,
        3066,
        497.3399999999999,
        0.23333333333333334,
    ),
    (3, 30, 3, False): (
        30,
        73,
        9,
        1297.2855927028313,
        248.89633434385584,
        1048.3892583589757,
        13091,
        507.00804662458927,
        0.3,
    ),
    (4, 400, 4, True): (
        400,
        1201,
        18,
        4657.0599999999995,
        150.0,
        4507.06,
        32470,
        2834.4568499999996,
        0.0425,
    ),
    (5, 400, 4, False): (
        400,
        1189,
        21,
        3243.605058893414,
        140.6106154693843,
        3102.99444342403,
        29544,
        1262.555948338091,
        0.0375,
    ),
    (6, 3000, 8, True): (
        3000,
        8815,
        25,
        4800.86,
        390.0,
        4410.86,
        24581,
        2536.1087199999993,
        0.008666666666666666,
    ),
    (7, 3000, 8, False): (
        3000,
        8968,
        25,
        7457.5322150969,
        221.88194016908918,
        7235.650274927812,
        35519,
        4275.3229956948935,
        0.007666666666666666,
    ),
}


@pytest.mark.parametrize("case", sorted(RANDOM_CASES))
def test_random_graphs_match_the_python_solver(
    case: tuple[int, int, int, bool],
) -> None:
    stats = _solve(*random_graph(*case))
    expected = RANDOM_CASES[case]
    assert (stats.num_nodes, stats.num_edges, stats.dag_depth) == expected[:3]
    assert stats.cp_comm_bytes == expected[6]
    assert stats.critical_path_ns == pytest.approx(expected[3], rel=1e-9)
    assert stats.cp_compute_ns == pytest.approx(expected[4], rel=1e-9)
    assert stats.cp_exposed_comm_ns == pytest.approx(expected[5], rel=1e-9)
    assert stats.mean_slack_ns == pytest.approx(expected[7], rel=1e-9)
    assert stats.zero_slack_ratio == pytest.approx(expected[8], rel=1e-9)


def test_the_result_does_not_depend_on_the_order_of_the_edge_rows() -> None:
    nodes, edges = _frames(*random_graph(11, 800, 4, True))
    first = dag.solve_distributed_dag(nodes, edges)
    for seed in range(4):
        shuffled = edges.sample(fraction=1.0, shuffle=True, seed=seed)
        assert dag.solve_distributed_dag(nodes, shuffled) == first


def _edge_pairs(
    ops: Sequence[tuple[object, ...]], edges: Sequence[tuple[int, int, int]]
) -> set[tuple[int, int]]:
    nodes, edge_table = _frames(ops, edges)
    sources, destinations = dag._global_edges(
        nodes.with_row_index("vid"), edge_table
    )
    return set(zip(sources.tolist(), destinations.tolist(), strict=True))


def test_node_ids_may_be_sparse_and_huge() -> None:
    ops = [
        (0, 7, "compute", 1, 0),
        (0, 2**63, "compute", 1, 0),
        (1, 2**64 - 1, "compute", 1, 0),
    ]
    assert _edge_pairs(ops, [(0, 7, 2**63), (0, 2**63, 5)]) == {(0, 1)}


def test_a_node_id_on_two_rows_links_every_pair() -> None:
    ops = [
        (0, 1, "compute", 1, 0),
        (0, 2, "compute", 1, 0),
        (0, 1, "compute", 1, 0),
        (0, 2, "compute", 1, 0),
    ]
    # Both rows that hold node 1 depend-link to both rows that hold node 2.
    assert _edge_pairs(ops, [(0, 1, 2)]) == {(0, 1), (0, 3), (2, 1), (2, 3)}


def test_ranks_in_separate_runs_are_still_found() -> None:
    ops = [
        (0, 1, "compute", 1, 0),
        (1, 1, "compute", 1, 0),
        (0, 2, "compute", 1, 0),
        (1, 2, "compute", 1, 0),
    ]
    assert _edge_pairs(ops, [(0, 1, 2), (1, 2, 1)]) == {(0, 2), (3, 1)}


def test_unmatched_sends_and_receives_make_no_link() -> None:
    ops = [
        (0, 1, "comm_send", 1, 8, 0, 1, 3),
        (0, 2, "comm_send", 1, 8, 0, 1, 3),
        (0, 3, "comm_send", 1, 8, 0, 1, 3),
        (1, 1, "comm_recv", 1, 8, 0, 1, 3),
        (1, 2, "comm_recv", 1, 8, 0, 1, 3),
        (1, 3, "comm_recv", 1, 8, 2, 1, 3),  # another channel
    ]
    # Three sends and two receives on one channel pair up twice, in row order.
    assert _edge_pairs(ops, []) == {(0, 3), (1, 4)}


def test_the_edges_do_not_depend_on_the_thread_count() -> None:
    nodes, edges = _frames(*random_graph(21, 2000, 6, True))
    indexed = nodes.with_row_index("vid")

    def build(threads: int) -> tuple[list[int], list[int]]:
        op_class = indexed["op_class"]
        sources, destinations, _ = native.build_dag_edges(
            dag._key(indexed["rank"]),
            dag._key(indexed["node_id"]),
            (op_class == "comm_send").to_numpy(),
            (op_class == "comm_recv").to_numpy(),
            indexed["src_rank"].cast(pl.Int64).to_numpy(),
            indexed["dst_rank"].cast(pl.Int64).to_numpy(),
            indexed["message_tag"].cast(pl.Int64).to_numpy(),
            dag._key(edges["rank"]),
            dag._key(edges["src_id"]),
            dag._key(edges["dst_id"]),
            threads,
        )
        return sources.tolist(), destinations.tolist()

    assert build(1) == build(8)
