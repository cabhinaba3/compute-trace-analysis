"""Tests of the alpha-beta interconnect model and what-if repricing.

The cost of each collective is checked against its closed form, the default
interconnect against the weights the solver always used, and the fit against
times made from a known link. The real MPI captures check the fit on
measured times, where a single alpha-beta line is known to be rough.
"""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest
from typer import testing

import cta
from cta.analysis import interconnect as ic

MIB = 1 << 20
cli_runner = testing.CliRunner()

# 1 us and 100 GB/s inside a node, 5 us and 25 GB/s between nodes.
NET = ic.Interconnect(
    gpus_per_node=8,
    intra=ic.Link(1000.0, 0.01),
    inter=ic.Link(5000.0, 0.04),
    algorithm="ring",
)


def _cost(
    kind: str, size: int, ranks: int, net: ic.Interconnect = NET
) -> float:
    return float(
        ic.collective_cost_ns(
            np.array([kind]), np.array([size]), np.array([ranks]), net
        )[0]
    )


def test_parse_reads_units_and_keeps_what_is_not_given() -> None:
    net = ic.parse(
        "gpus_per_node=4, intra=500ns:300GB/s, inter=2.5us:400Gbps, "
        "algo=Tree, group.ALL_REDUCE=64, group.*=8"
    )

    assert net.gpus_per_node == 4
    assert net.intra == ic.Link(500.0, 1 / 300)
    assert net.inter.alpha_ns == 2500.0
    assert net.inter.gb_per_s == pytest.approx(50.0)
    assert net.algorithm == "tree"
    assert net.group_sizes == {"ALL_REDUCE": 64, "*": 8}
    # A key that is left out keeps the base's value.
    assert ic.parse("algo=ring", base=net).intra == net.intra
    assert ic.parse("").algorithm == "flat"


@pytest.mark.parametrize(
    "text",
    [
        "speed=3",
        "intra=1us",
        "intra=fast:50GB/s",
        "inter=1us:50parsecs",
        "inter=1us:0GB/s",
        "algo=butterfly",
        "gpus_per_node=0",
        "group.ALL_REDUCE=0",
        "gpus_per_node",
    ],
)
def test_parse_rejects_what_it_cannot_read(text: str) -> None:
    with pytest.raises(ValueError):
        ic.parse(text)


def test_ring_allreduce_on_one_node_is_its_closed_form() -> None:
    n = 8
    expected = 2 * (n - 1) * 1000 + 2 * (n - 1) / n * MIB * 0.01

    assert _cost("ALL_REDUCE", MIB, n) == pytest.approx(expected)


def test_tree_allreduce_is_its_closed_form() -> None:
    net = ic.parse("algo=tree", base=NET)

    assert _cost("ALL_REDUCE", MIB, 8, net) == pytest.approx(
        2 * 3 * (1000 + MIB * 0.01)
    )


def test_auto_takes_the_cheaper_algorithm_of_each_op() -> None:
    auto = ic.parse("algo=auto", base=NET)
    tree = ic.parse("algo=tree", base=NET)

    small = _cost("ALL_REDUCE", 1024, 64, auto)
    large = _cost("ALL_REDUCE", 256 * MIB, 64, auto)

    assert small == pytest.approx(_cost("ALL_REDUCE", 1024, 64, tree))
    assert large == pytest.approx(_cost("ALL_REDUCE", 256 * MIB, 64, NET))
    assert small < _cost("ALL_REDUCE", 1024, 64, NET)
    assert large < _cost("ALL_REDUCE", 256 * MIB, 64, tree)


def test_a_ring_over_two_nodes_crosses_between_them_k_times() -> None:
    # 16 ranks on 2 nodes: 30 steps, 3.75 of them between nodes, and the
    # bandwidth of the slower link.
    steps = 2 * 15
    inter = steps * 2 / 16
    expected = (steps - inter) * 1000 + inter * 5000 + 2 * 15 / 16 * MIB * 0.04

    assert _cost("ALL_REDUCE", MIB, 16) == pytest.approx(expected)
    assert _cost("ALL_REDUCE", MIB, 16) > _cost("ALL_REDUCE", MIB, 8) * 2


@pytest.mark.parametrize(
    ("kind", "steps", "volume"),
    [
        ("ALL_GATHER", 7, 7 / 8),
        ("REDUCE_SCATTER", 7, 7 / 8),
        ("ALL_TO_ALL", 7, 7 / 8),
        ("BROADCAST", 7, 1.0),
        ("BARRIER", 7, 0.0),
    ],
)
def test_other_ring_collectives_are_their_closed_forms(
    kind: str, steps: int, volume: float
) -> None:
    assert _cost(kind, MIB, 8) == pytest.approx(
        steps * 1000 + volume * MIB * 0.01
    )


def test_a_group_of_one_costs_nothing_and_flat_is_one_transfer() -> None:
    assert _cost("ALL_REDUCE", MIB, 1) == 0.0
    flat = ic.parse("algo=flat", base=NET)
    assert _cost("ALL_REDUCE", MIB, 8, flat) == pytest.approx(1000 + MIB * 0.01)
    assert _cost("ALL_REDUCE", MIB, 64, flat) == pytest.approx(
        5000 + MIB * 0.04
    )


def test_a_transfer_uses_the_link_between_its_nodes() -> None:
    cost = ic.transfer_cost_ns(
        np.array([MIB, MIB, MIB]),
        np.array([0, 0, -1]),
        np.array([0, 1, 1]),
        NET,
    )

    assert cost[0] == pytest.approx(1000 + MIB * 0.01)
    assert cost[1] == pytest.approx(5000 + MIB * 0.04)
    # A rank the trace does not place is priced as inside a node.
    assert cost[2] == pytest.approx(cost[0])


def _nodes(rows: list[tuple]) -> pl.DataFrame:
    """Builds nodes from (rank, op_class, comm_type, ns, bytes, src, dst)."""
    return pl.DataFrame(
        {
            "rank": [r[0] for r in rows],
            "node_id": list(range(len(rows))),
            "name": [""] * len(rows),
            "op_class": [r[1] for r in rows],
            "comm_type": [r[2] for r in rows],
            "duration_ns": [r[3] for r in rows],
            "size_bytes": [r[4] for r in rows],
            "src_rank": [r[5] for r in rows],
            "dst_rank": [r[6] for r in rows],
            "message_tag": [0] * len(rows),
        },
        schema={
            "rank": pl.UInt32,
            "node_id": pl.UInt64,
            "name": pl.String,
            "op_class": pl.String,
            "comm_type": pl.String,
            "duration_ns": pl.Float64,
            "size_bytes": pl.UInt64,
            "src_rank": pl.Int32,
            "dst_rank": pl.Int32,
            "message_tag": pl.UInt32,
        },
    )


def test_a_send_and_a_receive_each_carry_half_of_the_transfer() -> None:
    nodes = _nodes(
        [
            (0, "comm_send", "", 0, MIB, 0, 9),
            (9, "comm_recv", "", 0, MIB, 0, 9),
            (0, "compute", "", 5, 0, -1, -1),
        ]
    )

    cost, defaulted = ic.node_cost_ns(nodes, NET)

    assert cost.tolist() == [
        0.5 * (5000 + MIB * 0.04),
        0.5 * (5000 + MIB * 0.04),
        0.0,
    ]
    assert defaulted == []


def test_collectives_with_no_group_size_use_every_rank_and_say_so() -> None:
    rows = [(r, "comm_coll", "ALL_REDUCE", 0, MIB, -1, -1) for r in range(8)]
    nodes = _nodes(rows)

    cost, defaulted = ic.node_cost_ns(nodes, NET)
    set_cost, set_defaulted = ic.node_cost_ns(
        nodes, ic.parse("group.ALL_REDUCE=2", base=NET)
    )

    assert defaulted == ["ALL_REDUCE"]
    assert cost[0] == pytest.approx(_cost("ALL_REDUCE", MIB, 8))
    assert set_defaulted == []
    assert set_cost[0] == pytest.approx(_cost("ALL_REDUCE", MIB, 2))


def test_fit_recovers_the_link_the_times_were_made_from() -> None:
    truth = ic.Link(3000.0, 0.02)
    sizes = np.array([4096, 65536, MIB, 8 * MIB, 64 * MIB] * 3)
    ranks = np.array([2, 4, 8] * 5)
    kinds = np.array(["allreduce", "allgather", "alltoall"] * 5)
    names = {
        "allreduce": "ALL_REDUCE",
        "allgather": "ALL_GATHER",
        "alltoall": "ALL_TO_ALL",
    }
    net = ic.Interconnect(intra=truth, inter=truth, algorithm="ring")
    durations = [
        _cost(names[k], int(s), int(r), net)
        for k, s, r in zip(kinds, sizes, ranks, strict=True)
    ]
    ops = pl.DataFrame(
        {
            "op": kinds,
            "bytes": sizes,
            "group_size": ranks,
            "duration_ns": durations,
        }
    ).cast({"duration_ns": pl.Int64})

    fit = ic.fit_link(ops, algorithm="ring")

    assert fit.link.alpha_ns == pytest.approx(truth.alpha_ns, rel=0.01)
    assert fit.link.ns_per_byte == pytest.approx(truth.ns_per_byte, rel=0.01)
    assert fit.median_relative_error < 0.01
    assert fit.r_squared > 0.999
    assert fit.calls == 15


def test_fit_never_returns_a_negative_latency_or_bandwidth() -> None:
    # Times that shrink with size would fit a negative beta.
    ops = pl.DataFrame(
        {
            "op": ["allreduce"] * 4,
            "bytes": [1000, 10_000, 100_000, 1_000_000],
            "group_size": [2] * 4,
            "duration_ns": [9000, 8000, 7000, 6000],
        }
    )

    fit = ic.fit_link(ops)

    assert fit.link.alpha_ns >= 0
    assert fit.link.ns_per_byte >= 0


def test_fit_needs_calls_with_a_size_and_two_ranks() -> None:
    ops = pl.DataFrame(
        {
            "op": ["allreduce", "barrier", "allreduce"],
            "bytes": [0, 0, 1024],
            "group_size": [2, 2, 1],
            "duration_ns": [100, 100, 100],
        }
    )

    with pytest.raises(ValueError, match="at least 2 calls"):
        ic.fit_link(ops)
    with pytest.raises(ValueError, match="'ring' or 'tree'"):
        ic.fit_link(ops, algorithm="flat")


REAL_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_real"
_REAL_MPI = REAL_DATA_DIR / "net_small" / "hostmpi_rank0.sqlite"


@pytest.mark.skipif(
    not _REAL_MPI.exists(), reason="real GPULab captures not present"
)
def test_fit_on_real_shared_memory_mpi_is_rough_not_exact(
    tmp_path: pathlib.Path,
) -> None:
    # Two ranks on one node, sizes 512 B to 32 MiB. Fitted on every other
    # size and tested on the rest, one alpha-beta line misses allreduce and
    # sendrecv by about 40-50% (protocols change with message size); this
    # test records that the model is rough there, not a claim that it is
    # accurate.
    report = cta.analyze(_REAL_MPI, cache_dir=tmp_path / "cache")
    tables = report.sources[report.groups[0].group_id][0]
    ops = (
        pl.read_parquet(tables["network_ops"])
        .filter(
            (pl.col("api") == "mpi")
            & pl.col("op").is_in(["allreduce", "sendrecv"])
        )
        .group_by("op", "bytes", "group_size")
        .agg(pl.col("duration_ns").median())
        .sort("op", "bytes")
        .with_columns(
            pl.int_range(pl.len()).over("op").alias("i"),
        )
    )
    train = ops.filter(pl.col("i") % 2 == 0)
    test = ops.filter(pl.col("i") % 2 == 1)

    fit = ic.fit_link(train, algorithm="ring")
    kind = test["op"].replace_strict(
        {"allreduce": "ALL_REDUCE", "sendrecv": "SEND"}, return_dtype=pl.String
    )
    net = ic.Interconnect(intra=fit.link, inter=fit.link, algorithm="ring")
    predicted = ic.collective_cost_ns(
        kind.to_numpy(),
        test["bytes"].to_numpy(),
        test["group_size"].to_numpy(),
        net,
    )
    measured = test["duration_ns"].to_numpy().astype(float)
    held_out = float(np.median(np.abs(predicted - measured) / measured))

    assert fit.link.alpha_ns >= 0 and fit.link.ns_per_byte > 0
    assert fit.median_relative_error < 0.7
    assert held_out < 0.8
