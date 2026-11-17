"""Tests for the network-ops table and the ``network`` aspect."""

from __future__ import annotations

import pathlib
import sqlite3

import numpy as np
import polars as pl
import pytest

from cta import models
from cta import plugins
from cta import table_schemas
from cta.adapters.kineto import reader
from cta.adapters.nsys import network_ops
from cta.analysis import network

REAL_DDP_KINETO_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "data_real" / "e" / "kineto"
)


def _kernels(names: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": names,
            "category": ["comm"] * len(names),
            "rank": [0] * len(names),
            "device": [0] * len(names),
            "stream": [7] * len(names),
            "step": [0] * len(names),
            "start_ns": [i * 1000 for i in range(len(names))],
            "end_ns": [i * 1000 + 500 for i in range(len(names))],
            "duration_ns": [500] * len(names),
            "bytes": [0] * len(names),
        }
    )


def test_kernel_name_gives_operation_algorithm_and_protocol() -> None:
    ops = network.ops_from_kernels(
        _kernels(
            [
                "ncclDevKernel_AllReduce_Sum_f32_RING_LL(args)",
                "ncclDevKernel_Broadcast_TREE_SIMPLE(x)",
                "ncclDevKernel_Generic_1(x)",
                # NCCL 2.32 drops the reduction and the protocol.
                "ncclDevKernel_AllReduce_f32_RING",
                "ncclDevKernel_SendRecv",
            ]
        )
    )
    assert ops["launch_ns"].to_list() == [-1] * 5
    assert ops["op"].to_list() == [
        "allreduce",
        "broadcast",
        "other",
        "allreduce",
        "sendrecv",
    ]
    assert ops["algo"].to_list() == ["RING", "TREE", "", "RING", ""]
    assert ops["proto"].to_list() == ["LL", "SIMPLE", "", "", ""]
    assert ops["bytes"].to_list() == [0, 0, 0, 0, 0]


def test_collective_metadata_sets_operation_and_bytes() -> None:
    collective = pl.DataFrame(
        {
            "row": [0, 1],
            "collective": ["all_gather", "allreduce"],
            "dtype": ["Float", "Half"],
            "group": ["0", "0"],
            "in_elems": [10, 100],
            "out_elems": [20, 100],
            "group_size": [2, 2],
        }
    )
    ops = network.ops_from_kernels(
        _kernels(
            ["ncclDevKernel_AllGather_RING_LL(x)", "ncclDevKernel_Generic_1(x)"]
        ),
        collective,
    )
    assert ops["op"].to_list() == ["allgather", "allreduce"]
    # An allgather counts its output; an allreduce its input.
    assert ops["bytes"].to_list() == [80, 200]
    assert ops["group_size"].to_list() == [2, 2]


def test_busbw_factors_and_buckets() -> None:
    assert network.busbw_factor("allreduce", 4) == pytest.approx(1.5)
    assert network.busbw_factor("allgather", 4) == pytest.approx(0.75)
    assert network.busbw_factor("broadcast", 4) == 1.0
    assert network.busbw_factor("allreduce", 1) == 1.0
    labels = network.bucket_bytes(np.array([0, 100, 5 << 20, 1 << 30]))
    assert list(labels) == ["unknown", "<1KB", "1-16MB", ">256MB"]


def _ops(
    durations_ns: dict[int, list[int]], size: int, api: str = "nccl"
) -> pl.DataFrame:
    rows = [
        {
            "rank": rank,
            "device": rank,
            "stream": 7,
            "step": 0,
            "api": api,
            "op": "allreduce",
            "posted": False,
            # The launch is 300 ns before the start, 100 ns on rank 1.
            "launch_ns": 1000 * i - (300 if rank == 0 else 100),
            "algo": "RING",
            "proto": "LL",
            "dtype": "Float",
            "group": "0",
            "group_size": 2,
            "peer": -1,
            "bytes": size,
            "start_ns": 1000 * i,
            "end_ns": 1000 * i + d,
            "duration_ns": d,
        }
        for rank, durations in durations_ns.items()
        for i, d in enumerate(durations)
    ]
    return pl.DataFrame(rows, schema=table_schemas.NETWORK_OPS)


def test_bandwidth_and_wait_skew() -> None:
    ops = _ops({0: [1000, 1000], 1: [400, 1000]}, size=2_000_000)
    per_key = network.per_operation(ops).row(0, named=True)
    assert (per_key["api"], per_key["op"], per_key["bucket"]) == (
        "nccl",
        "allreduce",
        "1-16MB",
    )
    assert per_key["calls"] == 4
    # 8 MB in 3.4 us of summed call time; two ranks make the factor 1.
    assert per_key["timed_bytes"] / per_key["timed_ns"] == pytest.approx(
        8_000_000 / 3400
    )
    assert per_key["weighted_bytes"] == per_key["timed_bytes"]
    skew = network.wait_skew_us(ops)
    # Call 0 differs by 600 ns between the ranks, call 1 by none.
    assert skew["skew_us"].to_list() == [pytest.approx(0.3)]


def test_launch_gap_is_start_minus_launch() -> None:
    ops = _ops({0: [1000, 1000], 1: [400, 1000]}, size=1)
    gap = network.launch_gap_us(ops)
    # Two calls at 300 ns and two at 100 ns.
    assert gap["gap_us"].to_list() == [pytest.approx(0.2)]
    unknown = ops.with_columns(launch_ns=pl.lit(-1))
    assert network.launch_gap_us(unknown).is_empty()


def test_posted_calls_are_counted_but_not_timed() -> None:
    ops = _ops({0: [1000], 1: [1000]}, size=2_000_000).with_columns(
        posted=pl.lit(True)
    )
    per_key = network.per_operation(ops).row(0, named=True)
    assert per_key["calls"] == 2
    assert per_key["bytes"] == 4_000_000
    assert per_key["timed_bytes"] == per_key["timed_ns"] == 0
    assert network.wait_skew_us(ops).is_empty()


def test_mpi_and_ucx_names() -> None:
    assert network.mpi_op("MPI_Allreduce") == ("allreduce", False)
    assert network.mpi_op("MPI_Bcast") == ("broadcast", False)
    assert network.mpi_op("MPI_Isend") == ("send", True)
    assert network.mpi_op("MPI_Reduce_scatter_block") == (
        "reducescatter",
        False,
    )
    assert network.mpi_op("MPI_Ireduce_scatter") == ("reducescatter", True)
    # "I" at the start of a name does not mean nonblocking.
    assert network.mpi_op("MPI_Init") == ("init", False)
    assert network.ucx_op("ucp_tag_send_nbx") == "send"
    assert network.ucx_op("ucp_ep_create") == "other"


def _mpi_export() -> sqlite3.Connection:
    """An in-memory export with MPI calls, a communicator and a UCX submit."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES (1, 'MPI_Allreduce'), (2, 'MPI_Isend'),
            (3, 'MPI_Bcast'), (4, 'ucp_tag_send_nbx');
        CREATE TABLE MPI_COMMUNICATORS (commUid INTEGER, size INTEGER);
        INSERT INTO MPI_COMMUNICATORS VALUES (7, 4);
        CREATE TABLE MPI_COLLECTIVES_EVENTS (start INTEGER, "end" INTEGER,
            textId INTEGER, commUid INTEGER, size INTEGER, recvSize INTEGER);
        INSERT INTO MPI_COLLECTIVES_EVENTS VALUES
            (1000, 3000, 1, 7, 16384, 16384), (5000, 6000, 3, 7, NULL, 128);
        CREATE TABLE MPI_P2P_EVENTS (start INTEGER, "end" INTEGER,
            textId INTEGER, commUid INTEGER, size INTEGER,
            remoteRank INTEGER);
        INSERT INTO MPI_P2P_EVENTS VALUES (1500, 1600, 2, 7, 1024, 1);
        CREATE TABLE UCP_SUBMIT_EVENTS (start INTEGER, "end" INTEGER,
            textId INTEGER, packedSize INTEGER);
        INSERT INTO UCP_SUBMIT_EVENTS VALUES (1520, 1550, 4, 1024);
        """
    )
    return conn


def test_host_ops_from_an_export() -> None:
    conn = _mpi_export()
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    steps = pl.DataFrame(
        {"step": [0, 1], "start_ns": [0, 4000], "end_ns": [3500, 9000]}
    )
    ops = network_ops.host_network_ops(conn, tables, rank=3, steps=steps)
    assert ops.schema == table_schemas.NETWORK_OPS
    rows = {(r["api"], r["op"]): r for r in ops.iter_rows(named=True)}
    allreduce = rows["mpi", "allreduce"]
    # nsys records the buffer times the four ranks; the buffer is 4096 bytes.
    assert allreduce["bytes"] == 4096
    assert allreduce["group_size"] == 4
    assert allreduce["rank"] == 3
    assert allreduce["step"] == 0
    # A broadcast receiver only knows the receive size.
    assert rows["mpi", "broadcast"]["bytes"] == 128
    assert rows["mpi", "broadcast"]["step"] == 1
    assert rows["mpi", "send"]["posted"]
    assert rows["mpi", "send"]["peer"] == 1
    assert rows["mpi", "allreduce"]["peer"] == -1
    assert rows["ucx", "send"]["peer"] == -1
    assert rows["ucx", "send"]["posted"]
    assert rows["ucx", "send"]["bytes"] == 1024


# (function, size, recvSize) as nsys recorded them for a 4096-byte buffer on
# four Open MPI ranks, and the size nccl-tests would report.
_MEASURED = [
    ("MPI_Allgather", 16384, 16384, 16384),
    ("MPI_Allgatherv", 16384, 16384, 16384),
    ("MPI_Allreduce", 16384, 16384, 4096),
    ("MPI_Alltoall", 16384, 16384, 16384),
    ("MPI_Reduce_scatter_block", 16384, 16384, 16384),
    ("MPI_Bcast", 16384, 4096, 4096),  # root
    ("MPI_Bcast", 0, 4096, 4096),
    ("MPI_Gather", 4096, 16384, 16384),  # root
    ("MPI_Gather", 4096, 0, 16384),
    ("MPI_Scatter", 16384, 4096, 16384),  # root
    ("MPI_Scatter", 0, 4096, 16384),
    ("MPI_Reduce", 4096, 16384, 4096),  # root
    ("MPI_Reduce", 4096, 0, 4096),
    ("MPI_Reduce", 0, 12288, 4096),  # root, MPI_IN_PLACE
]


@pytest.mark.parametrize(("name", "size", "recv", "expected"), _MEASURED)
def test_mpi_message_size_convention(
    name: str, size: int, recv: int, expected: int
) -> None:
    mpi = pl.DataFrame(
        {
            "start_ns": [0],
            "end_ns": [10],
            "name": [name],
            "group": ["1"],
            "group_size": [4],
            "peer": [-1],
            "size": [size],
            "recv_size": [recv],
        }
    )
    assert network_ops._mpi_rows(mpi)["bytes"].to_list() == [expected]


def test_export_without_network_tables_gives_no_rows() -> None:
    conn = sqlite3.connect(":memory:")
    steps = pl.DataFrame(
        schema={"step": pl.Int32, "start_ns": pl.Int64, "end_ns": pl.Int64}
    )
    ops = network_ops.host_network_ops(conn, set(), 0, steps)
    assert ops.is_empty()
    assert ops.schema == table_schemas.NETWORK_OPS


def test_peer_copies_become_p2p_rows() -> None:
    kernels = _kernels(["Memcpy_P2P", "Memcpy_H2D"]).with_columns(
        category=pl.lit("memcpy"), bytes=pl.Series([4096, 8192])
    )
    ops = network.ops_from_kernels(kernels)
    assert ops["api"].to_list() == ["p2p"]
    assert ops["bytes"].to_list() == [4096]


def test_aspect_compares_two_apis(tmp_path: pathlib.Path) -> None:
    groups = []
    tables: dict[str, list[dict[str, pathlib.Path]]] = {}
    for name, api, duration in (
        ("mpi_run", "mpi", 2000),
        ("nccl_run", "nccl", 1000),
    ):
        path = tmp_path / f"{name}.parquet"
        _ops(
            {0: [duration] * 3, 1: [duration] * 3}, 1 << 22, api
        ).write_parquet(path)
        groups.append(models.TrialGroup(name, name, ()))
        tables[name] = [{"kineto.network_ops": path}] * 3
    result = plugins.ASPECTS["network"].run(
        tuple(groups), tables, models.AnalysisConfig()
    )
    rows = {r["metric"]: r for r in result.summary.iter_rows(named=True)}
    # Each group ran one API, so the per-API metrics do not overlap...
    assert "busbw_GBps|mpi|allreduce|1-16MB" in rows
    assert "busbw_GBps|nccl|allreduce|1-16MB" in rows
    # ...and the pooled table matches them by operation and size.
    pooled = result.details["by_operation"].row(0, named=True)
    assert pooled["key"] == "allreduce|1-16MB"
    assert pooled["nccl_run"] == pytest.approx(2 * pooled["mpi_run"])


@pytest.mark.skipif(
    not (REAL_DDP_KINETO_DIR / "trace_rank1.json.gz").exists(),
    reason="real GPULab fixtures not present",
)
def test_real_ddp_traces_have_sized_collectives(tmp_path: pathlib.Path) -> None:
    artifact = models.TraceArtifact(
        format="kineto",
        paths=tuple(sorted(REAL_DDP_KINETO_DIR.glob("trace_rank*.json.gz"))),
        world_size=2,
    )
    paths = reader.KinetoAdapter().ingest(artifact, tmp_path)
    ops = pl.read_parquet(paths["network_ops"])
    kernels = pl.read_parquet(paths["kernels"])
    nccl = ops.filter(pl.col("api") == "nccl")
    assert nccl.height == kernels.filter(pl.col("category") == "comm").height
    assert set(ops["rank"].unique()) == {0, 1}
    allreduce = nccl.filter(pl.col("op") == "allreduce", pl.col("bytes") > 0)
    # 524800 float32 elements, seven times on each rank.
    assert allreduce.filter(pl.col("bytes") == 524800 * 4).height == 14
    assert set(allreduce["algo"]) == {"RING"}
    assert set(allreduce["group_size"]) == {2}
