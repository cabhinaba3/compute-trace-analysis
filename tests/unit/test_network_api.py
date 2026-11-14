"""Tests for the network-ops table and the ``network_api`` aspect."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest

from cta.adapters.kineto import reader
from cta.aspects import network as network_aspect
from cta.core import models
from cta.core import network
from cta.core import registry

REAL_DDP_KINETO_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "data_real" / "e" / "kineto"
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
        }
    )


def test_kernel_name_gives_operation_algorithm_and_protocol() -> None:
    ops = network.ops_from_kernels(
        _kernels(
            [
                "ncclDevKernel_AllReduce_Sum_f32_RING_LL(args)",
                "ncclDevKernel_Broadcast_TREE_SIMPLE(x)",
                "ncclDevKernel_Generic_1(x)",
            ]
        )
    )
    assert ops["op"].to_list() == ["allreduce", "broadcast", "other"]
    assert ops["algo"].to_list() == ["RING", "TREE", ""]
    assert ops["proto"].to_list() == ["LL", "SIMPLE", ""]
    assert ops["bytes"].to_list() == [0, 0, 0]


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


def _ops(durations_ns: dict[int, list[int]], size: int) -> pl.DataFrame:
    rows = [
        {
            "rank": rank, "device": rank, "stream": 7, "step": 0,
            "api": "nccl", "op": "allreduce", "algo": "RING", "proto": "LL",
            "dtype": "Float", "group": "0", "group_size": 2, "bytes": size,
            "start_ns": 1000 * i, "end_ns": 1000 * i + d, "duration_ns": d,
        }
        for rank, durations in durations_ns.items()
        for i, d in enumerate(durations)
    ]  # fmt: skip
    return pl.DataFrame(rows, schema=network.NETWORK_OPS_SCHEMA)


def test_bandwidth_and_wait_skew() -> None:
    ops = _ops({0: [1000, 1000], 1: [400, 1000]}, size=2_000_000)
    per_key = network_aspect.per_operation(ops).row(0, named=True)
    assert per_key["key"] == "allreduce|1-16MB"
    assert per_key["calls"] == 4
    # 8 MB in 3.4 us of summed call time.
    assert per_key["algbw"] == pytest.approx(8_000_000 / 3400)
    assert per_key["busbw"] == pytest.approx(per_key["algbw"])
    skew = network_aspect.wait_skew_us(ops)
    # Call 0 differs by 600 ns between the ranks, call 1 by none.
    assert skew["skew_us"].to_list() == [pytest.approx(0.3)]


def test_aspect_compares_two_groups(tmp_path: pathlib.Path) -> None:
    groups = []
    tables: dict[str, list[dict[str, pathlib.Path]]] = {}
    for name, duration in (("slow", 2000), ("fast", 1000)):
        path = tmp_path / f"{name}.parquet"
        _ops({0: [duration] * 3, 1: [duration] * 3}, 1 << 22).write_parquet(
            path
        )
        groups.append(models.TrialGroup(name, name, ()))
        tables[name] = [{"kineto.network_ops": path}] * 3
    result = registry.ASPECTS["network_api"].run(
        tuple(groups), tables, models.CompareConfig()
    )
    rows = {r["metric"]: r for r in result.summary.iter_rows(named=True)}
    busbw = rows["busbw_GBps|allreduce|1-16MB"]
    assert busbw["candidate"] == pytest.approx(2 * busbw["baseline"])
    assert "allreduce|1-16MB" in result.details["by_operation"]["key"].to_list()


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
    assert ops.height == kernels.filter(pl.col("category") == "comm").height
    assert set(ops["rank"].unique()) == {0, 1}
    allreduce = ops.filter(pl.col("op") == "allreduce", pl.col("bytes") > 0)
    # 524800 float32 elements, seven times on each rank.
    assert allreduce.filter(pl.col("bytes") == 524800 * 4).height == 14
    assert set(allreduce["algo"]) == {"RING"}
    assert set(allreduce["group_size"]) == {2}
