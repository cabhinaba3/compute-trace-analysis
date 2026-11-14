"""Tests for multi-rank Kineto traces: per-rank clocks, steps and scopes.

Every rank keeps its own clock and steps, kernels belong to the step that
launched them, and a kernel's scope is the innermost CPU op that encloses its
launch on the launching thread. Real two-rank DDP traces from
`tests/data_real/e` are used when they are present.
"""

from __future__ import annotations

import gzip
import json
import pathlib
from typing import Any

import polars as pl
import pytest

import cta
from cta.adapters.kineto import reader
from cta.core import models

REAL_DDP_KINETO_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "data_real" / "e" / "kineto"
)


def _rank_trace(
    path: pathlib.Path,
    *,
    t0_us: float,
    rank: int | None,
    device: int = 0,
    async_gap_us: float = 0.0,
) -> pathlib.Path:
    """Writes a three-step Kineto trace of one rank.

    Each step's 500 us CPU range launches one kernel whose correlation id is
    the step index. The kernel may start after the range has ended.

    Args:
        path: File to write; a `.gz` suffix gzips it.
        t0_us: Start of the first step, in microseconds.
        rank: Rank recorded in `distributedInfo`; None leaves that out.
        device: GPU that runs the kernels.
        async_gap_us: Delay from each launch to the start of its kernel.

    Returns:
        `path`.
    """
    events: list[dict[str, Any]] = []
    for i in range(3):
        step_start_us = t0_us + i * 1000.0
        launch_us = step_start_us + 100.0
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{i}",
                "ts": step_start_us,
                "dur": 500.0,
                "pid": 1,
                "tid": 1,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": launch_us,
                "dur": 5.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": i},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": f"gemm_{i}",
                "ts": launch_us + async_gap_us,
                "dur": 50.0,
                "pid": device,
                "tid": 7,
                "args": {"correlation": i, "device": device, "stream": 7},
            }
        )
    trace: dict[str, Any] = {"traceEvents": events}
    if rank is not None:
        trace["distributedInfo"] = {
            "backend": "nccl",
            "rank": rank,
            "world_size": 2,
        }
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as fh:
            json.dump(trace, fh)
    else:
        path.write_text(json.dumps(trace))
    return path


def _ingest(
    trace_paths: list[pathlib.Path], out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests the traces as one Kineto artifact and reads back its tables.

    Args:
        trace_paths: One trace file per rank, in any order.
        out_dir: Directory for the Parquet tables; created if missing.

    Returns:
        The tables, keyed by table name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(
        format="kineto",
        paths=tuple(sorted(trace_paths)),
        world_size=len(trace_paths),
    )
    table_paths = reader.KinetoAdapter().ingest(artifact, out_dir)
    return {
        table: pl.read_parquet(parquet_path)
        for table, parquet_path in table_paths.items()
    }


def test_ranks_keep_their_own_clocks_and_steps(tmp_path: pathlib.Path) -> None:
    rank0_path = _rank_trace(tmp_path / "a.json", t0_us=1_000.0, rank=0)
    # The clock of rank 1 is about 9 s away from that of rank 0.
    rank1_path = _rank_trace(
        tmp_path / "b.json", t0_us=9_000_000.0, rank=1, device=1
    )
    tables = _ingest([rank0_path, rank1_path], tmp_path / "o")
    kernels, steps = tables["kernels"], tables["steps"]
    assert sorted(kernels["rank"].unique().to_list()) == [0, 1]
    rank1_kernels = kernels.filter(pl.col("rank") == 1)
    assert rank1_kernels["device"].unique().to_list() == [1]
    steps_per_rank = steps.group_by("rank").len().sort("rank")["len"]
    assert steps_per_rank.to_list() == [3, 3]
    # Every kernel lands in a step of its own rank; a single merged lookup
    # would mix the two clocks.
    for rank in (0, 1):
        rank_kernels = kernels.filter(pl.col("rank") == rank)
        kernel_steps = rank_kernels.sort("start_ns")["step"].to_list()
        assert sorted(kernel_steps) == [0, 1, 2]


def test_rank_from_filename_when_trace_has_no_distributed_info(
    tmp_path: pathlib.Path,
) -> None:
    rank0_path = _rank_trace(
        tmp_path / "trace_rank0.json", t0_us=1_000.0, rank=None
    )
    rank1_path = _rank_trace(
        tmp_path / "trace_rank1.json", t0_us=5_000.0, rank=None
    )
    kernels = _ingest([rank0_path, rank1_path], tmp_path / "o")["kernels"]
    assert sorted(kernels["rank"].unique().to_list()) == [0, 1]


def test_kernel_is_attributed_to_the_step_that_launched_it(
    tmp_path: pathlib.Path,
) -> None:
    # Each kernel starts 700 us after its launch, which is after the end of
    # its step's 500 us CPU range (in the gap before the next step).
    trace_path = _rank_trace(
        tmp_path / "x.json.gz", t0_us=1_000.0, rank=0, async_gap_us=700.0
    )
    kernels = _ingest([trace_path], tmp_path / "o")["kernels"]
    # A lookup by GPU start time would give [-1, 0, 1].
    assert kernels.sort("start_ns")["step"].to_list() == [0, 1, 2]


def test_analyze_reports_steps_per_rank_not_summed(
    tmp_path: pathlib.Path,
) -> None:
    rank0_path = _rank_trace(tmp_path / "t_rank0.json", t0_us=1_000.0, rank=0)
    _rank_trace(tmp_path / "t_rank1.json", t0_us=7_000_000.0, rank=1)
    report = cta.analyze(rank0_path.parent, label="r", cache_dir=tmp_path / "c")
    latency = report.to_polars("latency_throughput")
    latency_by_metric = {
        row["metric"]: row for row in latency.iter_rows(named=True)
    }
    assert latency_by_metric["total_steps"]["r"] == 3.0
    overlap = report.to_polars("timeline_overlap")
    overlap_by_metric = {
        row["metric"]: row for row in overlap.iter_rows(named=True)
    }
    # Busy time is measured in per-rank windows, not in one ~7 s window that
    # spans both clocks.
    assert overlap_by_metric["gpu_busy_pct"]["r"] > 0


@pytest.mark.skipif(
    not (REAL_DDP_KINETO_DIR / "trace_rank1.json.gz").exists(),
    reason="real GPULab fixtures not present",
)
def test_real_two_rank_ddp_traces(tmp_path: pathlib.Path) -> None:
    trace_paths = sorted(REAL_DDP_KINETO_DIR.glob("trace_rank*.json.gz"))
    tables = _ingest(trace_paths, tmp_path / "o")
    kernels, steps = tables["kernels"], tables["steps"]
    devices_by_rank = (
        kernels.group_by("rank")
        .agg(pl.col("device").unique().sort())
        .sort("rank")
    )
    assert devices_by_rank["device"].to_list() == [[0], [1]]
    steps_per_rank = steps.group_by("rank").len().sort("rank")["len"]
    assert steps_per_rank.to_list() == [9, 9]
    assert (kernels["step"] >= 0).all()
    # Both ranks run NCCL kernels.
    comm_kernels = kernels.filter(pl.col("category") == "comm")
    assert comm_kernels.group_by("rank").len().height == 2
    report = cta.analyze(
        REAL_DDP_KINETO_DIR, label="ddp", cache_dir=tmp_path / "c"
    )
    latency = report.to_polars("latency_throughput")
    rows_by_metric = {
        row["metric"]: row for row in latency.iter_rows(named=True)
    }
    # Steps are counted per rank; a merged pile would report 18.
    assert rows_by_metric["total_steps"]["ddp"] == 9.0


def _scope_trace(path: pathlib.Path) -> pathlib.Path:
    """Writes a one-step Kineto trace whose kernels have known scopes.

    On thread 1, `outer` (0 to 1000 us) encloses `inner` (100 to 200 us).
    Kernel 1 is launched inside `inner`, kernel 2 at 500 us after `inner`
    has ended, kernel 3 by thread 2 inside `worker_op`, and kernel 4 at
    1500 us outside every op.

    Args:
        path: File to write.

    Returns:
        `path`.
    """

    def cpu_op(
        name: str, start_us: float, duration_us: float, tid: int = 1
    ) -> dict[str, Any]:
        """Returns a CPU op event of process 1."""
        return {
            "ph": "X",
            "cat": "cpu_op",
            "name": name,
            "ts": start_us,
            "dur": duration_us,
            "pid": 1,
            "tid": tid,
        }

    def launch(
        correlation: int, start_us: float, tid: int = 1
    ) -> dict[str, Any]:
        """Returns a 2 us `cudaLaunchKernel` call of process 1."""
        return {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "ts": start_us,
            "dur": 2.0,
            "pid": 1,
            "tid": tid,
            "args": {"correlation": correlation},
        }

    def kernel(correlation: int, start_us: float) -> dict[str, Any]:
        """Returns a 10 us kernel on device 0, stream 7."""
        return {
            "ph": "X",
            "cat": "kernel",
            "name": f"k{correlation}",
            "ts": start_us,
            "dur": 10.0,
            "pid": 0,
            "tid": 7,
            "args": {"correlation": correlation, "device": 0, "stream": 7},
        }

    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "ProfilerStep#0",
            "ts": 0.0,
            "dur": 2000.0,
            "pid": 1,
            "tid": 1,
        },
        cpu_op("outer", 0.0, 1000.0),
        cpu_op("inner", 100.0, 100.0),
        # Another thread: overlaps in time but is not the launcher.
        cpu_op("worker_op", 300.0, 400.0, tid=2),
        launch(1, 150.0),
        kernel(1, 3000.0),  # Inside inner (and outer).
        launch(2, 500.0),
        kernel(2, 3100.0),  # Inner has ended, outer still encloses.
        launch(3, 500.0, tid=2),
        kernel(3, 3200.0),  # Launched by the worker thread.
        launch(4, 1500.0),
        kernel(4, 3300.0),  # No enclosing op on its thread.
    ]
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def test_scope_is_innermost_enclosing_op_on_the_launching_thread(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _scope_trace(tmp_path / "s.json")
    kernels = _ingest([trace_path], tmp_path / "o")["kernels"]
    # A lookup of only the latest-starting op would miss `outer` for kernel 2.
    assert kernels.sort("start_ns")["scope"].to_list() == [
        "inner",
        "outer",
        "worker_op",
        "",
    ]


def test_scope_depth_picks_a_coarser_level(tmp_path: pathlib.Path) -> None:
    from cta.aspects.device import kernels as device_kernels

    trace_path = _scope_trace(tmp_path / "s.json")
    tables = _ingest([trace_path], tmp_path / "o")
    kernels = tables["kernels"].sort("start_ns")
    assert kernels["scope_path"].to_list() == [
        "outer > inner",
        "outer",
        "worker_op",
        "",
    ]
    no_depth = device_kernels.with_scope_level(kernels, None)
    assert no_depth["scope"].to_list() == ["inner", "outer", "worker_op", ""]
    depth_1 = device_kernels.with_scope_level(kernels, 1)
    assert depth_1["scope"].to_list() == ["outer", "outer", "worker_op", ""]
    # A depth beyond the deepest level is clamped to the innermost op.
    depth_5 = device_kernels.with_scope_level(kernels, 5)
    assert depth_5["scope"].to_list() == ["inner", "outer", "worker_op", ""]
