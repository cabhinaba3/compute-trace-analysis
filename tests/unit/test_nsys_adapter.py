"""Tests for the Nsight Systems SQLite adapter.

Small exports with the real Nsight Systems schema check how kernels are
attributed to steps and NVTX scopes and how memsets and ranks are read; real
GPULab captures check per-rank, multi-GPU and GPU-metrics reports.
"""

from __future__ import annotations

import pathlib
import sqlite3
import warnings

import polars as pl
import pytest

from cta.adapters.nsys import reader
from cta.core import models

REAL_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_real"


def _ingest(
    report_paths: list[pathlib.Path], out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests Nsight Systems exports as one artifact and reads its tables.

    Args:
        report_paths: One SQLite export per rank, in any order.
        out_dir: Directory for the adapter's Parquet output; it is created.

    Returns:
        Every table the adapter wrote, keyed by table name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(
        format="nsys",
        paths=tuple(sorted(report_paths)),
        world_size=len(report_paths),
    )
    table_paths = reader.NsysAdapter().ingest(artifact, out_dir)
    return {name: pl.read_parquet(path) for name, path in table_paths.items()}


def _write_nsys_export(
    db_path: pathlib.Path,
    *,
    kernels: list[tuple],
    nvtx: list[tuple],
    runtime: list[tuple],
    memset: list[tuple] | None = None,
) -> pathlib.Path:
    """Writes a minimal SQLite export with the real Nsight Systems schema.

    The string table maps id 1 to `gemm_kernel` and id 2 to
    `cudaLaunchKernel`, the name given to every runtime call.

    Args:
        db_path: Where to create the SQLite file.
        kernels: Kernel rows as `(name_id, start, end, device, stream,
            correlation_id)`.
        nvtx: NVTX ranges as `(text, start, end, thread_id)`.
        runtime: Runtime API calls as `(correlation_id, thread_id, start,
            end)`.
        memset: Memset rows as `(start, end, bytes, device, correlation_id)`,
            all on stream 7.

    Returns:
        `db_path`.
    """
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            demangledName INTEGER, shortName INTEGER, start INTEGER,
            end INTEGER, deviceId INTEGER, streamId INTEGER,
            correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            nameId INTEGER, start INTEGER, end INTEGER,
            correlationId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start INTEGER, end INTEGER, bytes INTEGER, deviceId INTEGER,
            streamId INTEGER, correlationId INTEGER);
        CREATE TABLE NVTX_EVENTS (
            eventType INTEGER, start INTEGER, end INTEGER, text TEXT,
            textId INTEGER, globalTid INTEGER);
        """
    )
    connection.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [(1, "gemm_kernel"), (2, "cudaLaunchKernel")],
    )
    # The name id fills both the demangled and the short name column.
    kernel_rows = [(name_id, name_id, *rest) for name_id, *rest in kernels]
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
        kernel_rows,
    )
    runtime_rows = [
        (start, end, correlation_id, thread_id)
        for correlation_id, thread_id, start, end in runtime
    ]
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2, ?, ?, ?, ?)",
        runtime_rows,
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?, ?, 7, ?)",
        memset or [],
    )
    nvtx_rows = [
        (start, end, text, thread_id) for text, start, end, thread_id in nvtx
    ]
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (59, ?, ?, ?, NULL, ?)", nvtx_rows
    )
    connection.commit()
    connection.close()
    return db_path


def test_kernel_attributed_to_launching_step_not_gpu_time(
    tmp_path: pathlib.Path,
) -> None:
    # Step 0's CPU range ends at 1_000. Its kernel is launched at 900 but only
    # runs at 1_500, inside step 1's CPU range.
    db_path = _write_nsys_export(
        tmp_path / "a.sqlite",
        kernels=[(1, 1_500, 1_900, 0, 7, 11), (1, 2_500, 2_900, 0, 7, 12)],
        nvtx=[
            ("ProfilerStep#0", 0, 1_000, 5),
            ("ProfilerStep#1", 1_100, 2_100, 5),
        ],
        runtime=[(11, 5, 900, 910), (12, 5, 1_900, 1_910)],
    )
    kernels = _ingest([db_path], tmp_path / "o1")["kernels"].sort("start_ns")
    # A lookup by GPU time would give [1, -1].
    assert kernels["step"].to_list() == [0, 1]


def test_scope_is_innermost_range_even_after_sibling_ended(
    tmp_path: pathlib.Path,
) -> None:
    # `outer` (0..1000) contains the inner range A (100..200). The launch at
    # 500 is inside `outer` only, because A has already ended.
    db_path = _write_nsys_export(
        tmp_path / "b.sqlite",
        kernels=[(1, 600, 700, 0, 7, 1), (1, 150, 180, 0, 7, 2)],
        nvtx=[
            ("ProfilerStep#0", 0, 1_000, 5),
            ("outer", 0, 1_000, 5),
            ("A", 100, 200, 5),
        ],
        runtime=[(1, 5, 500, 505), (2, 5, 150, 155)],
    )
    kernels = _ingest([db_path], tmp_path / "o2")["kernels"].sort("start_ns")
    # A lookup that only checks the latest-starting range would give "" for
    # the second kernel.
    assert kernels["scope"].to_list() == ["A", "outer"]


def test_memset_device_and_rank_are_ingested(tmp_path: pathlib.Path) -> None:
    rank0_db_path = _write_nsys_export(
        tmp_path / "x_rank0.sqlite",
        kernels=[(1, 10, 20, 0, 7, 1)],
        nvtx=[("ProfilerStep#0", 0, 100, 5)],
        runtime=[(1, 5, 5, 6)],
        memset=[(30, 40, 512, 0, 2)],
    )
    rank1_db_path = _write_nsys_export(
        tmp_path / "x_rank1.sqlite",
        kernels=[(1, 10, 20, 1, 7, 1)],
        nvtx=[("ProfilerStep#0", 0, 100, 5)],
        runtime=[(1, 5, 5, 6)],
    )
    tables = _ingest([rank0_db_path, rank1_db_path], tmp_path / "o3")
    kernels = tables["kernels"]
    # Both reports are read, not just the first.
    assert sorted(kernels["rank"].unique().to_list()) == [0, 1]
    assert kernels.filter(pl.col("rank") == 1)["device"].to_list() == [1]
    memsets = kernels.filter(pl.col("name") == "Memset")
    assert memsets["bytes"].to_list() == [512]
    assert tables["steps"].group_by("rank").len().height == 2


def test_missing_step_markers_warns(tmp_path: pathlib.Path) -> None:
    db_path = _write_nsys_export(
        tmp_path / "c.sqlite",
        kernels=[(1, 10, 20, 0, 7, 1)],
        nvtx=[],
        runtime=[(1, 5, 5, 6)],
    )
    with pytest.warns(RuntimeWarning, match="no NVTX step markers"):
        _ingest([db_path], tmp_path / "o4")


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "b" / "ranks_rank0.sqlite").exists(),
    reason="real GPULab fixtures not present",
)
def test_real_per_rank_and_multi_gpu_reports(tmp_path: pathlib.Path) -> None:
    rank_reports = [
        REAL_DATA_DIR / "b" / "ranks_rank0.sqlite",
        REAL_DATA_DIR / "b" / "ranks_rank1.sqlite",
    ]
    per_rank = _ingest(rank_reports, tmp_path / "r")
    per_rank_kernels = per_rank["kernels"]
    assert sorted(per_rank_kernels["rank"].unique().to_list()) == [0, 1]
    # NCCL all-reduce runs on both ranks.
    assert per_rank_kernels.filter(pl.col("category") == "comm").height > 0
    steps_per_rank = per_rank["steps"].group_by("rank").len()
    assert steps_per_rank["len"].to_list() == [6, 6]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multi_gpu = _ingest(
            [REAL_DATA_DIR / "b" / "multigpu.sqlite"], tmp_path / "m"
        )
    multi_gpu_kernels = multi_gpu["kernels"]
    assert sorted(multi_gpu_kernels["device"].unique().to_list()) == [0, 1]
    assert multi_gpu_kernels.filter(pl.col("name") == "Memset").height > 0


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "a_nsys" / "single_fp16_gpumetrics.sqlite").exists(),
    reason="real GPULab fixtures not present",
)
def test_real_gpu_metrics_keep_zero_samples_and_are_windowed(
    tmp_path: pathlib.Path,
) -> None:
    from cta.aspects.device import step_slicing

    report_path = REAL_DATA_DIR / "a_nsys" / "single_fp16_gpumetrics.sqlite"
    tables = _ingest([report_path], tmp_path / "g")
    gpu_metrics, steps = tables["gpu_metrics"], tables["steps"]
    assert gpu_metrics["metric_name"].n_unique() == 30
    # Zero-valued samples are kept.
    assert (gpu_metrics["value"] == 0).sum() > 0
    assert steps.height == 6
    assert "rank" in gpu_metrics.columns
    samples_in_steps = step_slicing.gpu_metrics_in_steps(gpu_metrics, steps)
    # Idle time outside the steps of the capture is excluded.
    assert 0 < samples_in_steps.height < gpu_metrics.height
