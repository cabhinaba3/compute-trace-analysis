from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import polars as pl
import pytest

from cta.adapters.nsys.reader import NsysAdapter
from cta.core.models import TraceArtifact

REAL = Path(__file__).resolve().parents[1] / "data_real"


def _ingest(paths: list[Path], out: Path) -> dict[str, pl.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    art = TraceArtifact(format="nsys", paths=tuple(sorted(paths)), world_size=len(paths))
    return {k: pl.read_parquet(v) for k, v in NsysAdapter().ingest(art, out).items()}


def _make_db(path: Path, *, kernels: list[tuple], nvtx: list[tuple], runtime: list[tuple], memset: list[tuple] | None = None) -> Path:
    """Real-schema Nsight Systems export: kernels=(name_id,start,end,device,stream,corr), nvtx=(text,start,end,tid),
    runtime=(corr,tid,start,end), memset=(start,end,bytes,device,corr)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (demangledName INTEGER, shortName INTEGER, start INTEGER, end INTEGER,
            deviceId INTEGER, streamId INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (nameId INTEGER, start INTEGER, end INTEGER, correlationId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER, bytes INTEGER, deviceId INTEGER, streamId INTEGER, correlationId INTEGER);
        CREATE TABLE NVTX_EVENTS (eventType INTEGER, start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        """
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "gemm_kernel"), (2, "cudaLaunchKernel")])
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)", [(k[0], k[0], *k[1:]) for k in kernels])
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2, ?, ?, ?, ?)", [(r[2], r[3], r[0], r[1]) for r in runtime])
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?, ?, 7, ?)", memset or [])
    conn.executemany("INSERT INTO NVTX_EVENTS VALUES (59, ?, ?, ?, NULL, ?)", [(n[1], n[2], n[0], n[3]) for n in nvtx])
    conn.commit()
    conn.close()
    return path


def test_kernel_attributed_to_launching_step_not_gpu_time(tmp_path: Path) -> None:
    # Step 0's CPU range ends at 1_000; its kernel is launched at 900 but only *runs* at 1_500, inside step 1's CPU range.
    db = _make_db(
        tmp_path / "a.sqlite",
        kernels=[(1, 1_500, 1_900, 0, 7, 11), (1, 2_500, 2_900, 0, 7, 12)],
        nvtx=[("ProfilerStep#0", 0, 1_000, 5), ("ProfilerStep#1", 1_100, 2_100, 5)],
        runtime=[(11, 5, 900, 910), (12, 5, 1_900, 1_910)],
    )
    k = _ingest([db], tmp_path / "o1")["kernels"].sort("start_ns")
    assert k["step"].to_list() == [0, 1]  # a GPU-time lookup would give [1, -1]


def test_scope_is_innermost_range_even_after_sibling_ended(tmp_path: Path) -> None:
    # outer 0..1000 contains inner A 100..200; launch at 500 is in `outer` only (A already ended).
    db = _make_db(
        tmp_path / "b.sqlite",
        kernels=[(1, 600, 700, 0, 7, 1), (1, 150, 180, 0, 7, 2)],
        nvtx=[("ProfilerStep#0", 0, 1_000, 5), ("outer", 0, 1_000, 5), ("A", 100, 200, 5)],
        runtime=[(1, 5, 500, 505), (2, 5, 150, 155)],
    )
    k = _ingest([db], tmp_path / "o2")["kernels"].sort("start_ns")
    assert k["scope"].to_list() == ["A", "outer"]  # latest-start-only lookup yields "" for the second


def test_memset_device_and_rank_are_ingested(tmp_path: Path) -> None:
    r0 = _make_db(tmp_path / "x_rank0.sqlite", kernels=[(1, 10, 20, 0, 7, 1)], nvtx=[("ProfilerStep#0", 0, 100, 5)],
                  runtime=[(1, 5, 5, 6)], memset=[(30, 40, 512, 0, 2)])
    r1 = _make_db(tmp_path / "x_rank1.sqlite", kernels=[(1, 10, 20, 1, 7, 1)], nvtx=[("ProfilerStep#0", 0, 100, 5)],
                  runtime=[(1, 5, 5, 6)])
    t = _ingest([r0, r1], tmp_path / "o3")
    k = t["kernels"]
    assert sorted(k["rank"].unique().to_list()) == [0, 1]  # both reports read, not just the first
    assert k.filter(pl.col("rank") == 1)["device"].to_list() == [1]
    assert k.filter(pl.col("name") == "Memset")["bytes"].to_list() == [512]
    assert t["steps"].group_by("rank").len().height == 2


def test_missing_step_markers_warns(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "c.sqlite", kernels=[(1, 10, 20, 0, 7, 1)], nvtx=[], runtime=[(1, 5, 5, 6)])
    with pytest.warns(RuntimeWarning, match="no NVTX step markers"):
        _ingest([db], tmp_path / "o4")


@pytest.mark.skipif(not (REAL / "b" / "ranks_rank0.sqlite").exists(), reason="real GPULab fixtures not present")
def test_real_per_rank_and_multi_gpu_reports(tmp_path: Path) -> None:
    ranks = _ingest([REAL / "b" / "ranks_rank0.sqlite", REAL / "b" / "ranks_rank1.sqlite"], tmp_path / "r")
    assert sorted(ranks["kernels"]["rank"].unique().to_list()) == [0, 1]
    assert ranks["kernels"].filter(pl.col("category") == "comm").height > 0  # NCCL all-reduce on both ranks
    assert ranks["steps"].group_by("rank").len()["len"].to_list() == [6, 6]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multi = _ingest([REAL / "b" / "multigpu.sqlite"], tmp_path / "m")
    assert sorted(multi["kernels"]["device"].unique().to_list()) == [0, 1]
    assert multi["kernels"].filter(pl.col("name") == "Memset").height > 0


@pytest.mark.skipif(not (REAL / "a_nsys" / "single_fp16_gpumetrics.sqlite").exists(), reason="real GPULab fixtures not present")
def test_real_gpu_metrics_keep_zero_samples_and_are_windowed(tmp_path: Path) -> None:
    from cta.aspects.device import _gpu_metrics_in_steps

    t = _ingest([REAL / "a_nsys" / "single_fp16_gpumetrics.sqlite"], tmp_path / "g")
    gm, steps = t["gpu_metrics"], t["steps"]
    assert gm["metric_name"].n_unique() == 30 and (gm["value"] == 0).sum() > 0  # zeros are kept
    assert t["steps"].height == 6 and "rank" in gm.columns
    inside = _gpu_metrics_in_steps(gm, steps)
    assert 0 < inside.height < gm.height  # whole-capture idle time is excluded
