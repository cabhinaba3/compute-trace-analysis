from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl

import cta
from cta.adapters.kineto.reader import KinetoAdapter
from cta.core.models import TraceArtifact


def _trace(path: Path, *, t0_us: float, rank: int | None, device: int = 0, async_gap_us: float = 0.0) -> Path:
    """3 steps; each step's CPU range issues one kernel (corr = step) that may start after the range has ended."""
    ev: list[dict[str, Any]] = []
    for i in range(3):
        s = t0_us + i * 1000.0
        ev.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}", "ts": s, "dur": 500.0, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": s + 100.0, "dur": 5.0, "pid": 1, "tid": 1, "args": {"correlation": i}})
        ev.append({"ph": "X", "cat": "kernel", "name": f"gemm_{i}", "ts": s + 100.0 + async_gap_us, "dur": 50.0, "pid": device, "tid": 7,
                   "args": {"correlation": i, "device": device, "stream": 7}})  # fmt: skip
    doc: dict[str, Any] = {"traceEvents": ev}
    if rank is not None:
        doc["distributedInfo"] = {"backend": "nccl", "rank": rank, "world_size": 2}
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as gz:
            json.dump(doc, gz)
    else:
        path.write_text(json.dumps(doc))
    return path


def _ingest(paths: list[Path], out: Path) -> dict[str, pl.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    art = TraceArtifact(format="kineto", paths=tuple(sorted(paths)), world_size=len(paths))
    return {k: pl.read_parquet(v) for k, v in KinetoAdapter().ingest(art, out).items()}


def test_ranks_keep_their_own_clocks_and_steps(tmp_path: Path) -> None:
    a = _trace(tmp_path / "a.json", t0_us=1_000.0, rank=0)
    b = _trace(tmp_path / "b.json", t0_us=9_000_000.0, rank=1, device=1)  # clock ~9 s away from rank 0
    t = _ingest([a, b], tmp_path / "o")
    k, st = t["kernels"], t["steps"]
    assert sorted(k["rank"].unique().to_list()) == [0, 1] and k.filter(pl.col("rank") == 1)["device"].unique().to_list() == [1]
    assert st.group_by("rank").len().sort("rank")["len"].to_list() == [3, 3]
    for r in (0, 1):  # every kernel lands in its own rank's step (a single merged lookup would mix the clocks)
        assert sorted(k.filter(pl.col("rank") == r).sort("start_ns")["step"].to_list()) == [0, 1, 2]


def test_rank_from_filename_when_trace_has_no_distributed_info(tmp_path: Path) -> None:
    f0 = _trace(tmp_path / "trace_rank0.json", t0_us=1_000.0, rank=None)
    f1 = _trace(tmp_path / "trace_rank1.json", t0_us=5_000.0, rank=None)
    assert sorted(_ingest([f0, f1], tmp_path / "o")["kernels"]["rank"].unique().to_list()) == [0, 1]


def test_kernel_is_attributed_to_the_step_that_launched_it(tmp_path: Path) -> None:
    # The kernel starts 700 us after its launch, i.e. after its step's 500 us CPU range (inside the next gap).
    f = _trace(tmp_path / "x.json.gz", t0_us=1_000.0, rank=0, async_gap_us=700.0)
    k = _ingest([f], tmp_path / "o")["kernels"].sort("start_ns")
    assert k["step"].to_list() == [0, 1, 2]  # a GPU-start lookup would give [-1, 0, 1]


def test_analyze_reports_steps_per_rank_not_summed(tmp_path: Path) -> None:
    a = _trace(tmp_path / "t_rank0.json", t0_us=1_000.0, rank=0)
    b = _trace(tmp_path / "t_rank1.json", t0_us=7_000_000.0, rank=1)
    rep = cta.analyze([a, b][0].parent, label="r", cache_dir=tmp_path / "c")
    rows = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    assert rows["total_steps"]["r"] == 3.0
    ov = {r["metric"]: r for r in rep.to_polars("timeline_overlap").iter_rows(named=True)}
    assert ov["gpu_busy_pct"]["r"] > 0  # per-rank windows: not a ~7 s window spanning both clocks


REAL_E = Path(__file__).resolve().parents[1] / "data_real" / "e" / "kineto"


import pytest  # noqa: E402


@pytest.mark.skipif(not (REAL_E / "trace_rank1.json.gz").exists(), reason="real GPULab fixtures not present")
def test_real_two_rank_ddp_traces(tmp_path: Path) -> None:
    t = _ingest(sorted(REAL_E.glob("trace_rank*.json.gz")), tmp_path / "o")
    k, st = t["kernels"], t["steps"]
    assert k.group_by("rank").agg(pl.col("device").unique().sort()).sort("rank")["device"].to_list() == [[0], [1]]
    assert st.group_by("rank").len().sort("rank")["len"].to_list() == [9, 9]
    assert (k["step"] >= 0).all() and k.filter(pl.col("category") == "comm").group_by("rank").len().height == 2  # NCCL on both
    rep = cta.analyze(REAL_E, label="ddp", cache_dir=tmp_path / "c")
    rows = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    assert rows["total_steps"]["ddp"] == 9.0  # steps per rank; a merged pile would report 18


def _scope_trace(path: Path) -> Path:
    """One step; on thread 1 `outer` (0..1000 us) contains `inner` (100..200 us, already finished) and a launch at 500."""
    def op(name: str, ts: float, dur: float, tid: int = 1) -> dict[str, Any]:
        return {"ph": "X", "cat": "cpu_op", "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": tid}

    def launch(corr: int, ts: float, tid: int = 1) -> dict[str, Any]:
        return {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 2.0, "pid": 1, "tid": tid, "args": {"correlation": corr}}

    def kernel(corr: int, ts: float) -> dict[str, Any]:
        return {"ph": "X", "cat": "kernel", "name": f"k{corr}", "ts": ts, "dur": 10.0, "pid": 0, "tid": 7, "args": {"correlation": corr, "device": 0, "stream": 7}}

    ev = [
        {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#0", "ts": 0.0, "dur": 2000.0, "pid": 1, "tid": 1},
        op("outer", 0.0, 1000.0), op("inner", 100.0, 100.0),
        op("worker_op", 300.0, 400.0, tid=2),  # another thread, overlapping in time but not the launcher
        launch(1, 150.0), kernel(1, 3000.0),  # inside inner (and outer)
        launch(2, 500.0), kernel(2, 3100.0),  # inner has ended, outer still encloses
        launch(3, 500.0, tid=2), kernel(3, 3200.0),  # launched by the worker thread
        launch(4, 1500.0), kernel(4, 3300.0),  # no enclosing op on its thread
    ]
    path.write_text(json.dumps({"traceEvents": ev}))
    return path


def test_scope_is_innermost_enclosing_op_on_the_launching_thread(tmp_path: Path) -> None:
    k = _ingest([_scope_trace(tmp_path / "s.json")], tmp_path / "o")["kernels"].sort("start_ns")
    assert k["scope"].to_list() == ["inner", "outer", "worker_op", ""]  # a latest-start-only lookup misses `outer` for kernel 2


def test_scope_depth_picks_a_coarser_level(tmp_path: Path) -> None:
    from cta.aspects.device import _with_scope_level

    k = _ingest([_scope_trace(tmp_path / "s.json")], tmp_path / "o")["kernels"].sort("start_ns")
    assert k["scope_path"].to_list() == ["outer > inner", "outer", "worker_op", ""]
    assert _with_scope_level(k, None)["scope"].to_list() == ["inner", "outer", "worker_op", ""]
    assert _with_scope_level(k, 1)["scope"].to_list() == ["outer", "outer", "worker_op", ""]
    assert _with_scope_level(k, 5)["scope"].to_list() == ["inner", "outer", "worker_op", ""]  # clamped to the innermost
