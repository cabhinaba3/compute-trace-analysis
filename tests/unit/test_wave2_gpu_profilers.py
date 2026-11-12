from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cta import compare
from cta.core.intervals import compute_timeline_overlap
from cta.synth.generators import write_kineto_trace, write_ncu_csv, write_nsys_sqlite


def test_sweepline_exact_conservation() -> None:
    cs = np.array([100, 500], dtype=np.int64)
    ce = np.array([400, 800], dtype=np.int64)
    ms = np.array([300, 900], dtype=np.int64)
    me = np.array([600, 1000], dtype=np.int64)
    ov = compute_timeline_overlap(cs, ce, ms, me, window_start=0, window_end=1100)
    # Compute: [100,400) U [500,800) = 600; Comm: [300,600) U [900,1000) = 400
    # Overlap: [300,400) U [500,600) = 200 -> comp_only = 400, comm_only = 200, idle = 1100 - 800 = 300
    assert ov.window_ns == 1100
    assert ov.overlap_ns == 200
    assert ov.compute_only_ns == 400
    assert ov.comm_only_ns == 200
    assert ov.idle_ns == 300
    assert ov.compute_only_ns + ov.comm_only_ns + ov.overlap_ns + ov.idle_ns == ov.window_ns


def test_nsys_sqlite_warmup_slicing_and_legacy_schema(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    base = write_nsys_sqlite(
        tmp_path / "base.sqlite",
        steps=4,
        gemm_us=800,
        comm_us=400,
        comm_overlap_us=50,
        warmup_penalty_us=2000,
        lock_wait_us=120,
        tensor_active_pct=45.0,
        legacy_gpu_metrics=True,
    )
    cand = write_nsys_sqlite(
        tmp_path / "cand.sqlite",
        steps=4,
        gemm_us=400,
        comm_us=400,
        comm_overlap_us=350,
        warmup_penalty_us=2000,
        lock_wait_us=20,
        tensor_active_pct=80.0,
        legacy_gpu_metrics=False,
    )

    rep = compare(base, cand, warmup_steps=1, cache_dir=cache)
    assert set(rep.aspects) == {"latency_throughput", "kernel_diff", "timeline_overlap", "os_runtime"}

    lat = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    # Warmup step 0 trimmed -> remaining 3 steps analyzed cleanly!
    assert lat["total_steps"]["baseline"] == pytest.approx(3.0)
    assert lat["mean_step_ms"]["speedup"] > 1.8

    ov = {r["metric"]: r for r in rep.to_polars("timeline_overlap").iter_rows(named=True)}
    assert ov["comm_overlap_pct"]["candidate"] > ov["comm_overlap_pct"]["baseline"]
    assert ov["tensor_active_median_pct"]["baseline"] == pytest.approx(45.0)
    assert ov["tensor_active_median_pct"]["candidate"] == pytest.approx(80.0)

    osrt = {r["metric"]: r for r in rep.to_polars("os_runtime").iter_rows(named=True)}
    assert osrt["lock_sync_ms"]["delta_pct"] < -80.0


def test_kineto_gz_family_and_cpu_op_scope_matching(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    fp16 = write_kineto_trace(tmp_path / "fp16.json.gz", steps=3, attn_us=400.0, gemm_us=600.0, kernel_variant="fp16")
    fp8 = write_kineto_trace(tmp_path / "fp8.json.gz", steps=3, attn_us=200.0, gemm_us=300.0, kernel_variant="fp8")

    rep = compare(fp16, fp8, warmup_steps=1, cache_dir=cache)
    # Even though kernel names changed completely (ampere_fp16 vs sm90_wgmma_fp8), family & CPU-op scope align 100%!
    by_fam = rep.to_polars("kernel_diff", table="by_family")
    fam_map = {r["op_family"]: r for r in by_fam.iter_rows(named=True)}
    assert fam_map["GEMM"]["delta_pct"] == pytest.approx(-50.0)
    assert fam_map["FlashAttention"]["delta_pct"] == pytest.approx(-50.0)

    by_scope = rep.to_polars("kernel_diff", table="by_scope")
    scope_map = {r["scope"]: r for r in by_scope.iter_rows(named=True)}
    assert scope_map["aten::scaled_dot_product_attention"]["delta_pct"] == pytest.approx(-50.0)
    assert scope_map["aten::linear"]["delta_pct"] == pytest.approx(-50.0)


def test_ncu_roofline_and_regime_shift(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    mem_bound = write_ncu_csv(tmp_path / "mem.csv", gemm_dur_ns=600_000.0, sm_pct=25.0, dram_pct=85.0, flops=1e10, dram_bytes=1e9)
    comp_bound = write_ncu_csv(tmp_path / "comp.csv", gemm_dur_ns=300_000.0, sm_pct=85.0, dram_pct=30.0, flops=4e10, dram_bytes=5e8)

    rep = compare(mem_bound, comp_bound, cache_dir=cache)
    rf = {r["metric"]: r for r in rep.to_polars("roofline_diff").iter_rows(named=True)}
    assert rf["achieved_tflops"]["delta_pct"] > 500.0
    assert rf["overall_arith_intensity"]["candidate"] > rf["overall_arith_intensity"]["baseline"]
    assert rep.diagnoses[0].baseline_regime == "Memory-BW-Dominated"
    assert rep.diagnoses[0].candidate_regime == "Compute-Dominated"
