"""Tests for the GPU profiler adapters on synthetic traces.

Covers the sweep-line timeline overlap, Nsight Systems SQLite warm-up slicing
with the legacy GPU-metrics schema, Kineto kernel alignment by op family and
CPU-op scope, and the Nsight Compute roofline and regime shift.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

import cta
from cta.core import intervals
from cta.core import models
from cta.synth import kineto
from cta.synth import ncu
from cta.synth import nsys


def _summary_by_metric(
    report: models.ComparisonReport, aspect_name: str
) -> dict[str, dict[str, Any]]:
    """Returns the summary rows of one aspect, keyed by metric name.

    Args:
        report: The comparison report to read.
        aspect_name: The aspect whose summary table is read.

    Returns:
        Each summary row as a dictionary, keyed by its `metric` value.
    """
    summary = report.to_polars(aspect_name)
    return {row["metric"]: row for row in summary.iter_rows(named=True)}


def test_sweepline_exact_conservation() -> None:
    compute_start_ns = np.array([100, 500], dtype=np.int64)
    compute_end_ns = np.array([400, 800], dtype=np.int64)
    comm_start_ns = np.array([300, 900], dtype=np.int64)
    comm_end_ns = np.array([600, 1000], dtype=np.int64)
    breakdown = intervals.compute_timeline_overlap(
        compute_start_ns,
        compute_end_ns,
        comm_start_ns,
        comm_end_ns,
        window_start=0,
        window_end=1100,
    )
    # Compute: [100,400) U [500,800) = 600; comm: [300,600) U [900,1000) = 400.
    # Overlap: [300,400) U [500,600) = 200, so compute only = 400, comm only =
    # 200 and idle = 1100 - 800 = 300.
    assert breakdown.window_ns == 1100
    assert breakdown.overlap_ns == 200
    assert breakdown.compute_only_ns == 400
    assert breakdown.comm_only_ns == 200
    assert breakdown.idle_ns == 300
    accounted_ns = (
        breakdown.compute_only_ns
        + breakdown.comm_only_ns
        + breakdown.overlap_ns
        + breakdown.idle_ns
    )
    assert accounted_ns == breakdown.window_ns


def test_nsys_sqlite_warmup_slicing_and_legacy_schema(
    tmp_path: pathlib.Path,
) -> None:
    cache_dir = tmp_path / ".cache"
    baseline_db_path = nsys.write_nsys_sqlite(
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
    candidate_db_path = nsys.write_nsys_sqlite(
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

    report = cta.compare(
        baseline_db_path, candidate_db_path, warmup_steps=1, cache_dir=cache_dir
    )
    assert set(report.aspects) == {
        "latency_throughput",
        "kernel_diff",
        "timeline_overlap",
        "os_runtime",
    }

    latency = _summary_by_metric(report, "latency_throughput")
    # Warm-up step 0 is trimmed, so the remaining 3 steps are analysed.
    assert latency["total_steps"]["baseline"] == pytest.approx(3.0)
    assert latency["mean_step_ms"]["speedup"] > 1.8

    overlap = _summary_by_metric(report, "timeline_overlap")
    comm_overlap_pct = overlap["comm_overlap_pct"]
    assert comm_overlap_pct["candidate"] > comm_overlap_pct["baseline"]
    tensor_active_pct = overlap["tensor_active_median_pct"]
    assert tensor_active_pct["baseline"] == pytest.approx(45.0)
    assert tensor_active_pct["candidate"] == pytest.approx(80.0)

    os_runtime = _summary_by_metric(report, "os_runtime")
    assert os_runtime["lock_sync_ms"]["delta_pct"] < -80.0


def test_kineto_gz_family_and_cpu_op_scope_matching(
    tmp_path: pathlib.Path,
) -> None:
    cache_dir = tmp_path / ".cache"
    fp16_trace_path = kineto.write_kineto_trace(
        tmp_path / "fp16.json.gz",
        steps=3,
        attn_us=400.0,
        gemm_us=600.0,
        kernel_variant="fp16",
    )
    fp8_trace_path = kineto.write_kineto_trace(
        tmp_path / "fp8.json.gz",
        steps=3,
        attn_us=200.0,
        gemm_us=300.0,
        kernel_variant="fp8",
    )

    report = cta.compare(
        fp16_trace_path, fp8_trace_path, warmup_steps=1, cache_dir=cache_dir
    )
    # The kernel names change completely (ampere_fp16 vs sm90_wgmma_fp8), yet
    # the op families and CPU-op scopes still align one to one.
    family_table = report.to_polars("kernel_diff", table="by_family")
    rows_by_family = {
        row["op_family"]: row for row in family_table.iter_rows(named=True)
    }
    assert rows_by_family["GEMM"]["delta_pct"] == pytest.approx(-50.0)
    flash_attention = rows_by_family["FlashAttention"]
    assert flash_attention["delta_pct"] == pytest.approx(-50.0)

    scope_table = report.to_polars("kernel_diff", table="by_scope")
    rows_by_scope = {
        row["scope"]: row for row in scope_table.iter_rows(named=True)
    }
    attention = rows_by_scope["aten::scaled_dot_product_attention"]
    assert attention["delta_pct"] == pytest.approx(-50.0)
    assert rows_by_scope["aten::linear"]["delta_pct"] == pytest.approx(-50.0)


def test_ncu_roofline_and_regime_shift(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / ".cache"
    memory_bound_csv_path = ncu.write_ncu_csv(
        tmp_path / "mem.csv",
        gemm_dur_ns=600_000.0,
        sm_pct=25.0,
        dram_pct=85.0,
        flops=1e10,
        dram_bytes=1e9,
    )
    compute_bound_csv_path = ncu.write_ncu_csv(
        tmp_path / "comp.csv",
        gemm_dur_ns=300_000.0,
        sm_pct=85.0,
        dram_pct=30.0,
        flops=4e10,
        dram_bytes=5e8,
    )

    report = cta.compare(
        memory_bound_csv_path, compute_bound_csv_path, cache_dir=cache_dir
    )
    roofline = _summary_by_metric(report, "roofline_diff")
    assert roofline["achieved_tflops"]["delta_pct"] > 500.0
    intensity = roofline["overall_arithmetic_intensity"]
    assert intensity["candidate"] > intensity["baseline"]
    assert report.diagnoses[0].baseline_regime == "Memory-BW-Dominated"
    assert report.diagnoses[0].candidate_regime == "Compute-Dominated"
