"""Tests for the TensorRT (trtexec) and Jetson jtop adapters.

Covers trtexec timing and layer-fusion comparisons, and jtop power traces
combined with trtexec timing in one bundle to get energy per query, including
multi-core boards and boards without a temperature sensor.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

import cta
from cta.core import models
from cta.synth import jtop
from cta.synth import tensorrt


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


def test_trtexec_timing_and_layer_fusion(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / ".cache"
    fp32_dir = tmp_path / "trt_fp32"
    int8_dir = tmp_path / "trt_int8"

    tensorrt.write_trtexec_timing(
        fp32_dir / "times.json",
        queries=25,
        compute_ms=4.0,
        h2d_ms=0.5,
        d2h_ms=0.4,
        overlap_ms=0.05,
    )
    tensorrt.write_trtexec_layers(
        fp32_dir / "profile.json",
        fp32_dir / "engine.json",
        fused=False,
        precision="FP32",
    )

    tensorrt.write_trtexec_timing(
        int8_dir / "times.json",
        queries=25,
        compute_ms=1.2,
        h2d_ms=0.5,
        d2h_ms=0.4,
        overlap_ms=0.4,
    )
    tensorrt.write_trtexec_layers(
        int8_dir / "profile.json",
        int8_dir / "engine.json",
        fused=True,
        precision="INT8",
    )

    report = cta.compare(
        fp32_dir, int8_dir, warmup_steps=5, cache_dir=cache_dir
    )
    assert set(report.aspects) == {
        "latency_throughput",
        "timeline_overlap",
        "layer_fusion",
    }

    latency = _summary_by_metric(report, "latency_throughput")
    assert latency["total_steps"]["baseline"] == pytest.approx(20.0)
    assert latency["throughput_per_s"]["delta_pct"] > 100.0

    fusion = _summary_by_metric(report, "layer_fusion")
    assert fusion["layer_count"]["baseline"] == pytest.approx(4.0)
    assert fusion["layer_count"]["candidate"] == pytest.approx(2.0)
    assert fusion["fusion_ratio_pct"]["candidate"] == pytest.approx(50.0)
    low_precision_pct = fusion["low_precision_layer_pct"]
    assert low_precision_pct["baseline"] == pytest.approx(0.0)
    assert low_precision_pct["candidate"] == pytest.approx(100.0)


def test_jtop_multicore_nano_fallback_and_bundle_joules_per_query(
    tmp_path: pathlib.Path,
) -> None:
    cache_dir = tmp_path / ".cache"
    max_n_dir = tmp_path / "orin_max_n"
    efficient_dir = tmp_path / "orin_efficient"

    # Baseline: 5 s at 20 W = 100 J over 100 queries, so 1000 mJ per query,
    # and a thermal throttle at 88 C.
    jtop.write_jtop_trace(
        max_n_dir / "jtop.json",
        duration_s=5,
        num_cores=12,
        power_mw=20_000.0,
        temp_c=88.0,
    )
    tensorrt.write_trtexec_timing(
        max_n_dir / "times.json", queries=100, compute_ms=2.0
    )

    # Candidate: 5 s at 10 W = 50 J over 100 queries, so 500 mJ per query (50%
    # less energy per query), and no temperature sensor (as on a Jetson Nano).
    jtop.write_jtop_trace(
        efficient_dir / "jtop.json",
        duration_s=5,
        num_cores=4,
        power_mw=10_000.0,
        temp_c=None,
    )
    tensorrt.write_trtexec_timing(
        efficient_dir / "times.json", queries=100, compute_ms=1.5
    )

    report = cta.compare(max_n_dir, efficient_dir, cache_dir=cache_dir)
    energy = _summary_by_metric(report, "system_energy")
    assert energy["total_energy_j"]["baseline"] == pytest.approx(100.0)
    assert energy["total_energy_j"]["candidate"] == pytest.approx(50.0)
    assert energy["avg_power_w"]["baseline"] == pytest.approx(20.0)
    assert energy["energy_per_query_mj"]["baseline"] == pytest.approx(1000.0)
    assert energy["energy_per_query_mj"]["candidate"] == pytest.approx(500.0)
    assert energy["queries_per_joule"]["delta_pct"] == pytest.approx(100.0)
    assert report.diagnoses[0].baseline_regime == "Thermal/Power-Throttled"
