from __future__ import annotations

from pathlib import Path

import pytest

from cta import compare
from cta.synth.generators import (
    write_jtop_trace,
    write_trtexec_layers,
    write_trtexec_timing,
)


def test_trtexec_timing_and_layer_fusion(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    b_dir = tmp_path / "trt_fp32"
    c_dir = tmp_path / "trt_int8"

    write_trtexec_timing(b_dir / "times.json", queries=25, compute_ms=4.0, h2d_ms=0.5, d2h_ms=0.4, overlap_ms=0.05)
    write_trtexec_layers(b_dir / "profile.json", b_dir / "engine.json", fused=False, precision="FP32")

    write_trtexec_timing(c_dir / "times.json", queries=25, compute_ms=1.2, h2d_ms=0.5, d2h_ms=0.4, overlap_ms=0.4)
    write_trtexec_layers(c_dir / "profile.json", c_dir / "engine.json", fused=True, precision="INT8")

    rep = compare(b_dir, c_dir, warmup_steps=5, cache_dir=cache)
    assert set(rep.aspects) == {"latency_throughput", "timeline_overlap", "layer_fusion"}

    lat = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    assert lat["total_steps"]["baseline"] == pytest.approx(20.0)
    assert lat["throughput_per_s"]["delta_pct"] > 100.0

    lf = {r["metric"]: r for r in rep.to_polars("layer_fusion").iter_rows(named=True)}
    assert lf["layer_count"]["baseline"] == pytest.approx(4.0)
    assert lf["layer_count"]["candidate"] == pytest.approx(2.0)
    assert lf["fusion_ratio_pct"]["candidate"] == pytest.approx(50.0)
    assert lf["low_precision_layer_pct"]["baseline"] == pytest.approx(0.0)
    assert lf["low_precision_layer_pct"]["candidate"] == pytest.approx(100.0)


def test_jtop_multicore_nano_fallback_and_bundle_joules_per_query(tmp_path: Path) -> None:
    cache = tmp_path / ".cache"
    b_bundle = tmp_path / "orin_max_n"
    c_bundle = tmp_path / "orin_efficient"

    # Baseline: 5 seconds @ 20W = 100 Joules, 100 queries -> 1000 mJ/query, thermal throttle @ 88C
    write_jtop_trace(b_bundle / "jtop.json", duration_s=5, num_cores=12, power_mw=20_000.0, temp_c=88.0)
    write_trtexec_timing(b_bundle / "times.json", queries=100, compute_ms=2.0)

    # Candidate: 5 seconds @ 10W = 50 Joules, 100 queries -> 500 mJ/query (-50% energy/query!), no temp sensor (Jetson Nano style)
    write_jtop_trace(c_bundle / "jtop.json", duration_s=5, num_cores=4, power_mw=10_000.0, temp_c=None)
    write_trtexec_timing(c_bundle / "times.json", queries=100, compute_ms=1.5)

    rep = compare(b_bundle, c_bundle, cache_dir=cache)
    se = {r["metric"]: r for r in rep.to_polars("system_energy").iter_rows(named=True)}
    assert se["total_energy_j"]["baseline"] == pytest.approx(100.0)
    assert se["total_energy_j"]["candidate"] == pytest.approx(50.0)
    assert se["avg_power_w"]["baseline"] == pytest.approx(20.0)
    assert se["energy_per_query_mj"]["baseline"] == pytest.approx(1000.0)
    assert se["energy_per_query_mj"]["candidate"] == pytest.approx(500.0)
    assert se["queries_per_joule"]["delta_pct"] == pytest.approx(100.0)
    assert rep.diagnoses[0].baseline_regime == "Thermal/Power-Throttled"
