"""Synthetic Jetson jtop telemetry logs."""

from __future__ import annotations

import json
import pathlib


def write_jtop_trace(
    path: pathlib.Path,
    *,
    duration_s: int = 5,
    num_cores: int = 8,
    power_mw: float = 15_000.0,
    gpu_util: float = 85.0,
    gpu_mem_mb: float = 2048.0,
    temp_c: float | None = 62.0,
) -> pathlib.Path:
    """Writes a Jetson `jtop` JSON telemetry trace.

    The trace holds one sample per second for ``duration_s + 1`` seconds, with
    nanosecond timestamps and power rails in milliwatts. Every value is
    constant over time; the GPU rail draws 65% of the total power.

    Args:
        path: Output JSON file; its parent directory is created if missing.
        duration_s: Trace length in seconds.
        num_cores: Number of ``CPU<N>`` utilisation columns.
        power_mw: Total board power in milliwatts.
        gpu_util: GPU utilisation in percent.
        gpu_mem_mb: GPU memory use in megabytes.
        temp_c: GPU temperature in degrees Celsius; ``None`` omits the column.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    start_ns = 1_700_000_000_000_000_000
    samples = []
    for second in range(duration_s + 1):
        sample: dict[str, object] = {
            "time": start_ns + second * 1_000_000_000,
            "GPU": gpu_util,
            "gpu_memory_usage": gpu_mem_mb,
            "RAM": 4096.0,
            "SWAP": 128.0,
            "Power TOT": power_mw,
            "VDD_GPU_SOC": power_mw * 0.65,
        }
        for core in range(1, num_cores + 1):
            sample[f"CPU{core}"] = 35.0
        if temp_c is not None:
            sample["Temp gpu"] = temp_c
        samples.append(sample)
    path.write_text(json.dumps(samples, indent=2))
    return path
