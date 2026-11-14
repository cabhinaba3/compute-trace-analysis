"""Synthetic Jetson jtop telemetry logs."""

from __future__ import annotations

import pathlib

from cta import _native as native


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
    native.write_jtop_trace(
        str(path),
        duration_s=duration_s,
        num_cores=num_cores,
        power_mw=power_mw,
        gpu_util=gpu_util,
        gpu_mem_mb=gpu_mem_mb,
        temp_c=temp_c,
    )
    return path
