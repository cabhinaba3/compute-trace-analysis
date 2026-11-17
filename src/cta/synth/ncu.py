"""Synthetic Nsight Compute CSV exports."""

from __future__ import annotations

import pathlib

from cta import _native as native


def write_ncu_csv(
    path: pathlib.Path,
    *,
    gemm_dur_ns: float = 500_000.0,
    sm_pct: float = 40.0,
    dram_pct: float = 82.0,
    tensor_pct: float = 35.0,
    flops: float = 2.0e10,
    dram_bytes: float = 1.0e9,
) -> pathlib.Path:
    """Writes a realistic Nsight Compute raw CSV export.

    The file starts with a ``==PROF==`` banner, a header row and a unit row,
    followed by one GEMM kernel row built from the arguments and one fixed
    element-wise kernel row.

    Args:
        path: Output CSV file; its parent directory is created if missing.
        gemm_dur_ns: GEMM kernel duration in nanoseconds.
        sm_pct: SM throughput in percent of peak; also sets the
            math-pipe-throttle stall column (40% of it).
        dram_pct: Memory throughput in percent of peak; also sets the
            long-scoreboard stall column (50% of it).
        tensor_pct: Tensor pipe activity in percent of peak.
        flops: Floating-point operations of the GEMM kernel.
        dram_bytes: DRAM bytes moved by the GEMM kernel.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_ncu_csv(
        str(path),
        gemm_dur_ns=gemm_dur_ns,
        sm_pct=sm_pct,
        dram_pct=dram_pct,
        tensor_pct=tensor_pct,
        flops=flops,
        dram_bytes=dram_bytes,
    )
    return path
