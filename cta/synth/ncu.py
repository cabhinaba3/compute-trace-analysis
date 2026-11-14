"""Synthetic Nsight Compute CSV exports."""

from __future__ import annotations

import pathlib


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
    gemm_row = (
        f'"0","ampere_fp16_s16816gemm","{gemm_dur_ns}","{sm_pct}",'
        f'"{dram_pct}","{tensor_pct}","62.5","{flops}","{dram_bytes}",'
        f'"{dram_pct * 0.5}","{sm_pct * 0.4}"'
    )
    lines = [
        "==PROF== Connected to process 12345",
        '"ID","Kernel Name","gpu__time_duration.sum",'
        '"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
        '"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",'
        '"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",'
        '"sm__warps_active.avg.pct_of_peak_sustained_active",'
        '"FLOPs","dram__bytes.sum",'
        '"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",'
        '"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"',
        '"","","nsecond","%","%","%","%","","byte","%","%"',
        gemm_row,
        '"1","vectorized_elementwise_relu","100000.0","20.0","88.0","0.0",'
        '"55.0","1e8","2e8","48.0","5.0"',
    ]
    path.write_text("\n".join(lines))
    return path
