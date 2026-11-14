"""Synthetic Nsight Systems SQLite exports."""

from __future__ import annotations

import pathlib

from cta import _native as native


def write_nsys_sqlite(
    path: pathlib.Path,
    *,
    steps: int = 4,
    gemm_us: int = 800,
    comm_us: int = 400,
    comm_overlap_us: int = 100,
    warmup_penalty_us: int = 1200,
    lock_wait_us: int = 50,
    tensor_active_pct: float = 65.0,
    legacy_gpu_metrics: bool = False,
) -> pathlib.Path:
    """Creates a realistic Nsight Systems `.sqlite` database.

    Each step has an NVTX ``ProfilerStep#<N>`` range, a GEMM kernel inside an
    NVTX ``TransformerBlock_0`` range, an NCCL kernel that overlaps the end of
    the GEMM, a host-to-device copy, a ``cudaLaunchKernel`` runtime call and a
    ``pthread_mutex_lock`` OS runtime call. Steps are separated by 50 us of
    idle time. GPU metrics use the modern or the legacy ``GPU_METRICS`` schema.

    Args:
        path: Output database; an existing file is replaced and the parent
            directory is created if missing.
        steps: Number of profiler steps.
        gemm_us: GEMM kernel duration per step in microseconds.
        comm_us: NCCL kernel duration per step in microseconds.
        comm_overlap_us: Overlap of the NCCL kernel with the GEMM kernel in
            microseconds (at most the GEMM duration).
        warmup_penalty_us: Extra GEMM time in the first step in microseconds.
        lock_wait_us: Duration of each mutex-lock call in microseconds.
        tensor_active_pct: Tensor Active throughput in percent; the SMs
            Active and SM Issue metrics are derived from it.
        legacy_gpu_metrics: Whether to write the legacy ``GPU_METRICS`` table
            (one text column of metric values) instead of the modern schema.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_nsys_sqlite(
        str(path),
        steps=steps,
        gemm_us=gemm_us,
        comm_us=comm_us,
        comm_overlap_us=comm_overlap_us,
        warmup_penalty_us=warmup_penalty_us,
        lock_wait_us=lock_wait_us,
        tensor_active_pct=tensor_active_pct,
        legacy_gpu_metrics=legacy_gpu_metrics,
    )
    return path
