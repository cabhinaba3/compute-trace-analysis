"""Synthetic Nsight Systems SQLite exports."""

from __future__ import annotations

import pathlib


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
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    connection = sqlite3.connect(str(path))
    try:
        # SQLite stores CREATE TABLE text verbatim, so the script layout is
        # part of the generated file.
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                demangledName INTEGER, shortName INTEGER, start INTEGER, end INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                copyKind INTEGER, start INTEGER, end INTEGER, bytes INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                nameId INTEGER, start INTEGER, end INTEGER
            );
            CREATE TABLE NVTX_EVENTS (
                textId INTEGER, text TEXT, start INTEGER, end INTEGER
            );
            CREATE TABLE OSRT_API (
                nameId INTEGER, start INTEGER, end INTEGER
            );
            """  # noqa: E501
        )
        strings_by_id = {
            1: "void cutlass::Kernel<fp16>(half const*, int)",
            2: "ncclDevKernel_AllReduce_Sum_f16_RING_LL",
            3: "cudaLaunchKernel",
            4: "pthread_mutex_lock",
            5: "poll",
            6: "TransformerBlock_0",
        }
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)", list(strings_by_id.items())
        )

        step_start_ns = 1_000_000
        for step in range(steps):
            warmup_extra_ns = warmup_penalty_us * 1000 if step == 0 else 0
            gemm_dur_ns = gemm_us * 1000 + warmup_extra_ns
            comm_dur_ns = comm_us * 1000
            overlap_ns = min(comm_overlap_us * 1000, gemm_dur_ns)

            gemm_start_ns = step_start_ns
            gemm_end_ns = gemm_start_ns + gemm_dur_ns
            comm_start_ns = gemm_end_ns - overlap_ns
            comm_end_ns = comm_start_ns + comm_dur_ns
            # Each step ends with a 50 us idle gap.
            step_end_ns = max(gemm_end_ns, comm_end_ns) + 50_000

            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (NULL, ?, ?, ?)",
                (f"ProfilerStep#{step}", step_start_ns, step_end_ns),
            )
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (6, NULL, ?, ?)",
                (gemm_start_ns, gemm_end_ns),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 1, ?, ?, 7)",
                (gemm_start_ns, gemm_end_ns),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2, 2, ?, ?, 8)",
                (comm_start_ns, comm_end_ns),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY "
                "VALUES (1, ?, ?, 65536, 9)",
                (gemm_start_ns, gemm_start_ns + 20_000),
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (3, ?, ?)",
                (gemm_start_ns - 5_000, gemm_start_ns),
            )
            connection.execute(
                "INSERT INTO OSRT_API VALUES (4, ?, ?)",
                (gemm_start_ns, gemm_start_ns + lock_wait_us * 1000),
            )
            step_start_ns = step_end_ns

        if legacy_gpu_metrics:
            connection.execute(
                "CREATE TABLE GPU_METRICS (rawTimestamp INTEGER, value TEXT)"
            )
            payload = str(
                {
                    "SMs Active [Throughput %]": str(tensor_active_pct + 10.0),
                    "Tensor Active [Throughput %]": str(tensor_active_pct),
                    "SM Issue [Throughput %]": str(tensor_active_pct - 5.0),
                }
            )
            connection.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, ?)",
                [(1000, payload), (2000, payload)],
            )
        else:
            connection.executescript(
                """
                CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER PRIMARY KEY, metricName TEXT);
                CREATE TABLE GPU_METRICS (metricId INTEGER, timestamp INTEGER, value REAL);
                """  # noqa: E501
            )
            connection.executemany(
                "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)",
                [
                    (1, "SMs Active [Throughput %]"),
                    (2, "Tensor Active [Throughput %]"),
                    (3, "SM Issue [Throughput %]"),
                ],
            )
            connection.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, ?, ?)",
                [
                    (1, 1000, tensor_active_pct + 10.0),
                    (2, 1000, tensor_active_pct),
                    (3, 1000, tensor_active_pct - 5.0),
                ],
            )
        connection.commit()
    finally:
        connection.close()
    return path
