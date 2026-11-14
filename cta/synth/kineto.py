"""Synthetic PyTorch Kineto Chrome traces."""

from __future__ import annotations

import json
import pathlib


def write_kineto_trace(
    path: pathlib.Path,
    *,
    steps: int = 3,
    attn_us: float = 400.0,
    gemm_us: float = 600.0,
    kernel_variant: str = "fp16",
) -> pathlib.Path:
    """Writes a PyTorch Profiler / Kineto Chrome trace with CPU-GPU correlation.

    Each ``ProfilerStep#<N>`` holds an attention op and a linear op; each op
    issues a ``cudaLaunchKernel`` whose correlation id matches one GPU kernel.

    Args:
        path: Output file; a ``.gz`` suffix writes gzip-compressed JSON. Its
            parent directory is created if missing.
        steps: Number of profiler steps.
        attn_us: Duration of each attention kernel in microseconds.
        gemm_us: Duration of each GEMM kernel in microseconds.
        kernel_variant: ``"fp16"`` for Ampere FP16 kernel names; any other
            value gives FP8 kernel names.

    Returns:
        ``path``.
    """
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    step_ts_us = 1000.0
    correlation_id = 100

    gemm_kernel_name = (
        "ampere_fp16_s16816gemm"
        if kernel_variant == "fp16"
        else "sm90_wgmma_fp8_gemm_kernel"
    )
    attention_kernel_name = (
        "fmha_cutlassF_f16"
        if kernel_variant == "fp16"
        else "flash_attn_fwd_fp8_kernel"
    )

    for step in range(steps):
        step_dur_us = attn_us + gemm_us + 50.0
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{step}",
                "ts": step_ts_us,
                "dur": step_dur_us,
                "pid": 1,
                "tid": 1,
            }
        )

        # Attention op -> launch -> GPU kernel.
        events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::scaled_dot_product_attention",
                "ts": step_ts_us + 2.0,
                "dur": 15.0,
                "pid": 1,
                "tid": 1,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": step_ts_us + 5.0,
                "dur": 5.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": correlation_id},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": attention_kernel_name,
                "ts": step_ts_us + 10.0,
                "dur": attn_us,
                "pid": 0,
                "tid": 7,
                "args": {"correlation": correlation_id, "stream": 7},
            }
        )
        correlation_id += 1

        # Linear op -> launch -> GPU GEMM kernel.
        gemm_ts_us = step_ts_us + 10.0 + attn_us
        events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::linear",
                "ts": step_ts_us + 20.0,
                "dur": 15.0,
                "pid": 1,
                "tid": 1,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": step_ts_us + 25.0,
                "dur": 5.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": correlation_id},
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": gemm_kernel_name,
                "ts": gemm_ts_us,
                "dur": gemm_us,
                "pid": 0,
                "tid": 7,
                "args": {"correlation": correlation_id, "stream": 7},
            }
        )
        correlation_id += 1

        step_ts_us += step_dur_us + 20.0

    payload = json.dumps({"schemaVersion": 1, "traceEvents": events})
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        path.write_text(payload)
    return path
