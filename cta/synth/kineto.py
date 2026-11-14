"""Synthetic PyTorch Kineto Chrome traces."""

from __future__ import annotations

import pathlib

from cta import _native as native


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
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_kineto_trace(
        str(path),
        steps=steps,
        attn_us=attn_us,
        gemm_us=gemm_us,
        kernel_variant=kernel_variant,
    )
    return path


def write_kineto_trace_large(
    path: pathlib.Path,
    *,
    target_bytes: int = 200 << 20,
    seed: int = 1,
    rank: int = 0,
    world_size: int = 1,
    layers_per_step: int = 24,
    child_cpu_ops_per_layer: int = 3,
    python_events_per_layer: int = 2,
    tensors_per_op: int = 3,
    distributed_info_after: bool = True,
) -> int:
    """Writes a large Kineto trace with a realistic event mix.

    Each step has CPU ops with large ``args``, ``cudaLaunchKernel`` calls,
    kernels with long demangled names (some with braces and quotes), memcpy and
    memset events, launch-to-kernel flow events, Python frames and metadata.
    The same seed and parameters give the same bytes.

    Args:
        path: Output file; a ``.gz`` suffix writes gzip-compressed JSON. Its
            parent directory is created if missing.
        target_bytes: Uncompressed size to reach; whole steps are written, so
            the file ends up slightly larger.
        seed: Random seed.
        rank: Rank recorded in ``distributedInfo`` and mixed into the seed.
        world_size: World size recorded in ``distributedInfo``.
        layers_per_step: Number of op/launch/kernel groups per step.
        child_cpu_ops_per_layer: Small CPU ops per group.
        python_events_per_layer: Python-frame events per group.
        tensors_per_op: Number of input tensors in each op's ``args``.
        distributed_info_after: Whether ``distributedInfo`` is written after
            the ``traceEvents`` array, as some PyTorch versions do.

    Returns:
        The number of events written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return native.write_kineto_trace_large(
        str(path),
        target_bytes=target_bytes,
        seed=seed,
        rank=rank,
        world_size=world_size,
        layers_per_step=layers_per_step,
        child_cpu_ops_per_layer=child_cpu_ops_per_layer,
        python_events_per_layer=python_events_per_layer,
        tensors_per_op=tensors_per_op,
        distributed_info_after=distributed_info_after,
    )
