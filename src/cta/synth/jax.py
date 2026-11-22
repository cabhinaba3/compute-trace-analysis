"""Synthetic JAX/XProf traces in Chrome JSON, XPlane, and Perfetto formats."""

from __future__ import annotations

import pathlib

from cta import _native as native


def write_trace(
    path: pathlib.Path,
    *,
    steps: int = 3,
    devices: int = 2,
    gemm_us: float = 180.0,
    comm_us: float = 60.0,
    annotated: bool = True,
) -> pathlib.Path:
    """Writes a synthetic JAX/XProf Chrome trace (.json or .json.gz).

    Each step issues host launches for compute fusions, an NCCL AllReduce,
    a peer-to-peer copy, and a device memset across ``devices`` GPUs, plus
    a derived ``XLA Ops`` line and an out-of-step warmup op.

    Args:
        path: Output file; a ``.gz`` suffix writes gzip-compressed JSON.
        steps: Number of training steps.
        devices: Number of GPU devices.
        gemm_us: Duration of each compute fusion kernel in microseconds.
        comm_us: Duration of each NCCL AllReduce kernel in microseconds.
        annotated: Whether to include ``StepTraceAnnotation`` (``step_num``)
            markers.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_jax_trace(
        str(path),
        steps=steps,
        devices=devices,
        gemm_us=gemm_us,
        comm_us=comm_us,
        annotated=annotated,
    )
    return path


def write_xplane(
    path: pathlib.Path,
    *,
    steps: int = 3,
    devices: int = 2,
    gemm_us: float = 180.0,
    comm_us: float = 60.0,
    annotated: bool = True,
) -> pathlib.Path:
    """Writes the synthetic JAX workload as an XPlane protobuf.

    Args:
        path: Output file (e.g. ``host.xplane.pb`` or ``host.xplane.pb.gz``).
        steps: Number of training steps.
        devices: Number of GPU devices.
        gemm_us: Duration of each compute fusion kernel in microseconds.
        comm_us: Duration of each NCCL AllReduce kernel in microseconds.
        annotated: Whether to include ``StepTraceAnnotation`` markers.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_jax_xplane(
        str(path),
        steps=steps,
        devices=devices,
        gemm_us=gemm_us,
        comm_us=comm_us,
        annotated=annotated,
    )
    return path


def write_pftrace(
    path: pathlib.Path,
    *,
    steps: int = 3,
    devices: int = 2,
    gemm_us: float = 180.0,
    comm_us: float = 60.0,
    annotated: bool = True,
    legacy_chrome: bool = False,
) -> pathlib.Path:
    """Writes the synthetic JAX workload as a binary Perfetto trace.

    Args:
        path: Output file (e.g. ``trace.pftrace`` or ``trace.pftrace.gz``).
        steps: Number of training steps.
        devices: Number of GPU devices.
        gemm_us: Duration of each compute fusion kernel in microseconds.
        comm_us: Duration of each NCCL AllReduce kernel in microseconds.
        annotated: Whether to include ``StepTraceAnnotation`` markers.
        legacy_chrome: If True, encode packets as ``ChromeEventBundle``;
            otherwise encode modern ``TrackDescriptor`` + ``TrackEvent`` +
            ``InternedData`` packets.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_jax_pftrace(
        str(path),
        steps=steps,
        devices=devices,
        gemm_us=gemm_us,
        comm_us=comm_us,
        annotated=annotated,
        legacy_chrome=legacy_chrome,
    )
    return path


def write_large_trace(
    path: pathlib.Path,
    *,
    target_bytes: int = 64 << 20,
    seed: int = 1,
    devices: int = 2,
    layers_per_step: int = 16,
    python_events_per_layer: int = 4,
    annotated: bool = True,
) -> int:
    """Writes a large deterministic JAX/XProf Chrome trace for benchmarking.

    Args:
        path: Output file; a ``.gz`` suffix writes gzip-compressed JSON.
        target_bytes: Uncompressed JSON byte size target.
        seed: Random seed.
        devices: Number of GPU devices.
        layers_per_step: Number of fused layers per step.
        python_events_per_layer: Host Python frame events per layer.
        annotated: Whether to include ``StepTraceAnnotation`` markers.

    Returns:
        The number of trace events written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return native.write_jax_trace_large(
        str(path),
        target_bytes=target_bytes,
        seed=seed,
        devices=devices,
        layers_per_step=layers_per_step,
        python_events_per_layer=python_events_per_layer,
        annotated=annotated,
    )
