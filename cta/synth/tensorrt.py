"""Synthetic trtexec timing and per-layer profile exports."""

from __future__ import annotations

import pathlib

from cta import _native as native


def write_trtexec_timing(
    path: pathlib.Path,
    *,
    queries: int = 20,
    h2d_ms: float = 0.4,
    compute_ms: float = 2.0,
    d2h_ms: float = 0.3,
    overlap_ms: float = 0.2,
    idle_gap_ms: float = 0.1,
) -> pathlib.Path:
    """Writes a `trtexec --exportTimes` JSON file.

    Each query copies its input to the device, computes and copies the result
    back. Compute starts ``overlap_ms`` before the input copy ends and the
    output copy starts ``overlap_ms / 2`` before compute ends, but no stage
    starts before the previous one.

    Args:
        path: Output JSON file; its parent directory is created if missing.
        queries: Number of inference queries.
        h2d_ms: Host-to-device copy time per query in milliseconds.
        compute_ms: Compute time per query in milliseconds.
        d2h_ms: Device-to-host copy time per query in milliseconds.
        overlap_ms: Overlap between a copy and compute in milliseconds.
        idle_gap_ms: Idle time between consecutive queries in milliseconds.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    native.write_trtexec_timing(
        str(path),
        queries=queries,
        h2d_ms=h2d_ms,
        compute_ms=compute_ms,
        d2h_ms=d2h_ms,
        overlap_ms=overlap_ms,
        idle_gap_ms=idle_gap_ms,
    )
    return path


def write_trtexec_layers(
    profile_path: pathlib.Path,
    engine_path: pathlib.Path,
    *,
    fused: bool = False,
    precision: str = "FP16",
) -> tuple[pathlib.Path, pathlib.Path]:
    """Writes companion `trtexec --exportProfile` and `--exportLayerInfo` files.

    Both files describe a Conv -> Bias -> Relu -> Linear network. The fused
    variant merges the first three layers into one ``CaskConvolution`` layer.

    Args:
        profile_path: Output per-layer timing JSON (``--exportProfile``).
        engine_path: Output engine layer-information JSON
            (``--exportLayerInfo``).
        fused: Whether to write the fused-layer variant.
        precision: Precision recorded for every engine layer.

    Returns:
        ``(profile_path, engine_path)``.
    """
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    native.write_trtexec_layers(
        str(profile_path), str(engine_path), fused=fused, precision=precision
    )
    return profile_path, engine_path
