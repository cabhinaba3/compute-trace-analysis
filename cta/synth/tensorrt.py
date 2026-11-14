"""Synthetic trtexec timing and per-layer profile exports."""

from __future__ import annotations

import json
import pathlib


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
    query_timings = []
    query_start_ms = 10.0
    for _ in range(queries):
        start_h2d_ms = query_start_ms
        end_h2d_ms = start_h2d_ms + h2d_ms
        start_compute_ms = max(start_h2d_ms, end_h2d_ms - overlap_ms)
        end_compute_ms = start_compute_ms + compute_ms
        start_d2h_ms = max(start_compute_ms, end_compute_ms - overlap_ms * 0.5)
        end_d2h_ms = start_d2h_ms + d2h_ms
        query_timings.append(
            {
                "startH2dMs": start_h2d_ms,
                "endH2dMs": end_h2d_ms,
                "startComputeMs": start_compute_ms,
                "endComputeMs": end_compute_ms,
                "startD2hMs": start_d2h_ms,
                "endD2hMs": end_d2h_ms,
                "h2dMs": h2d_ms,
                "computeMs": compute_ms,
                "d2hMs": d2h_ms,
                "latencyMs": end_d2h_ms - start_h2d_ms,
            }
        )
        query_start_ms = end_d2h_ms + idle_gap_ms
    path.write_text(json.dumps(query_timings, indent=2))
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

    if fused:
        layer_profile = [
            {"count": 100},
            {
                "name": "Conv_0 + Bias_1 + Relu_2",
                "timeMs": 80.0,
                "averageMs": 0.8,
                "medianMs": 0.79,
                "percentage": 66.7,
            },
            {
                "name": "Linear_3",
                "timeMs": 40.0,
                "averageMs": 0.4,
                "medianMs": 0.4,
                "percentage": 33.3,
            },
        ]
        layer_info = {
            "Layers": [
                {
                    "Name": "Conv_0 + Bias_1 + Relu_2",
                    "LayerType": "CaskConvolution",
                    "Precision": precision,
                    "TacticName": "sm87_xmma_fprop",
                },
                {
                    "Name": "Linear_3",
                    "LayerType": "CaskGemm",
                    "Precision": precision,
                    "TacticName": "sm87_gemm",
                },
            ]
        }
    else:
        layer_profile = [
            {"count": 100},
            {
                "name": "Conv_0",
                "timeMs": 110.0,
                "averageMs": 1.1,
                "medianMs": 1.1,
                "percentage": 55.0,
            },
            {
                "name": "Bias_1",
                "timeMs": 20.0,
                "averageMs": 0.2,
                "medianMs": 0.2,
                "percentage": 10.0,
            },
            {
                "name": "Relu_2",
                "timeMs": 15.0,
                "averageMs": 0.15,
                "medianMs": 0.15,
                "percentage": 7.5,
            },
            {
                "name": "Linear_3",
                "timeMs": 55.0,
                "averageMs": 0.55,
                "medianMs": 0.55,
                "percentage": 27.5,
            },
        ]
        layer_info = {
            "Layers": [
                {
                    "Name": "Conv_0",
                    "LayerType": "Convolution",
                    "Precision": precision,
                    "TacticName": "cudnn_conv",
                },
                {
                    "Name": "Bias_1",
                    "LayerType": "ElementWise",
                    "Precision": precision,
                    "TacticName": "ew_add",
                },
                {
                    "Name": "Relu_2",
                    "LayerType": "Activation",
                    "Precision": precision,
                    "TacticName": "relu",
                },
                {
                    "Name": "Linear_3",
                    "LayerType": "MatrixMultiply",
                    "Precision": precision,
                    "TacticName": "cublas_gemm",
                },
            ]
        }

    profile_path.write_text(json.dumps(layer_profile, indent=2))
    engine_path.write_text(json.dumps(layer_info, indent=2))
    return profile_path, engine_path
