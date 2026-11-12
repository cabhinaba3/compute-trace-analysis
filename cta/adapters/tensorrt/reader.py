"""TensorRT trtexec timing, layer profile, and engine layer-info JSON adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cta.adapters.nsys.reader import KERNELS_SCHEMA, STEPS_SCHEMA
from cta.core.alignment import enrich_ops_df
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

TRT_PROFILE_SCHEMA = pl.Schema(
    {
        "name": pl.String(),
        "op_family": pl.String(),
        "avg_ms": pl.Float64(),
        "median_ms": pl.Float64(),
        "pct_time": pl.Float64(),
        "is_fused": pl.Boolean(),
    }
)

TRT_ENGINE_SCHEMA = pl.Schema(
    {
        "name": pl.String(),
        "layer_type": pl.String(),
        "precision": pl.String(),
        "tactic": pl.String(),
        "is_fused": pl.Boolean(),
    }
)


def _peek_json_head(path: Path, nbytes: int = 8192) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(nbytes).decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _is_fused_layer_name(name: str) -> bool:
    return "+" in name or "||" in name or "fused" in name.lower() or "myelin" in name.lower()


@register_adapter("trtexec_timing", version="1.0")
class TrtTimingAdapter:
    format_name = "trtexec_timing"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        head = _peek_json_head(path)
        return "startComputeMs" in head and "endComputeMs" in head

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        step_rows: list[dict[str, object]] = []
        ev_rows: list[dict[str, object]] = []
        step_idx = 0

        for path in sorted(artifact.paths):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for item in records:
                if not isinstance(item, dict) or "startComputeMs" not in item or "endComputeMs" not in item:
                    continue
                scmp = int(round(float(item["startComputeMs"]) * 1e6))
                ecmp = int(round(float(item["endComputeMs"]) * 1e6))
                sh2d = int(round(float(item["startH2dMs"]) * 1e6)) if item.get("startH2dMs") is not None else scmp
                eh2d = int(round(float(item["endH2dMs"]) * 1e6)) if item.get("endH2dMs") is not None else sh2d
                sd2h = int(round(float(item["startD2hMs"]) * 1e6)) if item.get("startD2hMs") is not None else ecmp
                ed2h = int(round(float(item["endD2hMs"]) * 1e6)) if item.get("endD2hMs") is not None else sd2h

                q_start = min(sh2d, scmp, sd2h)
                q_end = max(eh2d, ecmp, ed2h)
                if q_end <= q_start:
                    continue

                step_rows.append(
                    {"step": step_idx, "start_ns": q_start, "end_ns": q_end, "duration_ns": q_end - q_start}
                )
                if eh2d > sh2d:
                    ev_rows.append(
                        {
                            "step": step_idx,
                            "scope": "trtexec_query",
                            "name": "Memcpy_H2D",
                            "category": "memcpy",
                            "start_ns": sh2d,
                            "end_ns": eh2d,
                            "duration_ns": eh2d - sh2d,
                            "bytes": 0,
                            "stream": 1,
                        }
                    )
                if ecmp > scmp:
                    ev_rows.append(
                        {
                            "step": step_idx,
                            "scope": "trtexec_query",
                            "name": "trt_compute",
                            "category": "compute",
                            "start_ns": scmp,
                            "end_ns": ecmp,
                            "duration_ns": ecmp - scmp,
                            "bytes": 0,
                            "stream": 0,
                        }
                    )
                if ed2h > sd2h:
                    ev_rows.append(
                        {
                            "step": step_idx,
                            "scope": "trtexec_query",
                            "name": "Memcpy_D2H",
                            "category": "memcpy",
                            "start_ns": sd2h,
                            "end_ns": ed2h,
                            "duration_ns": ed2h - sd2h,
                            "bytes": 0,
                            "stream": 1,
                        }
                    )
                step_idx += 1

        steps_df = pl.DataFrame(step_rows, schema=STEPS_SCHEMA) if step_rows else pl.DataFrame(schema=STEPS_SCHEMA)
        if ev_rows:
            kernels_df = enrich_ops_df(pl.DataFrame(ev_rows), "name").select(list(KERNELS_SCHEMA.keys())).cast(KERNELS_SCHEMA)
        else:
            kernels_df = pl.DataFrame(schema=KERNELS_SCHEMA)

        sp, kp = out_dir / "steps.parquet", out_dir / "kernels.parquet"
        steps_df.write_parquet(sp, compression="zstd")
        kernels_df.write_parquet(kp, compression="zstd")
        return {"steps": sp, "kernels": kp}


@register_adapter("trtexec_profile", version="1.0")
class TrtLayerProfileAdapter:
    format_name = "trtexec_profile"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        head = _peek_json_head(path)
        return ("averageMs" in head or "timeMs" in head) and "percentage" in head

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        rows: list[dict[str, object]] = []
        for path in sorted(artifact.paths):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for item in records:
                if not isinstance(item, dict) or "name" not in item:
                    continue
                nm = str(item["name"])
                avg_ms = float(item.get("averageMs", item.get("timeMs", 0.0)) or 0.0)
                med_ms = float(item.get("medianMs", avg_ms) or avg_ms)
                pct = float(item.get("percentage", 0.0) or 0.0)
                rows.append(
                    {
                        "name": nm,
                        "avg_ms": avg_ms,
                        "median_ms": med_ms,
                        "pct_time": pct,
                        "is_fused": _is_fused_layer_name(nm),
                    }
                )

        if rows:
            df = (
                enrich_ops_df(pl.DataFrame(rows), "name")
                .select("name", "op_family", "avg_ms", "median_ms", "pct_time", "is_fused")
                .cast(TRT_PROFILE_SCHEMA)
            )
        else:
            df = pl.DataFrame(schema=TRT_PROFILE_SCHEMA)

        out_p = out_dir / "trt_profile.parquet"
        df.write_parquet(out_p, compression="zstd")
        return {"trt_profile": out_p}


def _extract_trt_precision(layer: dict[str, Any]) -> str:
    raw_p = str(layer.get("Precision", layer.get("ParameterType", ""))).upper()
    if not raw_p:
        for port_key in ("Inputs", "Outputs"):
            ports = layer.get(port_key)
            if isinstance(ports, list) and ports and isinstance(ports[0], dict):
                raw_p = str(ports[0].get("Format/Datatype", ports[0].get("Datatype", ""))).upper()
                if raw_p:
                    break
    if "INT8" in raw_p:
        return "INT8"
    if "FP8" in raw_p:
        return "FP8"
    if "HALF" in raw_p or "FP16" in raw_p:
        return "FP16"
    if "BF16" in raw_p or "BFLOAT16" in raw_p:
        return "BF16"
    if "FLOAT" in raw_p or "FP32" in raw_p:
        return "FP32"
    return raw_p or "UNKNOWN"


@register_adapter("trtexec_engine", version="1.0")
class TrtEngineInfoAdapter:
    format_name = "trtexec_engine"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        head = _peek_json_head(path)
        return '"Layers"' in head and ('"LayerType"' in head or '"TacticValue"' in head or '"ParameterType"' in head)

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        rows: list[dict[str, object]] = []
        for path in sorted(artifact.paths):
            doc = json.loads(path.read_text(encoding="utf-8"))
            layers = doc.get("Layers", []) if isinstance(doc, dict) else []
            for entry in layers:
                if isinstance(entry, str):
                    rows.append(
                        {"name": entry, "layer_type": "Unknown", "precision": "UNKNOWN", "tactic": "", "is_fused": _is_fused_layer_name(entry)}
                    )
                elif isinstance(entry, dict):
                    nm = str(entry.get("Name", ""))
                    ltype = str(entry.get("LayerType", entry.get("ParameterType", "Unknown")))
                    tactic = str(entry.get("TacticName", entry.get("TacticValue", "")))
                    rows.append(
                        {
                            "name": nm,
                            "layer_type": ltype,
                            "precision": _extract_trt_precision(entry),
                            "tactic": tactic,
                            "is_fused": _is_fused_layer_name(nm),
                        }
                    )

        df = pl.DataFrame(rows, schema=TRT_ENGINE_SCHEMA) if rows else pl.DataFrame(schema=TRT_ENGINE_SCHEMA)
        out_p = out_dir / "trt_engine.parquet"
        df.write_parquet(out_p, compression="zstd")
        return {"trt_engine": out_p}
