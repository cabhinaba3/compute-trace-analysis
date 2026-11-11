"""NVIDIA Jetson jtop hardware & power telemetry JSON adapter."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

JTOP_SAMPLES_SCHEMA = pl.Schema(
    {
        "elapsed_s": pl.Float64(),
        "cpu_util_pct": pl.Float64(),
        "gpu_util_pct": pl.Float64(),
        "gpu_mem_mb": pl.Float64(),
        "ram_mb": pl.Float64(),
        "swap_mb": pl.Float64(),
        "gpu_temp_c": pl.Float64(),
        "power_tot_w": pl.Float64(),
        "power_gpu_w": pl.Float64(),
        "cum_energy_j": pl.Float64(),
    }
)

_CPU_CORE_RE = re.compile(r"^CPU\d+$", re.IGNORECASE)


def _parse_timestamps_to_seconds(raw_ts: list[Any]) -> np.ndarray:
    n = len(raw_ts)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if raw_ts[0] is None:
        return np.arange(n, dtype=np.float64)
    if isinstance(raw_ts[0], str):
        try:
            secs = np.array([datetime.fromisoformat(str(t)).timestamp() for t in raw_ts], dtype=np.float64)
            return secs - secs[0]
        except ValueError:
            return np.arange(n, dtype=np.float64)

    vals = np.asarray(raw_ts, dtype=np.float64)
    span_ref = abs(float(vals[0])) if vals[0] != 0 else (abs(float(vals[-1])) if n > 1 else 1.0)
    # Auto-detect epoch timestamp resolution: ns (~1.7e18), us (~1.7e15), ms (~1.7e12), or s
    if span_ref > 1e16:
        scale = 1e-9
    elif span_ref > 1e13:
        scale = 1e-6
    elif span_ref > 1e10:
        scale = 1e-3
    else:
        scale = 1.0
    return (vals - vals[0]) * scale


def _pick_power_watts(item: dict[str, Any], keys: tuple[str, ...]) -> float:
    for k in keys:
        if k in item and item[k] is not None:
            val = float(item[k])
            # Standard jtop exports power rails in mW (e.g. 1500..60000 mW) unless explicitly small W
            unit = str(item.get("power_unit", "mW")).lower()
            return val if unit == "w" else (val * 1e-3)
    return 0.0


@register_adapter("jtop", version="1.0")
class JtopAdapter:
    format_name = "jtop"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        try:
            with path.open("rb") as fh:
                head = fh.read(8192).decode("utf-8", errors="ignore")
            return ("Power TOT" in head or "VDD_IN" in head) or ("CPU1" in head and '"GPU"' in head and '"RAM"' in head)
        except OSError:
            return False

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        frames: list[pl.DataFrame] = []
        for path in sorted(artifact.paths):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                continue

            cpu_keys = sorted(k for k in records[0] if _CPU_CORE_RE.match(str(k)))
            raw_ts = [r.get("time") for r in records]
            elapsed = _parse_timestamps_to_seconds(raw_ts)

            n = len(records)
            cpu_u = np.zeros(n, dtype=np.float64)
            gpu_u = np.zeros(n, dtype=np.float64)
            gmem = np.zeros(n, dtype=np.float64)
            ram = np.zeros(n, dtype=np.float64)
            swap = np.zeros(n, dtype=np.float64)
            temp = np.zeros(n, dtype=np.float64)
            p_tot = np.zeros(n, dtype=np.float64)
            p_gpu = np.zeros(n, dtype=np.float64)

            for i, item in enumerate(records):
                if cpu_keys:
                    cpu_u[i] = sum(float(item.get(ck, 0.0) or 0.0) for ck in cpu_keys) / len(cpu_keys)
                gpu_u[i] = float(item.get("GPU", item.get("gpu_utilization", 0.0)) or 0.0)
                gmem[i] = float(item.get("gpu_memory_usage", item.get("GPU_MEM", 0.0)) or 0.0)
                ram[i] = float(item.get("RAM", item.get("ram_usage", 0.0)) or 0.0)
                swap[i] = float(item.get("SWAP", item.get("swap_memory_usage", 0.0)) or 0.0)
                temp[i] = float(item.get("Temp gpu", item.get("Temp GPU", item.get("Temp tj", 0.0))) or 0.0)
                p_tot[i] = _pick_power_watts(item, ("Power TOT", "VDD_IN", "POM_5V_IN", "power_tot_mw"))
                p_gpu[i] = _pick_power_watts(item, ("VDD_GPU_SOC", "Power GPU", "POM_5V_GPU", "VDD_GPU"))

            # Trapezoidal cumulative energy integration in Joules (W * s)
            cum_e = np.zeros(n, dtype=np.float64)
            if n > 1:
                dt = np.maximum(0.0, np.diff(elapsed))
                if np.all(dt == 0.0):
                    dt = np.ones(n - 1, dtype=np.float64)
                step_e = 0.5 * (p_tot[1:] + p_tot[:-1]) * dt
                cum_e[1:] = np.cumsum(step_e)

            frames.append(
                pl.DataFrame(
                    {
                        "elapsed_s": elapsed,
                        "cpu_util_pct": cpu_u,
                        "gpu_util_pct": gpu_u,
                        "gpu_mem_mb": gmem,
                        "ram_mb": ram,
                        "swap_mb": swap,
                        "gpu_temp_c": temp,
                        "power_tot_w": p_tot,
                        "power_gpu_w": p_gpu,
                        "cum_energy_j": cum_e,
                    },
                    schema=JTOP_SAMPLES_SCHEMA,
                )
            )

        df = pl.concat(frames) if frames else pl.DataFrame(schema=JTOP_SAMPLES_SCHEMA)
        out_p = out_dir / "samples.parquet"
        df.write_parquet(out_p, compression="zstd")
        return {"samples": out_p}
