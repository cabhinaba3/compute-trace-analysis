"""NVIDIA Jetson jtop hardware & power telemetry JSON adapter.

A jtop export is a JSON list of samples. Each file becomes one device in the
``samples`` table: utilisation, memory, GPU temperature, power rails in watts
and the cumulative energy integrated from the total power.
"""

from __future__ import annotations

import datetime as datetime_module
import json
import pathlib
import re
from typing import Any

import numpy as np
import polars as pl

from cta.core import models
from cta.core import registry

JTOP_SAMPLES_SCHEMA = pl.Schema(
    {
        "device": pl.UInt32(),
        "elapsed_s": pl.Float64(),
        "cpu_util_pct": pl.Float64(),
        "gpu_util_pct": pl.Float64(),
        "gpu_mem_mb": pl.Float64(),
        "ram_mb": pl.Float64(),
        "swap_mb": pl.Float64(),
        "gpu_temp_c": pl.Float64(),
        "power_total_w": pl.Float64(),
        "power_gpu_w": pl.Float64(),
        "cumulative_energy_j": pl.Float64(),
    }
)

_CPU_CORE_RE = re.compile(r"^CPU\d+$", re.IGNORECASE)

# Power rail keys, in order of preference.
_TOTAL_POWER_KEYS = ("Power TOT", "VDD_IN", "POM_5V_IN", "power_tot_mw")
_GPU_POWER_KEYS = ("VDD_GPU_SOC", "Power GPU", "POM_5V_GPU", "VDD_GPU")


def _parse_timestamps_to_seconds(raw_timestamps: list[Any]) -> np.ndarray:
    """Converts jtop ``time`` values to seconds elapsed since the first sample.

    ISO-8601 strings are parsed; numbers are epoch timestamps whose unit (ns,
    us, ms or s) is inferred from their magnitude. If the first timestamp is
    missing, or the strings are not all ISO-8601, samples are taken to be one
    second apart.
    """
    num_samples = len(raw_timestamps)
    if num_samples == 0:
        return np.empty(0, dtype=np.float64)
    if raw_timestamps[0] is None:
        return np.arange(num_samples, dtype=np.float64)
    if isinstance(raw_timestamps[0], str):
        parse_iso = datetime_module.datetime.fromisoformat
        try:
            epoch_s = np.array(
                [
                    parse_iso(str(timestamp)).timestamp()
                    for timestamp in raw_timestamps
                ],
                dtype=np.float64,
            )
            return epoch_s - epoch_s[0]
        except ValueError:
            return np.arange(num_samples, dtype=np.float64)

    values = np.asarray(raw_timestamps, dtype=np.float64)
    if values[0] != 0:
        magnitude = abs(float(values[0]))
    else:
        magnitude = abs(float(values[-1])) if num_samples > 1 else 1.0
    # Auto-detect the epoch timestamp resolution from its magnitude: ns
    # (~1.7e18), us (~1.7e15), ms (~1.7e12) or s.
    if magnitude > 1e16:
        seconds_per_unit = 1e-9
    elif magnitude > 1e13:
        seconds_per_unit = 1e-6
    elif magnitude > 1e10:
        seconds_per_unit = 1e-3
    else:
        seconds_per_unit = 1.0
    return (values - values[0]) * seconds_per_unit


def _pick_power_watts(sample: dict[str, Any], keys: tuple[str, ...]) -> float:
    """Returns the first power rail in ``keys`` present in a sample, in watts.

    Returns 0.0 when none of the keys has a value.
    """
    for key in keys:
        if key in sample and sample[key] is not None:
            value = float(sample[key])
            # Standard jtop exports power rails in mW (e.g. 1500..60000 mW)
            # unless the sample explicitly says its unit is W.
            unit = str(sample.get("power_unit", "mW")).lower()
            return value if unit == "w" else (value * 1e-3)
    return 0.0


def _cumulative_energy_j(
    power_total_w: np.ndarray, elapsed_s: np.ndarray
) -> np.ndarray:
    """Integrates total power over time with the trapezoidal rule, in joules.

    Negative time steps count as zero; if every time step is zero, samples are
    taken to be one second apart.
    """
    num_samples = len(power_total_w)
    cumulative_energy_j = np.zeros(num_samples, dtype=np.float64)
    if num_samples > 1:
        dt_s = np.maximum(0.0, np.diff(elapsed_s))
        if np.all(dt_s == 0.0):
            dt_s = np.ones(num_samples - 1, dtype=np.float64)
        step_energy_j = 0.5 * (power_total_w[1:] + power_total_w[:-1]) * dt_s
        cumulative_energy_j[1:] = np.cumsum(step_energy_j)
    return cumulative_energy_j


def _samples_frame(records: list[dict[str, Any]], device: int) -> pl.DataFrame:
    """Builds the ``samples`` rows of one jtop export (a non-empty list)."""
    cpu_core_keys = sorted(
        key for key in records[0] if _CPU_CORE_RE.match(str(key))
    )
    raw_timestamps = [record.get("time") for record in records]
    elapsed_s = _parse_timestamps_to_seconds(raw_timestamps)

    num_samples = len(records)
    cpu_util_pct = np.zeros(num_samples, dtype=np.float64)
    gpu_util_pct = np.zeros(num_samples, dtype=np.float64)
    gpu_mem_mb = np.zeros(num_samples, dtype=np.float64)
    ram_mb = np.zeros(num_samples, dtype=np.float64)
    swap_mb = np.zeros(num_samples, dtype=np.float64)
    gpu_temp_c = np.zeros(num_samples, dtype=np.float64)
    power_total_w = np.zeros(num_samples, dtype=np.float64)
    power_gpu_w = np.zeros(num_samples, dtype=np.float64)

    for i, record in enumerate(records):
        if cpu_core_keys:
            core_util_sum = sum(
                float(record.get(key, 0.0) or 0.0) for key in cpu_core_keys
            )
            cpu_util_pct[i] = core_util_sum / len(cpu_core_keys)
        gpu_util = record.get("GPU", record.get("gpu_utilization", 0.0))
        gpu_util_pct[i] = float(gpu_util or 0.0)
        gpu_mem = record.get("gpu_memory_usage", record.get("GPU_MEM", 0.0))
        gpu_mem_mb[i] = float(gpu_mem or 0.0)
        ram = record.get("RAM", record.get("ram_usage", 0.0))
        ram_mb[i] = float(ram or 0.0)
        swap = record.get("SWAP", record.get("swap_memory_usage", 0.0))
        swap_mb[i] = float(swap or 0.0)
        gpu_temp = record.get(
            "Temp gpu", record.get("Temp GPU", record.get("Temp tj", 0.0))
        )
        gpu_temp_c[i] = float(gpu_temp or 0.0)
        power_total_w[i] = _pick_power_watts(record, _TOTAL_POWER_KEYS)
        power_gpu_w[i] = _pick_power_watts(record, _GPU_POWER_KEYS)

    return pl.DataFrame(
        {
            "device": np.full(num_samples, device, dtype=np.uint32),
            "elapsed_s": elapsed_s,
            "cpu_util_pct": cpu_util_pct,
            "gpu_util_pct": gpu_util_pct,
            "gpu_mem_mb": gpu_mem_mb,
            "ram_mb": ram_mb,
            "swap_mb": swap_mb,
            "gpu_temp_c": gpu_temp_c,
            "power_total_w": power_total_w,
            "power_gpu_w": power_gpu_w,
            "cumulative_energy_j": _cumulative_energy_j(
                power_total_w, elapsed_s
            ),
        },
        schema=JTOP_SAMPLES_SCHEMA,
    )


@registry.register_adapter("jtop", version="1.2")
class JtopAdapter:
    """Reads jtop JSON telemetry exports into a ``samples`` table.

    Attributes:
        format_name: The registered trace format name, ``"jtop"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "jtop"
    version = "1.2"

    def detect(self, path: pathlib.Path) -> bool:
        """Checks whether a file looks like a jtop JSON export.

        Args:
            path: The candidate file.

        Returns:
            True for a .json file whose first 8 KiB name a jtop power rail
            (``Power TOT``/``VDD_IN``) or the ``CPU1``, ``GPU`` and ``RAM``
            keys.
        """
        if path.suffix.lower() != ".json":
            return False
        try:
            with path.open("rb") as fh:
                head = fh.read(8192).decode("utf-8", errors="ignore")
            has_power_rail = "Power TOT" in head or "VDD_IN" in head
            has_usage_keys = (
                "CPU1" in head and '"GPU"' in head and '"RAM"' in head
            )
            return has_power_rail or has_usage_keys
        except OSError:
            return False

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the artifact's jtop exports to a ``samples`` Parquet table.

        Each file becomes one device, numbered in sorted path order. Files
        that are not a non-empty JSON list are skipped.

        Args:
            artifact: The jtop artifact to read.
            out_dir: The directory to write ``samples.parquet`` to.

        Returns:
            The table name ``"samples"`` mapped to its Parquet path.
        """
        frames: list[pl.DataFrame] = []
        for device, path in enumerate(sorted(artifact.paths)):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                continue
            frames.append(_samples_frame(records, device))

        samples = (
            pl.concat(frames)
            if frames
            else pl.DataFrame(schema=JTOP_SAMPLES_SCHEMA)
        )
        samples_path = out_dir / "samples.parquet"
        samples.write_parquet(samples_path, compression="zstd")
        return {"samples": samples_path}
