"""NVIDIA Nsight Compute (.ncu-rep and exported .csv) kernel profile adapter."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from pathlib import Path

import polars as pl

from cta.core.alignment import enrich_ops_df
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

NCU_SCHEMA = pl.Schema(
    {
        "name": pl.String(),
        "demangled_name": pl.String(),
        "canonical_name": pl.String(),
        "op_family": pl.String(),
        "duration_ns": pl.Float64(),
        "sm_throughput_pct": pl.Float64(),
        "dram_throughput_pct": pl.Float64(),
        "tensor_pipe_pct": pl.Float64(),
        "occupancy_pct": pl.Float64(),
        "flops": pl.Float64(),
        "dram_bytes": pl.Float64(),
        "arith_intensity": pl.Float64(),
        "achieved_tflops": pl.Float64(),
        "stall_mem_pct": pl.Float64(),
        "stall_compute_pct": pl.Float64(),
    }
)

_COL_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("Kernel Name", "Function Name", "Name"),
    "duration_ns": (
        "gpu__time_duration.sum",
        "gpu__time_active.avg",
        "Duration",
    ),
    "sm_throughput_pct": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "Compute (SM) Throughput",
        "SM [%] ",
        "SM [%]",
    ),
    "dram_throughput_pct": (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory Throughput",
        "Memory [%]",
    ),
    "tensor_pipe_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "Tensor Pipe [%]",
    ),
    "occupancy_pct": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "Achieved Occupancy",
    ),
    "flops": (
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "flops",
        "FLOPs",
    ),
    "dram_bytes": (
        "dram__bytes.sum",
        "dram_bytes",
        "DRAM Bytes",
    ),
    "stall_mem_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "Stall Long Scoreboard [%]",
    ),
    "stall_compute_pct": (
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "Stall Math Pipe Throttle [%]",
    ),
}


def _clean_float(val: str | None) -> float:
    if not val:
        return 0.0
    s = val.strip().replace(",", "").rstrip("%")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _read_ncu_csv_text(path: Path) -> str:
    if path.suffix.lower() == ".ncu-rep":
        ncu_bin = shutil.which("ncu") or os.environ.get("NCU_PATH")
        if not ncu_bin:
            sibling = path.with_suffix(".csv")
            if sibling.exists():
                return sibling.read_text(encoding="utf-8", errors="ignore")
            raise RuntimeError(f"Cannot import '{path.name}': 'ncu' CLI not found on PATH and no '{sibling.name}' exists.")
        proc = subprocess.run(
            [ncu_bin, "--import", str(path), "--csv", "--page", "raw"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


@register_adapter("ncu", version="1.0")
class NcuAdapter:
    format_name = "ncu"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        suf = path.suffix.lower()
        if suf == ".ncu-rep":
            return True
        if suf == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    head = "".join(fh.readline() for _ in range(15))
                return any(
                    tok in head
                    for tok in ("Kernel Name", "sm__throughput", "gpu__time_duration", "==PROF==")
                )
            except OSError:
                return False
        return False

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        rows_out: list[dict[str, object]] = []
        for path in sorted(artifact.paths):
            raw_text = _read_ncu_csv_text(path)
            lines = [ln for ln in raw_text.splitlines() if ln.strip() and not ln.startswith("==PROF==")]
            if not lines:
                continue

            reader = list(csv.reader(io.StringIO("\n".join(lines))))
            if len(reader) < 2:
                continue
            header = [h.strip() for h in reader[0]]
            col_idx: dict[str, int] = {}
            for canon, aliases in _COL_ALIASES.items():
                for alias in aliases:
                    if alias in header:
                        col_idx[canon] = header.index(alias)
                        break

            # NCU raw CSV often includes a units row immediately after the header (e.g., "", "ns", "%", "byte")
            start_row = 1
            if len(reader) >= 2 and "duration_ns" in col_idx:
                probe = reader[1][col_idx["duration_ns"]].strip().lower()
                if probe in ("ns", "us", "ms", "s", "usecond", "nsecond", "msecond", "%"):
                    start_row = 2
                    dur_unit = probe
                else:
                    dur_unit = "ns"
            else:
                dur_unit = "ns"

            unit_mul = {"ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9}.get(
                dur_unit, 1.0
            )

            # Support both wide format (1 row per kernel invocation) and tall NCU details format ("Metric Name", "Metric Value")
            if "Metric Name" in header and "Metric Value" in header and "Kernel Name" in header:
                k_i = header.index("Kernel Name")
                id_i = header.index("ID") if "ID" in header else k_i
                mn_i = header.index("Metric Name")
                mv_i = header.index("Metric Value")
                grouped: dict[tuple[str, str], dict[str, str]] = {}
                for r in reader[1:]:
                    if len(r) <= max(k_i, mn_i, mv_i):
                        continue
                    key = (r[id_i], r[k_i])
                    grouped.setdefault(key, {"Kernel Name": r[k_i]})[r[mn_i].strip()] = r[mv_i].strip()
                for gdict in grouped.values():
                    kname = gdict.get("Kernel Name", "unknown")
                    vals: dict[str, float] = {}
                    for canon, aliases in _COL_ALIASES.items():
                        if canon == "name":
                            continue
                        vals[canon] = next((_clean_float(gdict[a]) for a in aliases if a in gdict), 0.0)
                    dur_ns = vals["duration_ns"]
                    flops = vals["flops"]
                    dbytes = vals["dram_bytes"]
                    ai = (flops / dbytes) if dbytes > 0 else 0.0
                    tflops = (flops / dur_ns / 1e3) if dur_ns > 0 else 0.0
                    rows_out.append({"name": kname, **vals, "arith_intensity": ai, "achieved_tflops": tflops})
            else:
                for r in reader[start_row:]:
                    if not r or "name" not in col_idx or len(r) <= col_idx["name"]:
                        continue
                    kname = r[col_idx["name"]].strip()
                    if not kname:
                        continue
                    vals = {
                        canon: (_clean_float(r[col_idx[canon]]) if canon in col_idx and len(r) > col_idx[canon] else 0.0)
                        for canon in _COL_ALIASES
                        if canon != "name"
                    }
                    dur_ns = vals["duration_ns"] * unit_mul
                    vals["duration_ns"] = dur_ns
                    flops = vals["flops"]
                    dbytes = vals["dram_bytes"]
                    ai = (flops / dbytes) if dbytes > 0 else 0.0
                    tflops = (flops / dur_ns / 1e3) if dur_ns > 0 else 0.0
                    rows_out.append({"name": kname, **vals, "arith_intensity": ai, "achieved_tflops": tflops})

        if rows_out:
            df = enrich_ops_df(pl.DataFrame(rows_out), "name").select(list(NCU_SCHEMA.keys())).cast(NCU_SCHEMA)
        else:
            df = pl.DataFrame(schema=NCU_SCHEMA)

        out_p = out_dir / "ncu_kernels.parquet"
        df.write_parquet(out_p, compression="zstd")
        return {"ncu_kernels": out_p}
