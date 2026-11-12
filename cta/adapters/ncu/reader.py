"""NVIDIA Nsight Compute adapter: ``.ncu-rep`` / ``.ncu-repz`` reports and CSV exports.

Reading order for a report: NVIDIA's ``ncu_report`` (``pip install "compute-trace-analysis[ncu]"``; wheels exist for
macOS, Linux and Windows, no ``ncu`` install needed) -> ``ncu --import --csv`` -> a ``.csv`` next to the report.
CSV exports (raw page, details page, or tall ``Metric Name``/``Metric Value`` tables) are always readable.

Metric semantics (verified against real H200 reports):

* FLOPs are *achieved* counts: ``smsp__sass_thread_inst_executed_op_{f,h,d}{add,mul,fma}_pred_on.sum.per_cycle_elapsed``
  times ``smsp__cycles_elapsed.avg`` (FMA = 2 FLOPs; packed half add/mul = 2, half FMA = 4) plus the tensor-core
  ``sm__ops_path_tensor_src_*.sum`` counters. The ``derived__sm__..._x2/_x4`` metrics are *peak* capacities (identical
  for every kernel) and must not be used as counts.
* DRAM traffic is ``dram__bytes_read.sum + dram__bytes_write.sum`` (there is no ``dram__bytes.sum``) and DRAM
  utilisation is ``gpu__dram_throughput``; ``gpu__compute_memory_throughput`` is the *maximum* over all memory units.
* A metric a report does not contain is ``null``, never ``0``.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from cta.core.alignment import enrich_ops_df
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

NCU_SCHEMA = pl.Schema(
    {
        "kernel_id": pl.Int32(),
        "name": pl.String(),
        "demangled_name": pl.String(),
        "canonical_name": pl.String(),
        "op_family": pl.String(),
        "duration_ns": pl.Float64(),
        "sm_throughput_pct": pl.Float64(),
        "dram_throughput_pct": pl.Float64(),
        "mem_throughput_pct": pl.Float64(),
        "tensor_pipe_pct": pl.Float64(),
        "occupancy_pct": pl.Float64(),
        "flops": pl.Float64(),
        "flops_tensor": pl.Float64(),
        "dram_bytes": pl.Float64(),
        "arith_intensity": pl.Float64(),
        "achieved_tflops": pl.Float64(),
        "stall_mem_pct": pl.Float64(),
        "stall_compute_pct": pl.Float64(),
        "grid_size": pl.Float64(),
        "block_size": pl.Float64(),
        "registers_per_thread": pl.Float64(),
    }
)

FINDINGS_SCHEMA = pl.Schema(
    {
        "kernel_id": pl.Int32(),
        "name": pl.String(),
        "rule": pl.String(),
        "section": pl.String(),
        "level": pl.String(),
        "title": pl.String(),
        "message": pl.String(),
        "speedup_pct": pl.Float64(),
        "speedup_type": pl.String(),
    }
)

Getter = Callable[[str], "float | None"]

# (instruction stem, FLOPs per thread-instruction)
_FLOP_TERMS: tuple[tuple[str, int], ...] = (
    ("fadd", 1), ("fmul", 1), ("ffma", 2),
    ("dadd", 1), ("dmul", 1), ("dfma", 2),
    ("hadd", 2), ("hmul", 2), ("hfma", 4),
)  # fmt: skip
_TENSOR_FLOP_METRICS = (
    "sm__ops_path_tensor_src_fp16.sum",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp64.sum",
    "sm__ops_path_tensor_src_fp8_sparsity_off.sum",
    "sm__ops_path_tensor_src_fp8_sparsity_on.sum",
)
_STALL_PREFIX, _STALL_SUFFIX = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
_STALL_MEM = ("long_scoreboard", "lg_throttle", "tex_throttle", "membar")
_STALL_COMPUTE = ("math_pipe_throttle", "gmma")
_STALL_IGNORED = ("selected", "not_selected")  # issue slots, not stalls
_STALL_ALL = (
    "barrier", "branch_resolving", "dispatch_stall", "drain", "gmma", "imc_miss", "lg_throttle", "long_scoreboard",
    "math_pipe_throttle", "membar", "mio_throttle", "misc", "no_instruction", "short_scoreboard", "sleeping",
    "tex_throttle", "wait",
)  # fmt: skip

# Unit -> multiplier to ns (time) or bytes. Anything else (%, inst, cycle, ...) is left unscaled.
_UNIT_SCALE: dict[str, float] = {
    "ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9,
    "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
}  # fmt: skip


def _first(get: Getter, *names: str) -> float | None:
    for n in names:
        v = get(n)
        if v is not None:
            return v
    return None


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip().replace(",", "").rstrip("%")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _stall_shares(get: Getter) -> tuple[float | None, float | None]:
    """(memory, compute) stall share in % of all stall cycles, from the per-reason issue-stall ratios."""
    vals = {r: get(f"{_STALL_PREFIX}{r}{_STALL_SUFFIX}") for r in _STALL_ALL}
    present = {r: v for r, v in vals.items() if v is not None}
    total = sum(present.values())
    if not present or total <= 0:
        return None, None
    return (
        100.0 * sum(present.get(r, 0.0) for r in _STALL_MEM) / total,
        100.0 * sum(present.get(r, 0.0) for r in _STALL_COMPUTE) / total,
    )


def _flops(get: Getter) -> tuple[float | None, float | None]:
    """(total FLOPs, tensor-core FLOPs) or (None, None) if the report has no FLOP counters at all."""
    cycles = get("smsp__cycles_elapsed.avg")
    cuda: float | None = None
    if cycles is not None:
        rates = [(w, get(f"smsp__sass_thread_inst_executed_op_{s}_pred_on.sum.per_cycle_elapsed")) for s, w in _FLOP_TERMS]
        present = [(w, r) for w, r in rates if r is not None]
        if present:
            cuda = sum(w * r for w, r in present) * cycles
    tens_vals = [get(m) for m in _TENSOR_FLOP_METRICS]
    tens_present = [v for v in tens_vals if v is not None]
    tensor = sum(tens_present) if tens_present else None
    if cuda is None and tensor is None:
        legacy = _first(get, "flops", "FLOPs", "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
        return legacy, None
    return (cuda or 0.0) + (tensor or 0.0), tensor


def _derive(get: Getter) -> dict[str, float | None]:
    """Canonical per-kernel metrics from a metric-name -> value lookup (report, raw CSV, details CSV or tall CSV)."""
    dur = _first(get, "gpu__time_duration.sum", "gpu__time_active.avg", "Duration")
    flops, flops_tensor = _flops(get)
    rd, wr = get("dram__bytes_read.sum"), get("dram__bytes_write.sum")
    dram_bytes = (rd or 0.0) + (wr or 0.0) if (rd is not None or wr is not None) else _first(get, "dram__bytes.sum", "dram_bytes", "DRAM Bytes")
    stall_mem, stall_cmp = _stall_shares(get)
    if stall_mem is None:  # legacy per-warp-active percentages
        stall_mem = _first(get, "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "Stall Long Scoreboard [%]")
        stall_cmp = _first(get, "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct", "Stall Math Pipe Throttle [%]")
    mem_pct = _first(get, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Memory Throughput", "Memory [%]")
    # True DRAM utilisation; sources that only report the overall memory throughput fall back to it.
    dram_pct = _first(
        get, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "DRAM Throughput",
    )  # fmt: skip
    if dram_pct is None:
        dram_pct = mem_pct
    ai = (flops / dram_bytes) if (flops is not None and dram_bytes) else None
    tflops = (flops / dur / 1e3) if (flops is not None and dur) else None  # FLOP/ns = GFLOP/s -> /1e3 = TFLOP/s
    return {
        "duration_ns": dur,
        "sm_throughput_pct": _first(get, "sm__throughput.avg.pct_of_peak_sustained_elapsed", "Compute (SM) Throughput", "SM [%]"),
        "dram_throughput_pct": dram_pct,
        "mem_throughput_pct": mem_pct,
        "tensor_pipe_pct": _first(
            get, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "Tensor Pipe [%]",
        ),  # fmt: skip
        "occupancy_pct": _first(get, "sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved Occupancy"),
        "flops": flops,
        "flops_tensor": flops_tensor,
        "dram_bytes": dram_bytes,
        "arith_intensity": ai,
        "achieved_tflops": tflops,
        "stall_mem_pct": stall_mem,
        "stall_compute_pct": stall_cmp,
        "grid_size": get("launch__grid_size"),
        "block_size": get("launch__block_size"),
        "registers_per_thread": get("launch__registers_per_thread"),
    }


# --------------------------------------------------------------------------------------- ncu_report (.ncu-rep)
_MSG_LEVELS = {0: "none", 1: "ok", 2: "optimization", 3: "warning", 4: "error"}
_SPEEDUP_TYPES = {0: "unknown", 1: "local", 2: "global"}


def _rows_from_report(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import ncu_report  # optional dependency (extra "ncu"); ImportError is handled by the caller

    ctx = ncu_report.load_report(str(path))
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    kernel_id = 0
    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            names = set(action.metric_names())

            def get(name: str, _a: Any = action, _names: set[str] = names) -> float | None:
                if name not in _names:
                    return None
                try:
                    v = float(_a.metric_by_name(name).as_double())
                except (TypeError, ValueError, RuntimeError):
                    return None
                return v if v == v else None  # NaN -> None

            kname = str(action.name())
            rows.append({"kernel_id": kernel_id, "name": kname, **_derive(get)})
            try:
                rules = action.rule_results_as_dicts()
            except Exception:  # rule evaluation is best-effort; metrics are the primary data
                rules = []
            for rule in rules:
                msg = rule.get("rule_message") or {}
                spd = rule.get("speedup_estimation") or {}
                findings.append(
                    {
                        "kernel_id": kernel_id,
                        "name": kname,
                        "rule": str(rule.get("rule_identifier", "")),
                        "section": str(rule.get("section_identifier", "")),
                        "level": _MSG_LEVELS.get(int(msg.get("type", 0)), "none"),
                        "title": str(msg.get("title", "")),
                        "message": str(msg.get("message", "")),
                        "speedup_pct": float(spd["speedup"]) if "speedup" in spd else None,
                        "speedup_type": _SPEEDUP_TYPES.get(int(spd["type"]), "unknown") if "type" in spd else None,
                    }
                )
            kernel_id += 1
    return rows, findings


# --------------------------------------------------------------------------------------------------- CSV
def _read_csv_text(path: Path) -> str:
    if path.suffix.lower() in (".ncu-rep", ".ncu-repz"):
        ncu_bin = shutil.which("ncu") or os.environ.get("NCU_PATH")
        if not ncu_bin:
            sibling = path.with_suffix(".csv")
            if sibling.exists():
                return sibling.read_text(encoding="utf-8", errors="ignore")
            raise RuntimeError(
                f"Cannot read '{path.name}': install the reader with pip install 'compute-trace-analysis[ncu]', "
                f"put the 'ncu' CLI on PATH, or provide '{sibling.name}'."
            )
        proc = subprocess.run([ncu_bin, "--import", str(path), "--csv", "--page", "raw"], check=True, capture_output=True, text=True)
        return proc.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


def _rows_from_csv(text: str) -> list[dict[str, Any]]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("==")]
    table = list(csv.reader(io.StringIO("\n".join(lines))))
    if len(table) < 2:
        return []
    header = [h.strip() for h in table[0]]
    out: list[dict[str, Any]] = []

    def emit(kernel_id: int, kname: str, values: dict[str, float]) -> None:
        out.append({"kernel_id": kernel_id, "name": kname, **_derive(values.get)})

    if "Metric Name" in header and "Metric Value" in header and "Kernel Name" in header:
        k_i, mn_i, mv_i = header.index("Kernel Name"), header.index("Metric Name"), header.index("Metric Value")
        id_i = header.index("ID") if "ID" in header else k_i
        unit_i = header.index("Metric Unit") if "Metric Unit" in header else None
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for r in table[1:]:
            if len(r) <= max(k_i, mn_i, mv_i, id_i):
                continue
            v = _to_float(r[mv_i])
            if v is None:
                continue
            if unit_i is not None and len(r) > unit_i:
                v *= _UNIT_SCALE.get(r[unit_i].strip(), 1.0)
            grouped.setdefault((r[id_i], r[k_i]), {}).setdefault(r[mn_i].strip(), v)
        for idx, ((_, kname), values) in enumerate(grouped.items()):
            emit(idx, kname, values)
        return out

    if "Kernel Name" not in header and "Function Name" not in header and "Name" not in header:
        return []
    name_i = next(header.index(c) for c in ("Kernel Name", "Function Name", "Name") if c in header)
    id_col = header.index("ID") if "ID" in header else None
    units: list[str] | None = None
    start = 1
    if len(table) > 1 and not table[1][name_i].strip():  # units row: kernel-name cell is empty
        units = [u.strip() for u in table[1]]
        start = 2
    scales = [_UNIT_SCALE.get(units[i], 1.0) if units and i < len(units) else 1.0 for i in range(len(header))]
    for n, r in enumerate(table[start:]):
        if len(r) <= name_i or not r[name_i].strip():
            continue
        row_vals: dict[str, float] = {}
        for i, col in enumerate(header):
            if i < len(r) and col and i != name_i:
                v = _to_float(r[i])
                if v is not None:
                    row_vals.setdefault(col, v * scales[i])
        kid = int(r[id_col]) if id_col is not None and r[id_col].strip().isdigit() else n
        emit(kid, r[name_i].strip(), row_vals)
    return out


# -------------------------------------------------------------------------------------------------- adapter
@register_adapter("ncu", version="2.0")
class NcuAdapter:
    format_name = "ncu"
    version = "2.0"

    def detect(self, path: Path) -> bool:
        suf = path.suffix.lower()
        if suf in (".ncu-rep", ".ncu-repz"):
            return True
        if suf == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    head = "".join(fh.readline() for _ in range(15))
                return any(tok in head for tok in ("Kernel Name", "sm__throughput", "gpu__time_duration", "==PROF=="))
            except OSError:
                return False
        return False

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        rows: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        for path in sorted(artifact.paths):
            if path.suffix.lower() in (".ncu-rep", ".ncu-repz"):
                try:
                    r, f = _rows_from_report(path)
                    rows += r
                    findings += f
                    continue
                except ImportError:
                    pass  # no ncu_report: fall through to the CLI / sibling-CSV route
            rows += _rows_from_csv(_read_csv_text(path))

        if rows:
            df = enrich_ops_df(pl.DataFrame(rows, schema_overrides={k: v for k, v in NCU_SCHEMA.items() if k not in ("name", "demangled_name", "canonical_name", "op_family")}, infer_schema_length=None), "name")
            df = df.select(list(NCU_SCHEMA.keys())).cast(NCU_SCHEMA)
        else:
            df = pl.DataFrame(schema=NCU_SCHEMA)
        fdf = pl.DataFrame(findings, schema=FINDINGS_SCHEMA) if findings else pl.DataFrame(schema=FINDINGS_SCHEMA)

        out: dict[str, Path] = {}
        for name, frame in (("ncu_kernels", df), ("ncu_findings", fdf)):
            p = out_dir / f"{name}.parquet"
            frame.write_parquet(p, compression="zstd")
            out[name] = p
        return out
