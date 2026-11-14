"""NVIDIA Nsight Compute adapter for reports and CSV exports.

Reading order for a ``.ncu-rep`` / ``.ncu-repz`` report: NVIDIA's
``ncu_report`` (``pip install "compute-trace-analysis[ncu]"``; wheels exist
for macOS, Linux and Windows, no ``ncu`` install needed) -> ``ncu --import
--csv`` -> a ``.csv`` next to the report. CSV exports (raw page, details page,
or tall ``Metric Name``/``Metric Value`` tables) are always readable.

Metric semantics (verified against real H200 reports):

* FLOPs are *achieved* counts:
  ``smsp__sass_thread_inst_executed_op_{f,h,d}{add,mul,fma}_pred_on.sum.per_cycle_elapsed``
  times ``smsp__cycles_elapsed.avg`` (FMA = 2 FLOPs; packed half add/mul = 2,
  half FMA = 4) plus the tensor-core ``sm__ops_path_tensor_src_*.sum``
  counters. The ``derived__sm__..._x2/_x4`` metrics are *peak* capacities
  (identical for every kernel) and must not be used as counts.
* DRAM traffic is ``dram__bytes_read.sum + dram__bytes_write.sum`` (there is
  no ``dram__bytes.sum``) and DRAM utilisation is ``gpu__dram_throughput``;
  ``gpu__compute_memory_throughput`` is the *maximum* over all memory units.
* A metric a report does not contain is ``null``, never ``0``.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import os
import pathlib
import shutil
import subprocess
from typing import Any

import polars as pl

from cta.core import alignment
from cta.core import models
from cta.core import registry

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
        "arithmetic_intensity": pl.Float64(),
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

# A metric lookup by name: the metric's value, or None if it is absent.
Getter = Callable[[str], "float | None"]

_REPORT_SUFFIXES = (".ncu-rep", ".ncu-repz")
# Text that marks a CSV file as an Nsight Compute export.
_CSV_MARKERS = (
    "Kernel Name",
    "sm__throughput",
    "gpu__time_duration",
    "==PROF==",
)
# Kernel-name columns of a wide CSV export, in order of preference.
_KERNEL_NAME_COLUMNS = ("Kernel Name", "Function Name", "Name")
# Text columns of NCU_SCHEMA; alignment.enrich_ops_df fills all but "name".
_NCU_TEXT_COLUMNS = ("name", "demangled_name", "canonical_name", "op_family")

# Achieved per-cycle thread-instruction rate of each FLOP instruction stem:
# f"{_FLOP_RATE_PREFIX}{stem}{_FLOP_RATE_SUFFIX}".
_FLOP_RATE_PREFIX = "smsp__sass_thread_inst_executed_op_"
_FLOP_RATE_SUFFIX = "_pred_on.sum.per_cycle_elapsed"
# (instruction stem, FLOPs per thread-instruction)
_FLOP_TERMS: tuple[tuple[str, int], ...] = (
    ("fadd", 1),
    ("fmul", 1),
    ("ffma", 2),
    ("dadd", 1),
    ("dmul", 1),
    ("dfma", 2),
    ("hadd", 2),
    ("hmul", 2),
    ("hfma", 4),
)
_TENSOR_FLOP_METRICS = (
    "sm__ops_path_tensor_src_fp16.sum",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp64.sum",
    "sm__ops_path_tensor_src_fp8_sparsity_off.sum",
    "sm__ops_path_tensor_src_fp8_sparsity_on.sum",
)
_STALL_PREFIX = "smsp__average_warps_issue_stalled_"
_STALL_SUFFIX = "_per_issue_active.ratio"
_STALL_MEM = ("long_scoreboard", "lg_throttle", "tex_throttle", "membar")
_STALL_COMPUTE = ("math_pipe_throttle", "gmma")
_STALL_IGNORED = ("selected", "not_selected")  # issue slots, not stalls
_STALL_ALL = (
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "gmma",
    "imc_miss",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instruction",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
)

# Source names of each derived metric, in order of preference: ncu metric
# names, details-page column names and legacy names.
_DURATION_METRICS = (
    "gpu__time_duration.sum",
    "gpu__time_active.avg",
    "Duration",
)
_DRAM_BYTES_METRICS = ("dram__bytes.sum", "dram_bytes", "DRAM Bytes")
_LEGACY_FLOP_METRICS = (
    "flops",
    "FLOPs",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
)
_LEGACY_STALL_MEM_METRICS = (
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "Stall Long Scoreboard [%]",
)
_LEGACY_STALL_COMPUTE_METRICS = (
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "Stall Math Pipe Throttle [%]",
)
_MEM_THROUGHPUT_METRICS = (
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "Memory Throughput",
    "Memory [%]",
)
_DRAM_THROUGHPUT_METRICS = (
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM Throughput",
)
_SM_THROUGHPUT_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Compute (SM) Throughput",
    "SM [%]",
)
_TENSOR_PIPE_METRICS = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "Tensor Pipe [%]",
)
_OCCUPANCY_METRICS = (
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "Achieved Occupancy",
)

# Unit -> multiplier to ns (time) or bytes. Anything else (%, inst, cycle,
# ...) is left unscaled.
_SCALE_BY_UNIT: dict[str, float] = {
    "ns": 1.0,
    "nsecond": 1.0,
    "us": 1e3,
    "usecond": 1e3,
    "ms": 1e6,
    "msecond": 1e6,
    "s": 1e9,
    "second": 1e9,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "Tbyte": 1e12,
}


def _first_present(get: Getter, *metric_names: str) -> float | None:
    """Returns the value of the first metric ``get`` knows, else None."""
    for metric_name in metric_names:
        value = get(metric_name)
        if value is not None:
            return value
    return None


def _to_float(raw: str | None) -> float | None:
    """Parses a CSV cell such as ``"1,234.5"`` or ``"12%"``; None if invalid."""
    if raw is None:
        return None
    text = raw.strip().replace(",", "").rstrip("%")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _stall_shares(get: Getter) -> tuple[float | None, float | None]:
    """Returns the memory and compute stall shares in % of all stalls.

    The shares come from the per-reason issue-stall ratios; both are None
    when the report has none of them or they sum to zero.
    """
    ratio_by_reason = {
        reason: get(f"{_STALL_PREFIX}{reason}{_STALL_SUFFIX}")
        for reason in _STALL_ALL
    }
    present = {
        reason: ratio
        for reason, ratio in ratio_by_reason.items()
        if ratio is not None
    }
    total = sum(present.values())
    if not present or total <= 0:
        return None, None
    mem_ratio = sum(present.get(reason, 0.0) for reason in _STALL_MEM)
    compute_ratio = sum(present.get(reason, 0.0) for reason in _STALL_COMPUTE)
    return 100.0 * mem_ratio / total, 100.0 * compute_ratio / total


def _flops(get: Getter) -> tuple[float | None, float | None]:
    """Returns the total and tensor-core FLOP counts of a kernel.

    Without any achieved-FLOP counter, a legacy FLOP count is returned with
    a None tensor-core count; (None, None) if there is none of those either.
    """
    cycles = get("smsp__cycles_elapsed.avg")
    cuda_core_flops: float | None = None
    if cycles is not None:
        weighted_rates = []
        for stem, flops_per_inst in _FLOP_TERMS:
            rate_metric = f"{_FLOP_RATE_PREFIX}{stem}{_FLOP_RATE_SUFFIX}"
            weighted_rates.append((flops_per_inst, get(rate_metric)))
        present = [
            (flops_per_inst, rate)
            for flops_per_inst, rate in weighted_rates
            if rate is not None
        ]
        if present:
            flops_per_cycle = sum(
                flops_per_inst * rate for flops_per_inst, rate in present
            )
            cuda_core_flops = flops_per_cycle * cycles
    tensor_counts = [get(metric) for metric in _TENSOR_FLOP_METRICS]
    tensor_present = [count for count in tensor_counts if count is not None]
    tensor_flops = sum(tensor_present) if tensor_present else None
    if cuda_core_flops is None and tensor_flops is None:
        legacy_flops = _first_present(get, *_LEGACY_FLOP_METRICS)
        return legacy_flops, None
    return (cuda_core_flops or 0.0) + (tensor_flops or 0.0), tensor_flops


def _derive_metrics(get: Getter) -> dict[str, float | None]:
    """Computes the canonical per-kernel metrics of NCU_SCHEMA.

    Args:
        get: A metric-name -> value lookup over a report action, a raw CSV
            row, a details CSV row or a tall CSV group.

    Returns:
        The numeric NCU_SCHEMA columns, by name; a metric the source lacks
        is None.
    """
    duration_ns = _first_present(get, *_DURATION_METRICS)
    flops, flops_tensor = _flops(get)
    read_bytes = get("dram__bytes_read.sum")
    write_bytes = get("dram__bytes_write.sum")
    dram_bytes: float | None
    if read_bytes is not None or write_bytes is not None:
        dram_bytes = (read_bytes or 0.0) + (write_bytes or 0.0)
    else:
        dram_bytes = _first_present(get, *_DRAM_BYTES_METRICS)
    stall_mem, stall_compute = _stall_shares(get)
    if stall_mem is None:  # legacy per-warp-active percentages
        stall_mem = _first_present(get, *_LEGACY_STALL_MEM_METRICS)
        stall_compute = _first_present(get, *_LEGACY_STALL_COMPUTE_METRICS)
    mem_throughput_pct = _first_present(get, *_MEM_THROUGHPUT_METRICS)
    # True DRAM utilisation; sources that only report the overall memory
    # throughput fall back to it.
    dram_throughput_pct = _first_present(get, *_DRAM_THROUGHPUT_METRICS)
    if dram_throughput_pct is None:
        dram_throughput_pct = mem_throughput_pct
    arithmetic_intensity: float | None
    achieved_tflops: float | None
    if flops is not None and dram_bytes:
        arithmetic_intensity = flops / dram_bytes
    else:
        arithmetic_intensity = None
    if flops is not None and duration_ns:
        # FLOP/ns = GFLOP/s; / 1e3 = TFLOP/s.
        achieved_tflops = flops / duration_ns / 1e3
    else:
        achieved_tflops = None
    return {
        "duration_ns": duration_ns,
        "sm_throughput_pct": _first_present(get, *_SM_THROUGHPUT_METRICS),
        "dram_throughput_pct": dram_throughput_pct,
        "mem_throughput_pct": mem_throughput_pct,
        "tensor_pipe_pct": _first_present(get, *_TENSOR_PIPE_METRICS),
        "occupancy_pct": _first_present(get, *_OCCUPANCY_METRICS),
        "flops": flops,
        "flops_tensor": flops_tensor,
        "dram_bytes": dram_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "achieved_tflops": achieved_tflops,
        "stall_mem_pct": stall_mem,
        "stall_compute_pct": stall_compute,
        "grid_size": get("launch__grid_size"),
        "block_size": get("launch__block_size"),
        "registers_per_thread": get("launch__registers_per_thread"),
    }


def _kernel_row(
    kernel_id: int, kernel_name: str, metric_values: dict[str, float]
) -> dict[str, Any]:
    """Builds one NCU_SCHEMA row from a kernel's raw metric values."""
    return {
        "kernel_id": kernel_id,
        "name": kernel_name,
        **_derive_metrics(metric_values.get),
    }


# ncu_report reader (.ncu-rep).

_LEVEL_BY_MESSAGE_TYPE = {
    0: "none",
    1: "ok",
    2: "optimization",
    3: "warning",
    4: "error",
}
_SPEEDUP_TYPE_BY_ID = {0: "unknown", 1: "local", 2: "global"}


def _finding_row(
    kernel_id: int, kernel_name: str, rule: dict[str, Any]
) -> dict[str, Any]:
    """Builds one FINDINGS_SCHEMA row from an ncu_report rule result."""
    message = rule.get("rule_message") or {}
    speedup = rule.get("speedup_estimation") or {}
    return {
        "kernel_id": kernel_id,
        "name": kernel_name,
        "rule": str(rule.get("rule_identifier", "")),
        "section": str(rule.get("section_identifier", "")),
        "level": _LEVEL_BY_MESSAGE_TYPE.get(
            int(message.get("type", 0)), "none"
        ),
        "title": str(message.get("title", "")),
        "message": str(message.get("message", "")),
        "speedup_pct": (
            float(speedup["speedup"]) if "speedup" in speedup else None
        ),
        "speedup_type": (
            _SPEEDUP_TYPE_BY_ID.get(int(speedup["type"]), "unknown")
            if "type" in speedup
            else None
        ),
    }


def _rows_from_report(
    path: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reads kernel metrics and guided-analysis findings with ncu_report.

    Args:
        path: The ``.ncu-rep`` / ``.ncu-repz`` report.

    Returns:
        The NCU_SCHEMA rows and FINDINGS_SCHEMA rows, with kernel ids
        numbered over all ranges of the report.

    Raises:
        ImportError: The optional ``ncu_report`` module is not installed.
    """
    # Optional dependency (extra "ncu"); the caller handles ImportError.
    import ncu_report

    ctx = ncu_report.load_report(str(path))
    kernel_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    kernel_id = 0
    for range_index in range(ctx.num_ranges()):
        profile_range = ctx.range_by_idx(range_index)
        for action_index in range(profile_range.num_actions()):
            action = profile_range.action_by_idx(action_index)
            metric_names = set(action.metric_names())

            def get(
                name: str,
                _action: Any = action,
                _metric_names: set[str] = metric_names,
            ) -> float | None:
                """Returns a metric of the bound action; None if absent/NaN."""
                if name not in _metric_names:
                    return None
                try:
                    value = float(_action.metric_by_name(name).as_double())
                except (TypeError, ValueError, RuntimeError):
                    return None
                return value if value == value else None  # NaN -> None

            kernel_name = str(action.name())
            kernel_rows.append(
                {
                    "kernel_id": kernel_id,
                    "name": kernel_name,
                    **_derive_metrics(get),
                }
            )
            try:
                rules = action.rule_results_as_dicts()
            except Exception:
                # Rule evaluation is best-effort; metrics are the primary
                # data.
                rules = []
            for rule in rules:
                finding_rows.append(_finding_row(kernel_id, kernel_name, rule))
            kernel_id += 1
    return kernel_rows, finding_rows


# CSV reader.

_CONVERTER_TIMEOUT_S = float(os.environ.get("CTA_CONVERTER_TIMEOUT_S", "3600"))


def _read_csv_text(path: pathlib.Path) -> str:
    """Returns the CSV text of a CSV export or of a report.

    A report is converted with ``ncu --import --csv --page raw`` when the
    ``ncu`` CLI is found (on PATH or via ``NCU_PATH``), else its sibling
    ``.csv`` is read.

    Args:
        path: A CSV export or a ``.ncu-rep`` / ``.ncu-repz`` report.

    Returns:
        The CSV text.

    Raises:
        RuntimeError: A report has neither the ``ncu`` CLI nor a sibling CSV.
    """
    if path.suffix.lower() in _REPORT_SUFFIXES:
        ncu_executable = shutil.which("ncu") or os.environ.get("NCU_PATH")
        if not ncu_executable:
            sibling_path = path.with_suffix(".csv")
            if sibling_path.exists():
                return sibling_path.read_text(encoding="utf-8", errors="ignore")
            raise RuntimeError(
                f"Cannot read '{path.name}': install the reader with pip "
                "install 'compute-trace-analysis[ncu]', put the 'ncu' CLI on "
                f"PATH, or provide '{sibling_path.name}'."
            )
        completed = subprocess.run(
            [ncu_executable, "--import", str(path), "--csv", "--page", "raw"],
            check=True,
            capture_output=True,
            text=True,
            timeout=_CONVERTER_TIMEOUT_S,
        )
        return completed.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


def _rows_from_csv(text: str) -> list[dict[str, Any]]:
    """Parses an ncu CSV export into NCU_SCHEMA rows.

    ``==PROF==`` banner lines and blank lines are skipped. Tall exports (one
    ``Metric Name``/``Metric Value`` pair per line) and wide exports (one
    kernel per line, raw or details page) are both understood.

    Args:
        text: The CSV text.

    Returns:
        One row per kernel; empty if the text has no kernel table.
    """
    lines = (
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("==")
    )
    table = list(csv.reader(lines))
    if len(table) < 2:
        return []
    header = [cell.strip() for cell in table[0]]
    if (
        "Metric Name" in header
        and "Metric Value" in header
        and "Kernel Name" in header
    ):
        return _rows_from_tall_csv(header, table[1:])
    return _rows_from_wide_csv(header, table[1:])


def _rows_from_tall_csv(
    header: list[str], body: list[list[str]]
) -> list[dict[str, Any]]:
    """Parses a tall export: one ``Metric Name``/``Metric Value`` per line.

    Lines are grouped by (``ID``, ``Kernel Name``); kernels are numbered in
    order of first appearance. Values are scaled by ``Metric Unit`` when
    present, and the first value of a repeated metric wins.

    Args:
        header: The stripped column names.
        body: The lines after the header.

    Returns:
        One row per kernel.
    """
    kernel_col = header.index("Kernel Name")
    metric_name_col = header.index("Metric Name")
    metric_value_col = header.index("Metric Value")
    id_col = header.index("ID") if "ID" in header else kernel_col
    unit_col = header.index("Metric Unit") if "Metric Unit" in header else None
    last_needed_col = max(kernel_col, metric_name_col, metric_value_col, id_col)
    metric_values_by_kernel: dict[tuple[str, str], dict[str, float]] = {}
    for row in body:
        if len(row) <= last_needed_col:
            continue
        value = _to_float(row[metric_value_col])
        if value is None:
            continue
        if unit_col is not None and len(row) > unit_col:
            value *= _SCALE_BY_UNIT.get(row[unit_col].strip(), 1.0)
        metric_values = metric_values_by_kernel.setdefault(
            (row[id_col], row[kernel_col]), {}
        )
        metric_values.setdefault(row[metric_name_col].strip(), value)
    return [
        _kernel_row(kernel_id, kernel_name, metric_values)
        for kernel_id, ((_, kernel_name), metric_values) in enumerate(
            metric_values_by_kernel.items()
        )
    ]


def _rows_from_wide_csv(
    header: list[str], body: list[list[str]]
) -> list[dict[str, Any]]:
    """Parses a wide export: one kernel per line, one metric per column.

    An optional units line (empty kernel-name cell) right after the header
    scales the values of its columns. Kernel ids come from a numeric ``ID``
    cell, else the line's position among the data lines.

    Args:
        header: The stripped column names.
        body: The lines after the header (units line included).

    Returns:
        One row per kernel; empty if no kernel-name column exists.
    """
    name_col = next(
        (
            header.index(column)
            for column in _KERNEL_NAME_COLUMNS
            if column in header
        ),
        None,
    )
    if name_col is None:
        return []
    id_col = header.index("ID") if "ID" in header else None
    units: list[str] | None = None
    data_rows = body
    if body and not body[0][name_col].strip():
        units = [unit.strip() for unit in body[0]]
        data_rows = body[1:]
    scales = [
        _SCALE_BY_UNIT.get(units[i], 1.0) if units and i < len(units) else 1.0
        for i in range(len(header))
    ]
    kernel_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(data_rows):
        if len(row) <= name_col or not row[name_col].strip():
            continue
        metric_values: dict[str, float] = {}
        for i, column in enumerate(header):
            if i < len(row) and column and i != name_col:
                value = _to_float(row[i])
                if value is not None:
                    metric_values.setdefault(column, value * scales[i])
        if id_col is not None and row[id_col].strip().isdigit():
            kernel_id = int(row[id_col])
        else:
            kernel_id = row_index
        kernel_rows.append(
            _kernel_row(kernel_id, row[name_col].strip(), metric_values)
        )
    return kernel_rows


# Adapter.


@registry.register_adapter("ncu", version="2.1")
class NcuAdapter:
    """Reads Nsight Compute reports and CSV exports into per-kernel metrics.

    Attributes:
        format_name: The registered format name, ``"ncu"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "ncu"
    version = "2.1"

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file is an Nsight Compute report or CSV export.

        Args:
            path: The candidate file.

        Returns:
            True for a ``.ncu-rep`` / ``.ncu-repz`` report, or a ``.csv``
            file whose first 15 lines contain an ncu column name or the
            ``==PROF==`` banner.
        """
        suffix = path.suffix.lower()
        if suffix in _REPORT_SUFFIXES:
            return True
        if suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    head = "".join(fh.readline() for _ in range(15))
                return any(marker in head for marker in _CSV_MARKERS)
            except OSError:
                return False
        return False

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the reports and CSV exports of an artifact to tables.

        Reports are read with ``ncu_report`` when it is installed (which
        also yields guided-analysis findings), else through the ``ncu`` CLI
        or a sibling CSV.

        Args:
            artifact: The artifact whose files are read.
            out_dir: The directory the Parquet tables are written to.

        Returns:
            The paths of the ``ncu_kernels`` and ``ncu_findings`` Parquet
            tables, by table name.

        Raises:
            RuntimeError: A report cannot be read by any route.
        """
        kernel_rows: list[dict[str, Any]] = []
        finding_rows: list[dict[str, Any]] = []
        for path in sorted(artifact.paths):
            if path.suffix.lower() in _REPORT_SUFFIXES:
                try:
                    report_kernels, report_findings = _rows_from_report(path)
                    kernel_rows += report_kernels
                    finding_rows += report_findings
                    continue
                except ImportError:
                    # No ncu_report: fall through to the CLI / sibling-CSV
                    # route.
                    pass
            kernel_rows += _rows_from_csv(_read_csv_text(path))

        if kernel_rows:
            numeric_schema = {
                column: dtype
                for column, dtype in NCU_SCHEMA.items()
                if column not in _NCU_TEXT_COLUMNS
            }
            kernels = alignment.enrich_ops_df(
                pl.DataFrame(
                    kernel_rows,
                    schema_overrides=numeric_schema,
                    infer_schema_length=None,
                ),
                "name",
            )
            kernels = kernels.select(list(NCU_SCHEMA.keys())).cast(NCU_SCHEMA)
        else:
            kernels = pl.DataFrame(schema=NCU_SCHEMA)
        if finding_rows:
            findings = pl.DataFrame(finding_rows, schema=FINDINGS_SCHEMA)
        else:
            findings = pl.DataFrame(schema=FINDINGS_SCHEMA)

        table_paths: dict[str, pathlib.Path] = {}
        for table_name, table in (
            ("ncu_kernels", kernels),
            ("ncu_findings", findings),
        ):
            table_path = out_dir / f"{table_name}.parquet"
            table.write_parquet(table_path, compression="zstd")
            table_paths[table_name] = table_path
        return table_paths
