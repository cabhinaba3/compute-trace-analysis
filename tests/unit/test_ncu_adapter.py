"""Tests for the Nsight Compute (ncu) adapter.

Covers unit handling and null metrics in CSV exports, FLOP counting from
achieved counters, roofline aggregation, and agreement between the `.ncu-rep`
and CSV ingest paths on real GPULab captures.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import polars as pl
import pytest

import cta
from cta.adapters.ncu import reader
from cta.core import models

REAL_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data_real"
NCU_RUN_DIR = REAL_DATA_DIR / "a"
NSIGHT_PYTHON_RUN_DIR = REAL_DATA_DIR / "c"
needs_real_fixtures = pytest.mark.skipif(
    not (NCU_RUN_DIR / "ncu_fp32.ncu-rep").exists(),
    reason="real GPULab fixtures not present",
)

FFMA_PER_CYCLE_METRIC = (
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"
)
FADD_PER_CYCLE_METRIC = (
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
)


def _ingest(report_path: pathlib.Path) -> pl.DataFrame:
    """Ingests one Nsight Compute file and returns its kernels table.

    Args:
        report_path: A `.ncu-rep` report or a CSV export.

    Returns:
        The `ncu_kernels` table, read back from the adapter's Parquet output.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact = models.TraceArtifact(
            format="ncu", paths=(report_path,), world_size=1
        )
        table_paths = reader.NcuAdapter().ingest(
            artifact, pathlib.Path(tmp_dir)
        )
        return pl.read_parquet(table_paths["ncu_kernels"])


def _csv_line(cells: list[str]) -> str:
    """Joins cells into one CSV line, quoting every cell as ncu does."""
    return ",".join(f'"{cell}"' for cell in cells)


def _write_ncu_csv(
    csv_path: pathlib.Path,
    columns: list[str],
    units: list[str],
    *kernel_rows: list[str],
) -> pathlib.Path:
    """Writes a raw Nsight Compute CSV export with a units row.

    Args:
        csv_path: Where to write the file.
        columns: The column names (ncu metric names after ID and kernel name).
        units: The unit of each column, empty where there is none.
        *kernel_rows: The cell values of one kernel per row.

    Returns:
        `csv_path`.
    """
    lines = [
        "==PROF== Connected",
        _csv_line(columns),
        _csv_line(units),
        *(_csv_line(row) for row in kernel_rows),
    ]
    csv_path.write_text("\n".join(lines))
    return csv_path


def test_detects_ncu_repz() -> None:
    adapter = reader.NcuAdapter()
    assert adapter.detect(pathlib.Path("x.ncu-repz"))
    assert adapter.detect(pathlib.Path("x.ncu-rep"))


def test_csv_units_are_applied_to_time_and_bytes(
    tmp_path: pathlib.Path,
) -> None:
    csv_path = _write_ncu_csv(
        tmp_path / "u.csv",
        [
            "ID",
            "Kernel Name",
            "gpu__time_duration.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
        ],
        ["", "", "usecond", "Mbyte", "Kbyte"],
        ["0", "gemm", "6.5", "1.5", "2"],
    )
    kernel = _ingest(csv_path).row(0, named=True)
    assert kernel["duration_ns"] == pytest.approx(6500.0)
    # Read plus write bytes; the export has no dram__bytes.sum column.
    assert kernel["dram_bytes"] == pytest.approx(1.5e6 + 2e3)


def test_missing_metrics_are_null_not_zero(tmp_path: pathlib.Path) -> None:
    csv_path = _write_ncu_csv(
        tmp_path / "m.csv",
        ["ID", "Kernel Name", "gpu__time_duration.sum"],
        ["", "", "nsecond"],
        ["0", "k", "100"],
    )
    kernel = _ingest(csv_path).row(0, named=True)
    assert kernel["duration_ns"] == 100.0
    assert kernel["flops"] is None
    assert kernel["dram_bytes"] is None
    assert kernel["achieved_tflops"] is None
    assert kernel["dram_throughput_pct"] is None
    assert kernel["stall_mem_pct"] is None


def test_flops_use_achieved_counts_not_peak_derived_metrics(
    tmp_path: pathlib.Path,
) -> None:
    # The derived__*_x2 metrics are per-device peak constants (the same for
    # every kernel); they must not feed the FLOP count.
    csv_path = _write_ncu_csv(
        tmp_path / "f.csv",
        [
            "ID",
            "Kernel Name",
            "gpu__time_duration.sum",
            "smsp__cycles_elapsed.avg",
            FFMA_PER_CYCLE_METRIC,
            FADD_PER_CYCLE_METRIC,
            "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
            "sm__ops_path_tensor_src_fp16.sum",
        ],
        ["", "", "nsecond", "cycle", "inst/cycle", "inst/cycle", "inst", ""],
        ["0", "a", "1000", "100", "10", "4", "33792", "1000"],
        ["1", "b", "1000", "100", "0", "0", "33792", "0"],
    )
    kernels = _ingest(csv_path).sort("kernel_id")
    assert kernels["flops"].to_list() == [
        pytest.approx((10 * 2 + 4) * 100 + 1000),
        0.0,
    ]
    assert kernels["flops_tensor"].to_list() == [1000.0, 0.0]


def test_legacy_aliases_still_work(tmp_path: pathlib.Path) -> None:
    from cta.synth import ncu

    csv_path = ncu.write_ncu_csv(
        tmp_path / "legacy.csv", flops=2e10, dram_bytes=1e9, dram_pct=82.0
    )
    kernel = _ingest(csv_path).row(0, named=True)
    assert kernel["flops"] == 2e10
    assert kernel["dram_bytes"] == 1e9
    # Falls back to the overall memory throughput when the DRAM-specific
    # metric is absent.
    assert kernel["dram_throughput_pct"] == 82.0


def test_roofline_points_aggregate_sums_not_mean_of_ratios(
    tmp_path: pathlib.Path,
) -> None:
    columns = [
        "ID",
        "Kernel Name",
        "gpu__time_duration.sum",
        "smsp__cycles_elapsed.avg",
        FADD_PER_CYCLE_METRIC,
        "dram__bytes_read.sum",
    ]
    units = ["", "", "nsecond", "cycle", "inst/cycle", "byte"]
    # Two launches of the same kernel: 1e6 FLOPs over 1e6 B (intensity 1)
    # and 1e3 FLOPs over 1 B (intensity 1000).
    csv_path = _write_ncu_csv(
        tmp_path / "w.csv",
        columns,
        units,
        ["0", "same_kernel", "1000", "1000", "1000", "1000000"],
        ["1", "same_kernel", "1000", "1000", "1", "1"],
    )
    report = cta.analyze(
        csv_path, aspects=["roofline_diff"], cache_dir=tmp_path / "c"
    )
    roofline_tables = report.aspect("roofline_diff").details
    point = next(
        table
        for name, table in roofline_tables.items()
        if name.startswith("roofline_")
    ).row(0, named=True)
    # Summing gives about 1.001; a mean of ratios would give about 500.
    assert point["arithmetic_intensity"] == pytest.approx(
        (1e6 + 1e3) / (1e6 + 1)
    )


@needs_real_fixtures
def test_real_report_and_csv_paths_agree() -> None:
    pytest.importorskip("ncu_report")
    from_report = _ingest(NCU_RUN_DIR / "ncu_fp32.ncu-rep")
    from_csv = _ingest(NCU_RUN_DIR / "ncu_fp32_raw.csv")
    assert from_report.height == from_csv.height == 24
    for column in (
        "duration_ns",
        "flops",
        "flops_tensor",
        "dram_bytes",
        "dram_throughput_pct",
        "sm_throughput_pct",
        "grid_size",
    ):
        np.testing.assert_allclose(
            from_report[column].fill_null(-1).to_numpy(),
            from_csv[column].fill_null(-1).to_numpy(),
            rtol=2e-3,
            atol=1e-6,
            err_msg=column,
        )
    # The CSV export prints rounded stall percentages.
    for column in ("stall_mem_pct", "stall_compute_pct"):
        np.testing.assert_allclose(
            from_report[column].fill_null(-1).to_numpy(),
            from_csv[column].fill_null(-1).to_numpy(),
            atol=0.2,
            err_msg=column,
        )


@needs_real_fixtures
def test_real_flop_counts_match_known_gemm_work_and_are_not_constants() -> None:
    pytest.importorskip("ncu_report")
    kernels = _ingest(NCU_RUN_DIR / "ncu_fp32.ncu-rep")
    gemm_kernels = kernels.filter(
        pl.col("name").str.contains("gemm") & (pl.col("flops") > 1e8)
    )
    assert gemm_kernels.height >= 1
    # An FP32 GEMM does exactly 2*M*N*K FLOPs.
    for flops in gemm_kernels["flops"]:
        assert flops == pytest.approx(2 * 1024 * 512 * 512, rel=2e-3)
    # Peak-capacity derived metrics would give one value for every kernel.
    assert kernels["flops"].n_unique() > 5

    fp16_kernels = _ingest(NCU_RUN_DIR / "ncu_fp16.ncu-rep")
    # HGMMA (tensor-core) work is counted.
    assert fp16_kernels.filter(pl.col("flops_tensor") > 0).height > 0
    # DRAM throughput is not aliased to the overall memory throughput.
    dram_pct = fp16_kernels["dram_throughput_pct"]
    memory_pct = fp16_kernels["mem_throughput_pct"]
    assert (dram_pct != memory_pct).any()


@needs_real_fixtures
def test_real_report_exposes_ncu_guided_findings() -> None:
    pytest.importorskip("ncu_report")
    artifact = models.TraceArtifact(
        format="ncu", paths=(NCU_RUN_DIR / "ncu_fp16.ncu-rep",), world_size=1
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        table_paths = reader.NcuAdapter().ingest(
            artifact, pathlib.Path(tmp_dir)
        )
        findings = pl.read_parquet(table_paths["ncu_findings"])
    assert findings.height > 24
    rules = set(findings["rule"].to_list())
    assert {"SOLBottleneck", "LaunchConfiguration"} <= rules


@pytest.mark.skipif(
    not (NSIGHT_PYTHON_RUN_DIR / "nsp_ncu-output-gemm-0.ncu-rep").exists(),
    reason="real GPULab fixtures not present",
)
def test_real_nsight_python_report_with_metric_subset() -> None:
    """Reads a report collected through NVIDIA's nsight-python.

    The report holds only the requested metrics: FLOPs are exact and every
    metric that was not collected is null.
    """
    pytest.importorskip("ncu_report")
    report_path = NSIGHT_PYTHON_RUN_DIR / "nsp_ncu-output-gemm-0.ncu-rep"
    kernels = _ingest(report_path).sort("kernel_id")
    # The first kernel is an FP32 GEMM, measured with FFMA counters only.
    assert kernels["flops"][0] == pytest.approx(2 * 1024 * 512 * 512, rel=1e-9)
    # The third kernel is an FP16 GEMM that runs entirely on the tensor cores.
    assert kernels["flops_tensor"][2] == kernels["flops"][2] > 5e8
    # Metrics that were not collected are null.
    assert kernels["sm_throughput_pct"].is_null().all()
    assert kernels["stall_mem_pct"].is_null().all()
