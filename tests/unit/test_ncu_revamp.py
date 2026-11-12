from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import cta
from cta.adapters.ncu.reader import NcuAdapter
from cta.core.models import TraceArtifact

REAL = Path(__file__).resolve().parents[1] / "data_real" / "a"
needs_real = pytest.mark.skipif(not (REAL / "ncu_fp32.ncu-rep").exists(), reason="real GPULab fixtures not present")


def _ingest(path: Path) -> pl.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        out = NcuAdapter().ingest(TraceArtifact(format="ncu", paths=(path,), world_size=1), Path(d))
        return pl.read_parquet(out["ncu_kernels"])


def _csv(tmp: Path, name: str, header: str, units: str, *rows: str) -> Path:
    p = tmp / name
    p.write_text("\n".join(["==PROF== Connected", header, units, *rows]))
    return p


def test_detects_ncu_repz() -> None:
    assert NcuAdapter().detect(Path("x.ncu-repz")) and NcuAdapter().detect(Path("x.ncu-rep"))


def test_csv_units_are_applied_to_time_and_bytes(tmp_path: Path) -> None:
    p = _csv(
        tmp_path, "u.csv",
        '"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum"',
        '"","","usecond","Mbyte","Kbyte"',
        '"0","gemm","6.5","1.5","2"',
    )  # fmt: skip
    r = _ingest(p).row(0, named=True)
    assert r["duration_ns"] == pytest.approx(6500.0)
    assert r["dram_bytes"] == pytest.approx(1.5e6 + 2e3)  # read + write; there is no dram__bytes.sum


def test_missing_metrics_are_null_not_zero(tmp_path: Path) -> None:
    p = _csv(tmp_path, "m.csv", '"ID","Kernel Name","gpu__time_duration.sum"', '"","","nsecond"', '"0","k","100"')
    r = _ingest(p).row(0, named=True)
    assert r["duration_ns"] == 100.0
    assert r["flops"] is None and r["dram_bytes"] is None and r["achieved_tflops"] is None
    assert r["dram_throughput_pct"] is None and r["stall_mem_pct"] is None


def test_flops_use_achieved_counts_not_peak_derived_metrics(tmp_path: Path) -> None:
    # derived__*_x2 are per-device peak constants (same for every kernel); they must not feed the FLOP count.
    p = _csv(
        tmp_path, "f.csv",
        '"ID","Kernel Name","gpu__time_duration.sum","smsp__cycles_elapsed.avg",'
        '"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",'
        '"smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",'
        '"derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2","sm__ops_path_tensor_src_fp16.sum"',
        '"","","nsecond","cycle","inst/cycle","inst/cycle","inst",""',
        '"0","a","1000","100","10","4","33792","1000"',
        '"1","b","1000","100","0","0","33792","0"',
    )  # fmt: skip
    df = _ingest(p).sort("kernel_id")
    assert df["flops"].to_list() == [pytest.approx((10 * 2 + 4) * 100 + 1000), 0.0]
    assert df["flops_tensor"].to_list() == [1000.0, 0.0]


def test_legacy_aliases_still_work(tmp_path: Path) -> None:
    from cta.synth.generators import write_ncu_csv

    r = _ingest(write_ncu_csv(tmp_path / "legacy.csv", flops=2e10, dram_bytes=1e9, dram_pct=82.0)).row(0, named=True)
    assert r["flops"] == 2e10 and r["dram_bytes"] == 1e9
    assert r["dram_throughput_pct"] == 82.0  # falls back to overall memory throughput when DRAM-specific is absent


def test_roofline_points_aggregate_sums_not_mean_of_ratios(tmp_path: Path) -> None:
    hdr = '"ID","Kernel Name","gpu__time_duration.sum","smsp__cycles_elapsed.avg","smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed","dram__bytes_read.sum"'
    mk = lambda n: _csv(  # noqa: E731
        tmp_path, n, hdr, '"","","nsecond","cycle","inst/cycle","byte"',
        '"0","same_kernel","1000","1000","1000","1000000"',  # 1e6 FLOPs / 1e6 B  -> AI 1
        '"1","same_kernel","1000","1000","1","1"',  # 1e3 FLOPs / 1 B     -> AI 1000
    )  # fmt: skip
    rep = cta.analyze(mk("w.csv"), aspects=["roofline_diff"], cache_dir=tmp_path / "c")
    pt = next(df for k, df in rep.aspect("roofline_diff").details.items() if k.startswith("roofline_")).row(0, named=True)
    assert pt["arith_intensity"] == pytest.approx((1e6 + 1e3) / (1e6 + 1))  # ~1.001; a mean of ratios would be ~500


@needs_real
def test_real_report_and_csv_paths_agree() -> None:
    pytest.importorskip("ncu_report")
    rep = _ingest(REAL / "ncu_fp32.ncu-rep")
    raw = _ingest(REAL / "ncu_fp32_raw.csv")
    assert rep.height == raw.height == 24
    for col in ("duration_ns", "flops", "flops_tensor", "dram_bytes", "dram_throughput_pct", "sm_throughput_pct", "grid_size"):
        np.testing.assert_allclose(
            rep[col].fill_null(-1).to_numpy(), raw[col].fill_null(-1).to_numpy(), rtol=2e-3, atol=1e-6, err_msg=col
        )
    for col in ("stall_mem_pct", "stall_compute_pct"):  # CSV prints rounded values
        np.testing.assert_allclose(rep[col].fill_null(-1).to_numpy(), raw[col].fill_null(-1).to_numpy(), atol=0.2, err_msg=col)


@needs_real
def test_real_flop_counts_match_known_gemm_work_and_are_not_constants() -> None:
    pytest.importorskip("ncu_report")
    df = _ingest(REAL / "ncu_fp32.ncu-rep")
    gemm = df.filter(pl.col("name").str.contains("gemm") & (pl.col("flops") > 1e8))
    assert gemm.height >= 1
    for f in gemm["flops"]:
        assert f == pytest.approx(2 * 1024 * 512 * 512, rel=2e-3)  # FP32 GEMM: exactly 2*M*N*K
    assert df["flops"].n_unique() > 5  # peak-capacity derived metrics would give one value for every kernel
    fp16 = _ingest(REAL / "ncu_fp16.ncu-rep")
    assert fp16.filter(pl.col("flops_tensor") > 0).height > 0  # HGMMA/tensor-core work is counted
    assert (fp16["dram_throughput_pct"] != fp16["mem_throughput_pct"]).any()  # DRAM is not aliased to overall memory


@needs_real
def test_real_report_exposes_ncu_guided_findings() -> None:
    pytest.importorskip("ncu_report")
    with tempfile.TemporaryDirectory() as d:
        out = NcuAdapter().ingest(TraceArtifact(format="ncu", paths=(REAL / "ncu_fp16.ncu-rep",), world_size=1), Path(d))
        f = pl.read_parquet(out["ncu_findings"])
    assert f.height > 24 and {"SOLBottleneck", "LaunchConfiguration"} <= set(f["rule"].to_list())


@pytest.mark.skipif(
    not (REAL.parent / "c" / "nsp_ncu-output-gemm-0.ncu-rep").exists(), reason="real GPULab fixtures not present"
)
def test_real_nsight_python_report_with_metric_subset() -> None:
    """A report collected through NVIDIA's nsight-python (only the requested metrics): exact FLOPs, nulls elsewhere."""
    pytest.importorskip("ncu_report")
    df = _ingest(REAL.parent / "c" / "nsp_ncu-output-gemm-0.ncu-rep").sort("kernel_id")
    assert df["flops"][0] == pytest.approx(2 * 1024 * 512 * 512, rel=1e-9)  # FP32 GEMM, FFMA counters only
    assert df["flops_tensor"][2] == df["flops"][2] > 5e8  # FP16 GEMM runs entirely on the tensor cores
    assert df["sm_throughput_pct"].is_null().all() and df["stall_mem_pct"].is_null().all()  # not collected -> null
