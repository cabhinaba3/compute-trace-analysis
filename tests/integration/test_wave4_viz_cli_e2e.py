from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import cta
from cta.cli.app import app
from cta.core.manifest import load_manifest
from cta.synth.generators import (
    write_chakra_et_run,
    write_kineto_trace,
    write_ncu_csv,
    write_trtexec_timing,
)

runner = CliRunner()


def test_all_six_plot_kinds_and_formats(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    # 1. Distributed report (dag_critical_path, comm_collectives, rank_skew, traffic_matrix)
    c0, c1, c2 = tmp_path / "c0", tmp_path / "c1", tmp_path / "c2"
    write_chakra_et_run(c0, world_size=4, compute_us=800, allreduce_bytes=400_000)
    write_chakra_et_run(
        c1, world_size=4, compute_us=800, allreduce_bytes=800_000, straggler_rank=2, straggler_factor=1.8
    )
    write_chakra_et_run(c2, world_size=4, compute_us=500, allreduce_bytes=200_000)

    rep_dist = cta.compare(c0, c1, labels=["Ring_Base", "Ring_Skew"], cache_dir=cache_dir)
    for kind, ext, style in (
        ("breakdown", "pdf", "paper_single_col"),
        ("tornado", "svg", "paper_double_col"),
        ("traffic_delta", "png", "presentation"),
        ("skew", "pdf", "paper_single_col"),
        ("scaling", "pdf", "paper_single_col"),
    ):
        out = tmp_path / "figs" / f"dist_{kind}.{ext}"
        rep_dist.plot(kind=kind, path=out, style=style)
        assert out.exists() and out.stat().st_size > 500

    # 3-way (multi) breakdown, traffic heatmap, and scaling
    rep_multi = cta.compare(c0, c1, c2, labels=["Base", "Skew", "Fast"], cache_dir=cache_dir)
    for kind in ("breakdown", "traffic_delta", "scaling"):
        out_m = tmp_path / "figs" / f"multi_{kind}.pdf"
        rep_multi.plot(kind=kind, path=out_m)
        assert out_m.exists() and out_m.stat().st_size > 500

    # 2. Device & Roofline plots (timeline_overlap, kernel_diff, roofline_diff)
    k0 = write_kineto_trace(tmp_path / "k0.json.gz", attn_us=420.0, gemm_us=580.0, kernel_variant="fp16")
    k1 = write_kineto_trace(tmp_path / "k1.json.gz", attn_us=190.0, gemm_us=260.0, kernel_variant="fp8")
    rep_kin = cta.compare(k0, k1, cache_dir=cache_dir)
    for kind in ("breakdown", "tornado"):
        out_k = tmp_path / "figs" / f"kin_{kind}.pdf"
        rep_kin.plot(kind=kind, path=out_k)
        assert out_k.exists() and out_k.stat().st_size > 500

    n0 = write_ncu_csv(tmp_path / "n0.csv", sm_pct=34.0, dram_pct=85.0)
    n1 = write_ncu_csv(tmp_path / "n1.csv", gemm_dur_ns=250_000.0, sm_pct=82.0, dram_pct=36.0, flops=4e10, dram_bytes=5e8)
    rep_ncu = cta.compare(n0, n1, cache_dir=cache_dir)
    out_roof = tmp_path / "figs" / "roofline.pdf"
    rep_ncu.plot(kind="roofline", path=out_roof)
    assert out_roof.exists() and out_roof.stat().st_size > 500

    with pytest.raises(ValueError, match="Unknown plot kind"):
        rep_ncu.plot(kind="nonexistent_kind")


def test_multitrial_toml_manifest_stats_and_pandas_latex(tmp_path: Path) -> None:
    # 4 trials per group to exercise Welch t-test, Mann-Whitney U, Cohen's d, and 95% CI
    for i, ms in enumerate((3.10, 3.20, 3.15, 3.25)):
        write_trtexec_timing(tmp_path / f"base_{i}.json", queries=25, compute_ms=ms)
    for i, ms in enumerate((1.40, 1.45, 1.38, 1.42)):
        write_trtexec_timing(tmp_path / f"cand_{i}.json", queries=25, compute_ms=ms)

    manifest_toml = tmp_path / "sweep.toml"
    manifest_toml.write_text(
        """
[config]
warmup_steps = 2
alpha = 0.05
cache_dir = ".cache_toml"

[config.gates]
mean_step_ms = 5.0

[[groups]]
id = "fp16"
label = "FP16_4Trials"
trials = ["base_0.json", "base_1.json", "base_2.json", "base_3.json"]

[[groups]]
id = "int8"
label = "INT8_4Trials"
trials = ["cand_0.json", "cand_1.json", "cand_2.json", "cand_3.json"]
"""
    )

    groups, cfg = load_manifest(manifest_toml)
    report = cta.run_comparison(groups, cfg)
    df = report.to_polars("latency_throughput")
    row = df.filter(df["metric"] == "mean_step_ms").row(0, named=True)
    assert row["significant"] is True
    assert float(row["p_welch"]) < 0.01
    assert abs(float(row["cohens_d"])) > 2.0
    assert float(row["delta"]) < -1.5
    assert not report.violations

    pdf_df = report.to_pandas("latency_throughput")
    assert len(pdf_df) == df.height

    tex = report.to_latex(tmp_path / "out.tex")
    assert "\\toprule" in tex and "\\bottomrule" in tex


def test_cli_end_to_end_workflow(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demo"
    cache_dir = tmp_path / "cli_cache"
    export_dir = tmp_path / "exports"
    plot_dir = tmp_path / "plots"

    # 1. cta synth
    r_synth = runner.invoke(app, ["synth", str(demo_dir)])
    assert r_synth.exit_code == 0, r_synth.output
    assert (demo_dir / "sweep.yaml").exists()

    # 2. cta inspect
    r_insp = runner.invoke(app, ["inspect", str(demo_dir / "gpu_bundle_fp16"), "--cache-dir", str(cache_dir)])
    assert r_insp.exit_code == 0, r_insp.output
    assert "nsys" in r_insp.output and "trtexec_timing" in r_insp.output

    # 3. cta compare --manifest + bundle export
    r_cmp = runner.invoke(
        app,
        [
            "compare",
            "--manifest",
            str(demo_dir / "sweep.yaml"),
            "--out-dir",
            str(export_dir),
            "--export",
            "markdown,latex,csv,json,pdf,svg",
        ],
    )
    assert r_cmp.exit_code == 0, r_cmp.output
    for fname in ("report.md", "report.tex", "report.json", "breakdown.pdf", "breakdown.svg"):
        assert (export_dir / fname).exists()

    # 4. cta compare gate violation returns exit code 2 (reverse INT8 -> FP16 regresses latency)
    r_gate = runner.invoke(
        app,
        [
            "compare",
            str(demo_dir / "gpu_bundle_int8"),
            str(demo_dir / "gpu_bundle_fp16"),
            "--gate",
            "mean_step_ms:+5%",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert r_gate.exit_code == 2

    # 5. cta equiv pass & fail
    r_eq_ok = runner.invoke(
        app,
        ["equiv", str(demo_dir / "chakra_ring"), str(demo_dir / "goal" / "ring.bin"), "--cache-dir", str(cache_dir)],
    )
    assert r_eq_ok.exit_code == 0, r_eq_ok.output

    r_eq_fail = runner.invoke(
        app,
        ["equiv", str(demo_dir / "chakra_skewed"), str(demo_dir / "goal" / "ring.bin"), "--cache-dir", str(cache_dir)],
    )
    assert r_eq_fail.exit_code == 2

    # 6. cta plot
    r_plot = runner.invoke(
        app,
        [
            "plot",
            str(demo_dir / "ncu" / "baseline.csv"),
            str(demo_dir / "ncu" / "optimized.csv"),
            "--kind",
            "roofline,tornado,scaling",
            "--format",
            "svg",
            "--out-dir",
            str(plot_dir),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert r_plot.exit_code == 0, r_plot.output
    for k in ("roofline", "tornado", "scaling"):
        assert (plot_dir / f"{k}.svg").exists()

    # 7. cta cache status & clear
    r_st = runner.invoke(app, ["cache", "status", "--cache-dir", str(cache_dir)])
    assert r_st.exit_code == 0 and "Cache entries:" in r_st.output
    r_cl = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir)])
    assert r_cl.exit_code == 0 and "Cleared" in r_cl.output


def test_quickstarts_tall_ncu_legacy_nsys_and_bundle_equiv(tmp_path: Path) -> None:
    import sqlite3

    from cta.synth.generators import write_goal_allreduce, write_nsys_sqlite
    from examples import quickstart_chakra_goal, quickstart_gpu_bundle

    # 1. Run both quickstart scripts end-to-end
    quickstart_chakra_goal.main(tmp_path / "qs_cg")
    quickstart_gpu_bundle.main(tmp_path / "qs_gpu")
    assert (tmp_path / "qs_cg" / "figures" / "breakdown.pdf").exists()
    assert (tmp_path / "qs_gpu" / "artifacts" / "tornado.pdf").exists()

    # 2. Tall NCU details CSV format ("ID,Kernel Name,Metric Name,Metric Value")
    tall_a = tmp_path / "tall_a.csv"
    tall_b = tmp_path / "tall_b.csv"
    for p, dur, sm in ((tall_a, "400000", "40.0"), (tall_b, "200000", "82.0")):
        p.write_text(
            "ID,Kernel Name,Metric Name,Metric Value\n"
            f"0,ampere_gemm_f16,gpu__time_duration.sum,{dur}\n"
            f"0,ampere_gemm_f16,sm__throughput.avg.pct_of_peak_sustained_elapsed,{sm}\n"
            "0,ampere_gemm_f16,dram__bytes.sum,500000000\n"
            "0,ampere_gemm_f16,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,20000000000\n"
        )
    rep_tall = cta.compare(tall_a, tall_b, cache_dir=tmp_path / "cache")
    assert "roofline_diff" in rep_tall.aspects

    # 3. Legacy Nsight Systems SQLite with GENERIC_EVENTS table (instead of GPU_METRICS)
    leg_a = write_nsys_sqlite(tmp_path / "leg_a.sqlite", gemm_us=500)
    leg_b = write_nsys_sqlite(tmp_path / "leg_b.sqlite", gemm_us=300)
    for db_p, sm_val in ((leg_a, 45.0), (leg_b, 78.0)):
        with sqlite3.connect(db_p) as conn:
            conn.execute("DROP TABLE GPU_METRICS")
            conn.execute("CREATE TABLE GENERIC_EVENTS (data TEXT)")
            conn.execute("INSERT INTO GENERIC_EVENTS VALUES (?)", (str({"SM Active [Throughput %]": sm_val}),))
    rep_leg = cta.compare(leg_a, leg_b, cache_dir=tmp_path / "cache")
    assert "kernel_diff" in rep_leg.aspects

    # 4. Dual-artifact RunBundle (Chakra-ET + GOAL) exercising schedule_equiv aspect
    b0, b1 = tmp_path / "bundle_ring", tmp_path / "bundle_bad"
    write_chakra_et_run(b0, world_size=4, compute_us=600, allreduce_bytes=400_000)
    write_goal_allreduce(b0 / "sched.goal", world_size=4, compute_ns=600_000, allreduce_bytes=400_000, algo="ring")
    write_chakra_et_run(b1, world_size=4, compute_us=600, allreduce_bytes=400_000)
    write_goal_allreduce(b1 / "sched.goal", world_size=4, compute_ns=900_000, allreduce_bytes=800_000, algo="ring")
    rep_eq = cta.compare(b0, b1, cache_dir=tmp_path / "cache")
    assert "schedule_equiv" in rep_eq.aspects


def test_single_trace_analyze_and_single_trace_plots(tmp_path: Path) -> None:
    cache_dir = tmp_path / "single_cache"
    demo_dir = tmp_path / "demo_single"
    assert runner.invoke(app, ["synth", str(demo_dir)]).exit_code == 0

    # 1. SDK single-trace analysis on distributed Chakra-ET run + plots
    rep_c = cta.analyze(demo_dir / "chakra_skewed", label="Skewed_4R", cache_dir=cache_dir)
    assert rep_c.mode == "single"
    assert len(rep_c.diagnoses) == 1
    assert "Bottleneck Regime:" in rep_c.diagnoses[0].summary
    by_comm = rep_c.to_polars("comm_collectives", table="by_comm_type")
    assert "share_pct" in by_comm.columns
    for kind in ("breakdown", "topk", "traffic", "skew", "scaling"):
        fig_p = tmp_path / "sfigs" / f"c_{kind}.pdf"
        rep_c.plot(kind=kind, path=fig_p)
        assert fig_p.exists() and fig_p.stat().st_size > 500

    # 2. SDK single-trace analysis on NCU roofline CSV
    rep_ncu = cta.analyze(demo_dir / "ncu" / "baseline.csv", label="NCU_Single", cache_dir=cache_dir)
    roof_p = tmp_path / "sfigs" / "ncu_roofline.pdf"
    rep_ncu.plot(kind="roofline", path=roof_p)
    assert roof_p.exists() and roof_p.stat().st_size > 500

    # 3. CLI `cta analyze` + CLI single-trace `cta plot <single_trace>`
    out_single = tmp_path / "cli_single_out"
    r_an = runner.invoke(
        app,
        [
            "analyze",
            str(demo_dir / "gpu_bundle_fp16"),
            "--warmup-steps",
            "2",
            "--out-dir",
            str(out_single),
            "--export",
            "markdown,latex,csv,json,pdf",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert r_an.exit_code == 0, r_an.output
    assert (out_single / "report.md").exists() and (out_single / "breakdown.pdf").exists()

    splot_dir = tmp_path / "cli_single_plots"
    r_sp = runner.invoke(
        app,
        [
            "plot",
            str(demo_dir / "gpu_bundle_fp16"),
            "--kind",
            "breakdown,topk",
            "--format",
            "svg",
            "--out-dir",
            str(splot_dir),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert r_sp.exit_code == 0, r_sp.output
    assert (splot_dir / "breakdown.svg").exists() and (splot_dir / "topk.svg").exists()


