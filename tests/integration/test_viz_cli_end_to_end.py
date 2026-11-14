"""End-to-end tests for plotting, exports and the CLI workflow.

Covers every plot kind and output format, multi-trial TOML manifests with
statistics and LaTeX output, the full `synth`/`inspect`/`compare`/
`verify-equivalence`/`plot`/`cache` CLI workflow, the quickstart scripts,
less common input formats, and single-trace analysis and plots.
"""

from __future__ import annotations

import pathlib

import pytest
from typer import testing

import cta
from cta.cli import app as cli_app
from cta.core import manifest
from cta.synth import chakra
from cta.synth import kineto
from cta.synth import ncu
from cta.synth import tensorrt

cli_runner = testing.CliRunner()


def test_all_six_plot_kinds_and_formats(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    figure_dir = tmp_path / "figs"

    # 1. Distributed report (dag_critical_path, comm_collectives, rank_skew,
    # traffic_matrix).
    base_run_dir = tmp_path / "c0"
    skew_run_dir = tmp_path / "c1"
    fast_run_dir = tmp_path / "c2"
    chakra.write_chakra_et_run(
        base_run_dir, world_size=4, compute_us=800, allreduce_bytes=400_000
    )
    chakra.write_chakra_et_run(
        skew_run_dir,
        world_size=4,
        compute_us=800,
        allreduce_bytes=800_000,
        straggler_rank=2,
        straggler_factor=1.8,
    )
    chakra.write_chakra_et_run(
        fast_run_dir, world_size=4, compute_us=500, allreduce_bytes=200_000
    )

    distributed_report = cta.compare(
        base_run_dir,
        skew_run_dir,
        labels=["Ring_Base", "Ring_Skew"],
        cache_dir=cache_dir,
    )
    for kind, extension, style in (
        ("breakdown", "pdf", "paper_single_col"),
        ("tornado", "svg", "paper_double_col"),
        ("traffic_delta", "png", "presentation"),
        ("skew", "pdf", "paper_single_col"),
        ("scaling", "pdf", "paper_single_col"),
    ):
        figure_path = figure_dir / f"dist_{kind}.{extension}"
        distributed_report.plot(kind=kind, path=figure_path, style=style)
        assert figure_path.exists() and figure_path.stat().st_size > 500

    # 3-way (multi) breakdown, traffic heatmap, and scaling.
    three_way_report = cta.compare(
        base_run_dir,
        skew_run_dir,
        fast_run_dir,
        labels=["Base", "Skew", "Fast"],
        cache_dir=cache_dir,
    )
    for kind in ("breakdown", "traffic_delta", "scaling"):
        figure_path = figure_dir / f"multi_{kind}.pdf"
        three_way_report.plot(kind=kind, path=figure_path)
        assert figure_path.exists() and figure_path.stat().st_size > 500

    # 2. Device and roofline plots (timeline_overlap, kernel_diff,
    # roofline_diff).
    fp16_trace_path = kineto.write_kineto_trace(
        tmp_path / "k0.json.gz",
        attn_us=420.0,
        gemm_us=580.0,
        kernel_variant="fp16",
    )
    fp8_trace_path = kineto.write_kineto_trace(
        tmp_path / "k1.json.gz",
        attn_us=190.0,
        gemm_us=260.0,
        kernel_variant="fp8",
    )
    kineto_report = cta.compare(
        fp16_trace_path, fp8_trace_path, cache_dir=cache_dir
    )
    for kind in ("breakdown", "tornado"):
        figure_path = figure_dir / f"kin_{kind}.pdf"
        kineto_report.plot(kind=kind, path=figure_path)
        assert figure_path.exists() and figure_path.stat().st_size > 500

    memory_bound_csv_path = ncu.write_ncu_csv(
        tmp_path / "n0.csv", sm_pct=34.0, dram_pct=85.0
    )
    compute_bound_csv_path = ncu.write_ncu_csv(
        tmp_path / "n1.csv",
        gemm_dur_ns=250_000.0,
        sm_pct=82.0,
        dram_pct=36.0,
        flops=4e10,
        dram_bytes=5e8,
    )
    roofline_report = cta.compare(
        memory_bound_csv_path, compute_bound_csv_path, cache_dir=cache_dir
    )
    roofline_path = figure_dir / "roofline.pdf"
    roofline_report.plot(kind="roofline", path=roofline_path)
    assert roofline_path.exists() and roofline_path.stat().st_size > 500

    with pytest.raises(ValueError, match="Unknown plot kind"):
        roofline_report.plot(kind="nonexistent_kind")


def test_multitrial_toml_manifest_stats_and_pandas_latex(
    tmp_path: pathlib.Path,
) -> None:
    # 4 trials per group exercise Welch's t-test, the Mann-Whitney U test,
    # Cohen's d and the 95% confidence interval.
    for i, compute_ms in enumerate((3.10, 3.20, 3.15, 3.25)):
        tensorrt.write_trtexec_timing(
            tmp_path / f"base_{i}.json", queries=25, compute_ms=compute_ms
        )
    for i, compute_ms in enumerate((1.40, 1.45, 1.38, 1.42)):
        tensorrt.write_trtexec_timing(
            tmp_path / f"cand_{i}.json", queries=25, compute_ms=compute_ms
        )

    manifest_path = tmp_path / "sweep.toml"
    manifest_path.write_text(
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

    groups, config = manifest.load_manifest(manifest_path)
    report = cta.run_comparison(groups, config)
    summary = report.to_polars("latency_throughput")
    row = summary.filter(summary["metric"] == "mean_step_ms").row(0, named=True)
    assert row["significant"] is True
    assert float(row["p_welch"]) < 0.01
    assert abs(float(row["cohens_d"])) > 2.0
    assert float(row["delta"]) < -1.5
    assert not report.violations

    pandas_summary = report.to_pandas("latency_throughput")
    assert len(pandas_summary) == summary.height

    tex = report.to_latex(tmp_path / "out.tex")
    assert "\\toprule" in tex and "\\bottomrule" in tex


def test_cli_end_to_end_workflow(tmp_path: pathlib.Path) -> None:
    demo_dir = tmp_path / "demo"
    cache_dir = tmp_path / "cli_cache"
    export_dir = tmp_path / "exports"
    plot_dir = tmp_path / "plots"

    # 1. cta synth
    synth_result = cli_runner.invoke(cli_app.app, ["synth", str(demo_dir)])
    assert synth_result.exit_code == 0, synth_result.output
    assert (demo_dir / "sweep.yaml").exists()

    # 2. cta inspect
    inspect_result = cli_runner.invoke(
        cli_app.app,
        [
            "inspect",
            str(demo_dir / "gpu_bundle_fp16"),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "nsys" in inspect_result.output
    assert "trtexec_timing" in inspect_result.output

    # 3. cta compare --manifest + bundle export
    compare_result = cli_runner.invoke(
        cli_app.app,
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
    assert compare_result.exit_code == 0, compare_result.output
    for file_name in (
        "report.md",
        "report.tex",
        "report.json",
        "breakdown.pdf",
        "breakdown.svg",
    ):
        assert (export_dir / file_name).exists()

    # 4. A gate violation in cta compare returns exit code 2 (the reverse
    # INT8 -> FP16 comparison regresses latency).
    gate_result = cli_runner.invoke(
        cli_app.app,
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
    assert gate_result.exit_code == 2

    # 5. cta verify-equivalence pass and fail
    equivalent_result = cli_runner.invoke(
        cli_app.app,
        [
            "verify-equivalence",
            str(demo_dir / "chakra_ring"),
            str(demo_dir / "goal" / "ring.bin"),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert equivalent_result.exit_code == 0, equivalent_result.output

    not_equivalent_result = cli_runner.invoke(
        cli_app.app,
        [
            "verify-equivalence",
            str(demo_dir / "chakra_skewed"),
            str(demo_dir / "goal" / "ring.bin"),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert not_equivalent_result.exit_code == 2

    # 6. cta plot
    plot_result = cli_runner.invoke(
        cli_app.app,
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
    assert plot_result.exit_code == 0, plot_result.output
    for kind in ("roofline", "tornado", "scaling"):
        assert (plot_dir / f"{kind}.svg").exists()

    # 7. cta cache status and clear
    status_result = cli_runner.invoke(
        cli_app.app, ["cache", "status", "--cache-dir", str(cache_dir)]
    )
    assert status_result.exit_code == 0
    assert "Cache entries:" in status_result.output
    clear_result = cli_runner.invoke(
        cli_app.app, ["cache", "clear", "--cache-dir", str(cache_dir)]
    )
    assert clear_result.exit_code == 0 and "Cleared" in clear_result.output


def test_quickstarts_tall_ncu_legacy_nsys_and_bundle_equivalence(
    tmp_path: pathlib.Path,
) -> None:
    import sqlite3

    from cta.synth import goal
    from cta.synth import nsys
    from examples import quickstart_chakra_goal
    from examples import quickstart_gpu_bundle

    cache_dir = tmp_path / "cache"

    # 1. Run both quickstart scripts end to end.
    quickstart_chakra_goal.main(tmp_path / "qs_cg")
    quickstart_gpu_bundle.main(tmp_path / "qs_gpu")
    assert (tmp_path / "qs_cg" / "figures" / "breakdown.pdf").exists()
    assert (tmp_path / "qs_gpu" / "artifacts" / "tornado.pdf").exists()

    # 2. The tall NCU details CSV format
    # ("ID,Kernel Name,Metric Name,Metric Value").
    tall_a = tmp_path / "tall_a.csv"
    tall_b = tmp_path / "tall_b.csv"
    for csv_path, duration_ns, sm_pct in (
        (tall_a, "400000", "40.0"),
        (tall_b, "200000", "82.0"),
    ):
        csv_path.write_text(
            "ID,Kernel Name,Metric Name,Metric Value\n"
            f"0,ampere_gemm_f16,gpu__time_duration.sum,{duration_ns}\n"
            "0,ampere_gemm_f16,"
            f"sm__throughput.avg.pct_of_peak_sustained_elapsed,{sm_pct}\n"
            "0,ampere_gemm_f16,dram__bytes.sum,500000000\n"
            "0,ampere_gemm_f16,"
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,20000000000\n"
        )
    tall_report = cta.compare(tall_a, tall_b, cache_dir=cache_dir)
    assert "roofline_diff" in tall_report.aspects

    # 3. A legacy Nsight Systems SQLite export with a GENERIC_EVENTS table
    # instead of GPU_METRICS.
    legacy_a_path = nsys.write_nsys_sqlite(
        tmp_path / "leg_a.sqlite", gemm_us=500
    )
    legacy_b_path = nsys.write_nsys_sqlite(
        tmp_path / "leg_b.sqlite", gemm_us=300
    )
    for db_path, sm_active_pct in (
        (legacy_a_path, 45.0),
        (legacy_b_path, 78.0),
    ):
        with sqlite3.connect(db_path) as connection:
            connection.execute("DROP TABLE GPU_METRICS")
            connection.execute("CREATE TABLE GENERIC_EVENTS (data TEXT)")
            event = str({"SM Active [Throughput %]": sm_active_pct})
            connection.execute(
                "INSERT INTO GENERIC_EVENTS VALUES (?)", (event,)
            )
    legacy_report = cta.compare(
        legacy_a_path, legacy_b_path, cache_dir=cache_dir
    )
    assert "kernel_diff" in legacy_report.aspects

    # 4. A RunBundle with both Chakra-ET and GOAL artifacts, which exercises
    # the schedule_equivalence aspect.
    ring_bundle_dir = tmp_path / "bundle_ring"
    bad_bundle_dir = tmp_path / "bundle_bad"
    chakra.write_chakra_et_run(
        ring_bundle_dir, world_size=4, compute_us=600, allreduce_bytes=400_000
    )
    goal.write_goal_allreduce(
        ring_bundle_dir / "sched.goal",
        world_size=4,
        compute_ns=600_000,
        allreduce_bytes=400_000,
        algorithm="ring",
    )
    chakra.write_chakra_et_run(
        bad_bundle_dir, world_size=4, compute_us=600, allreduce_bytes=400_000
    )
    goal.write_goal_allreduce(
        bad_bundle_dir / "sched.goal",
        world_size=4,
        compute_ns=900_000,
        allreduce_bytes=800_000,
        algorithm="ring",
    )
    equivalence_report = cta.compare(
        ring_bundle_dir, bad_bundle_dir, cache_dir=cache_dir
    )
    assert "schedule_equivalence" in equivalence_report.aspects


def test_single_trace_analyze_and_single_trace_plots(
    tmp_path: pathlib.Path,
) -> None:
    cache_dir = tmp_path / "single_cache"
    demo_dir = tmp_path / "demo_single"
    figure_dir = tmp_path / "sfigs"
    assert (
        cli_runner.invoke(cli_app.app, ["synth", str(demo_dir)]).exit_code == 0
    )

    # 1. SDK single-trace analysis of a distributed Chakra-ET run, and plots.
    chakra_report = cta.analyze(
        demo_dir / "chakra_skewed", label="Skewed_4R", cache_dir=cache_dir
    )
    assert chakra_report.mode == "single"
    assert len(chakra_report.diagnoses) == 1
    assert "Bottleneck Regime:" in chakra_report.diagnoses[0].summary
    by_comm_type = chakra_report.to_polars(
        "comm_collectives", table="by_comm_type"
    )
    assert "share_pct" in by_comm_type.columns
    for kind in ("breakdown", "topk", "traffic", "skew", "scaling"):
        figure_path = figure_dir / f"c_{kind}.pdf"
        chakra_report.plot(kind=kind, path=figure_path)
        assert figure_path.exists() and figure_path.stat().st_size > 500

    # 2. SDK single-trace analysis of an NCU roofline CSV.
    roofline_report = cta.analyze(
        demo_dir / "ncu" / "baseline.csv",
        label="NCU_Single",
        cache_dir=cache_dir,
    )
    roofline_path = figure_dir / "ncu_roofline.pdf"
    roofline_report.plot(kind="roofline", path=roofline_path)
    assert roofline_path.exists() and roofline_path.stat().st_size > 500

    # 3. CLI `cta analyze` and CLI single-trace `cta plot <single_trace>`.
    analyze_dir = tmp_path / "cli_single_out"
    analyze_result = cli_runner.invoke(
        cli_app.app,
        [
            "analyze",
            str(demo_dir / "gpu_bundle_fp16"),
            "--warmup-steps",
            "2",
            "--out-dir",
            str(analyze_dir),
            "--export",
            "markdown,latex,csv,json,pdf",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert analyze_result.exit_code == 0, analyze_result.output
    assert (analyze_dir / "report.md").exists()
    assert (analyze_dir / "breakdown.pdf").exists()

    plot_dir = tmp_path / "cli_single_plots"
    plot_result = cli_runner.invoke(
        cli_app.app,
        [
            "plot",
            str(demo_dir / "gpu_bundle_fp16"),
            "--kind",
            "breakdown,topk",
            "--format",
            "svg",
            "--out-dir",
            str(plot_dir),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert plot_result.exit_code == 0, plot_result.output
    assert (plot_dir / "breakdown.svg").exists()
    assert (plot_dir / "topk.svg").exists()
