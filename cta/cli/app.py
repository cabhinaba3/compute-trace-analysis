from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from cta import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

    from cta.core.models import ComparisonReport
    from cta.profile.runners import ProfileResult

# Heavy modules (polars, scipy, profilers) are imported inside each command so that `cta --help`/`--version` stay fast.

app = typer.Typer(
    name="cta",
    help="CTA v1.0: Multi-trace & multi-aspect performance comparison framework.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Manage the content-addressed Parquet trace cache.")
app.add_typer(cache_app, name="cache")
profile_app = typer.Typer(
    help="Collect a profile with the vendor tool (nsys, ncu, kineto, trtexec, jtop) in a form `cta analyze` reads.",
    no_args_is_help=True,
)
app.add_typer(profile_app, name="profile")
console = Console()
err_console = Console(stderr=True)

_FORMAT_HELP = "Stdout format: 'rich' (tables) or 'json' (machine-readable, for piping to jq)."


def _version_cb(value: bool) -> None:
    if value:
        typer.echo(f"cta {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[bool, typer.Option("--version", callback=_version_cb, is_eager=True, help="Show version and exit.")] = False,
) -> None:
    pass


def _csv(value: str | None) -> list[str] | None:
    return [x.strip() for x in value.split(",") if x.strip()] if value else None


def _emit(report: ComparisonReport, output: str, out_dir: Path | None, export: str) -> None:
    """Render to stdout (rich|json) and optionally export files; status lines go to stderr in json mode."""
    from cta.exporters.tabular import report_to_json
    from cta.exporters.terminal import render_terminal_report

    if output not in ("rich", "json"):
        raise typer.BadParameter(f"Unknown --format '{output}' (expected 'rich' or 'json').")
    log = err_console if output == "json" else console
    if output == "json":
        typer.echo(report_to_json(report))
    else:
        render_terminal_report(report, console=console)
    if out_dir is not None:
        for fmt_name, fpath in report.export(out_dir, formats=_csv(export) or []).items():
            log.print(f"[green]✓ Exported {fmt_name}:[/green] {fpath}")


@app.command("compare")
def compare_cmd(
    traces: Annotated[list[str] | None, typer.Argument(help="Two or more trace files, globs, or RunBundle directories.")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest", "-m", help="Declarative YAML/TOML sweep manifest.")] = None,
    aspects: Annotated[str | None, typer.Option("--aspects", "-a", help="Comma-separated aspect names.")] = None,
    warmup_steps: Annotated[int | None, typer.Option("--warmup-steps", help="Skip initial warmup steps/iterations (overrides manifest).")] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Max steps/iterations to analyze (overrides manifest).")] = None,
    gate: Annotated[str | None, typer.Option("--gate", "-g", help="Regression gates, e.g. 'p99_latency_ms:+5%'")] = None,
    allow_missing_gates: Annotated[bool, typer.Option("--allow-missing-gates", help="Do not fail when a gate metric is absent from the report.")] = False,
    collective_algo: Annotated[str | None, typer.Option("--collective-algo", help="Collective decomposition model: ring, tree, direct (default ring; overrides manifest).")] = None,
    alpha: Annotated[float | None, typer.Option("--alpha", help="Significance level for Welch/Mann-Whitney tests (default 0.05; overrides manifest).")] = None,
    scope_depth: Annotated[int | None, typer.Option("--scope-depth", min=1, help="Group kernels by the N-th enclosing op/range from the outermost (default: innermost).")] = None,
    labels: Annotated[str | None, typer.Option("--labels", "-l", help="Comma-separated group labels.")] = None,
    out_dir: Annotated[Path | None, typer.Option("--out-dir", "-o", help="Directory to export reports/plots.")] = None,
    export: Annotated[str, typer.Option("--export", help="Comma-separated export formats (markdown,latex,csv,json,pdf,svg,png).")] = "markdown,csv,json",
    output: Annotated[str, typer.Option("--format", help=_FORMAT_HELP)] = "rich",
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", help="Cache directory (default .cta_cache; overrides manifest).")] = None,
) -> None:
    """Compare two (Baseline vs Candidate) or N traces/bundles across performance aspects."""
    from cta.core.diagnosis import parse_gate_spec
    from cta.core.engine import compare as run_compare_sdk
    from cta.core.engine import run_comparison
    from cta.core.manifest import load_manifest

    try:
        gates = parse_gate_spec(gate)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--gate") from exc
    try:
        if manifest is not None:
            if traces:
                err_console.print("[yellow]warning: positional traces are ignored when --manifest is given.[/yellow]")
            groups, cfg = load_manifest(manifest)
            overrides = {
                "aspects": tuple(_csv(aspects) or ()) or None,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "collective_algo": collective_algo,
                "alpha": alpha,
                "scope_depth": scope_depth,
                "cache_dir": cache_dir,
            }
            for key, val in overrides.items():
                if val is not None:
                    setattr(cfg, key, val)
            cfg.gates.update(gates)
            cfg.allow_missing_gates = cfg.allow_missing_gates or allow_missing_gates
            report = run_comparison(groups, cfg)
        else:
            if not traces or len(traces) < 2:
                raise typer.BadParameter("Provide at least two trace paths/directories or --manifest.")
            report = run_compare_sdk(
                traces[0],
                traces[1],
                *traces[2:],
                aspects=_csv(aspects),
                warmup_steps=warmup_steps or 0,
                max_steps=max_steps,
                gates=gates,
                allow_missing_gates=allow_missing_gates,
                collective_algo=collective_algo or "ring",
                alpha=alpha if alpha is not None else 0.05,
                scope_depth=scope_depth,
                cache_dir=cache_dir or Path(".cta_cache"),
                labels=_csv(labels),
            )
    except ValueError as exc:  # unknown gate metric, no common formats, inapplicable aspects, ...
        err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    _emit(report, output, out_dir, export)
    if report.violations:
        raise typer.Exit(code=2)


@app.command("equiv")
def equiv_cmd(
    chakra_trace: Annotated[str, typer.Argument(help="Chakra-ET directory, glob, or .et file.")],
    goal_schedule: Annotated[str, typer.Argument(help="GOAL .goal or .bin schedule file.")],
    collective_algo: Annotated[str, typer.Option("--collective-algo", help="Collective decomposition algorithm (ring, tree, direct).")] = "ring",
    rel_tol: Annotated[float, typer.Option("--rel-tol", help="Relative error tolerance.")] = 0.02,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache"),
) -> None:
    """Verify compute duration and decomposed collective byte equivalence between Chakra-ET and GOAL."""
    from cta.aspects.distributed import verify_chakra_goal_equivalence

    verdict = verify_chakra_goal_equivalence(
        chakra_trace,
        goal_schedule,
        collective_algo=collective_algo,
        rel_tol=rel_tol,
        cache_dir=cache_dir,
    )
    tbl = Table(title="Chakra-ET <-> GOAL Equivalence Check")
    for col in ("check", "expected", "observed", "rel_error", "passed"):
        tbl.add_column(col, justify="left" if col == "check" else "right")
    for row in verdict.checks.iter_rows(named=True):
        ok_str = "[green]PASS[/green]" if row["passed"] else "[bold red]FAIL[/bold red]"
        tbl.add_row(
            str(row["check"]),
            f"{float(row['expected']):.4g}",
            f"{float(row['observed']):.4g}",
            f"{float(row['rel_error']) * 100:.2f}%",
            ok_str,
        )
    console.print(tbl)
    if not verdict.equivalent:
        raise typer.Exit(code=2)


@app.command("analyze")
def analyze_cmd(
    trace: Annotated[str, typer.Argument(help="Single trace file, glob, or RunBundle directory.")],
    aspects: Annotated[str | None, typer.Option("--aspects", "-a", help="Comma-separated aspect names.")] = None,
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", help="Skip initial warmup steps/iterations.")] = 0,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Max steps/iterations to analyze.")] = None,
    collective_algo: Annotated[str, typer.Option("--collective-algo", help="Collective decomposition model (ring, tree, direct).")] = "ring",
    alpha: Annotated[float, typer.Option("--alpha", help="Significance level for statistical tests.")] = 0.05,
    scope_depth: Annotated[int | None, typer.Option("--scope-depth", min=1, help="Group kernels by the N-th enclosing op/range from the outermost (default: innermost).")] = None,
    label: Annotated[str | None, typer.Option("--label", "-l", help="Display label for the trace/bundle.")] = None,
    out_dir: Annotated[Path | None, typer.Option("--out-dir", "-o", help="Directory to export reports/plots.")] = None,
    export: Annotated[str, typer.Option("--export", help="Comma-separated export formats (markdown,latex,csv,json,pdf,svg,png).")] = "markdown,csv,json",
    output: Annotated[str, typer.Option("--format", help=_FORMAT_HELP)] = "rich",
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Cache directory.")] = Path(".cta_cache"),
) -> None:
    """Analyze and characterize a single trace, glob, or multi-artifact RunBundle directory."""
    from cta.core.engine import analyze as run_analyze_sdk

    report = run_analyze_sdk(
        trace,
        aspects=_csv(aspects),
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        collective_algo=collective_algo,
        alpha=alpha,
        scope_depth=scope_depth,
        cache_dir=cache_dir,
        label=label,
    )
    _emit(report, output, out_dir, export)


@app.command("plot")
def plot_cmd(
    traces: Annotated[list[str] | None, typer.Argument(help="One (single-trace) or more (comparative) trace files/bundles.")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest", "-m", help="Declarative YAML/TOML sweep manifest.")] = None,
    kind: Annotated[str, typer.Option("--kind", "-k", help="Comma-separated plot kinds (breakdown,topk,traffic_delta,skew,roofline,scaling; tornado = alias of topk).")] = "breakdown",
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory for generated figures.")] = Path("figures"),
    fmt: Annotated[str, typer.Option("--format", "-f", help="Figure file extension (pdf, svg, png).")] = "pdf",
    style: Annotated[str, typer.Option("--style", help="Preset style (paper_single_col, paper_double_col, presentation).")] = "paper_single_col",
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", help="Skip initial warmup steps/iterations (ignored with --manifest).")] = 0,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Max steps/iterations to analyze (ignored with --manifest).")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache"),
) -> None:
    """Render publication-ready single-trace or comparative plots to PDF, SVG, or PNG."""
    from cta.core.engine import analyze as run_analyze_sdk
    from cta.core.engine import compare as run_compare_sdk
    from cta.core.engine import run_comparison
    from cta.core.manifest import load_manifest

    if manifest is not None:
        groups, cfg = load_manifest(manifest)
        report = run_comparison(groups, cfg)
    else:
        if not traces:
            raise typer.BadParameter("Provide at least one trace path/directory or --manifest.")
        if len(traces) == 1:
            report = run_analyze_sdk(traces[0], warmup_steps=warmup_steps, max_steps=max_steps, cache_dir=cache_dir)
        else:
            report = run_compare_sdk(
                traces[0], traces[1], *traces[2:], warmup_steps=warmup_steps, max_steps=max_steps, cache_dir=cache_dir
            )

    ext = fmt.lstrip(".").lower()
    for k in (x.strip() for x in kind.split(",") if x.strip()):
        dst = out_dir / f"{k}.{ext}"
        report.plot(kind=k, path=dst, style=style)
        console.print(f"[green]✓ Saved figure:[/green] {dst}")


@app.command("synth")
def synth_cmd(
    out_dir: Annotated[Path, typer.Argument(help="Directory to populate with synthetic demo traces & sweep manifest.")] = Path("demo_traces"),
) -> None:
    """Generate miniature valid traces across all 7 ecosystems for demos and offline testing."""
    from cta.synth.generators import (
        compile_goal_to_bin,
        write_chakra_et_run,
        write_goal_allreduce,
        write_jtop_trace,
        write_kineto_trace,
        write_ncu_csv,
        write_nsys_sqlite,
        write_trtexec_layers,
        write_trtexec_timing,
    )

    write_chakra_et_run(out_dir / "chakra_ring", world_size=4, compute_us=900, allreduce_bytes=400_000)
    write_chakra_et_run(out_dir / "chakra_skewed", world_size=4, compute_us=900, allreduce_bytes=800_000, straggler_rank=2, straggler_factor=1.8)

    g_ring = write_goal_allreduce(out_dir / "goal" / "ring.goal", world_size=4, compute_ns=900_000, allreduce_bytes=400_000, algo="ring")
    compile_goal_to_bin(g_ring, out_dir / "goal" / "ring.bin")
    write_goal_allreduce(out_dir / "goal" / "direct.goal", world_size=4, compute_ns=900_000, allreduce_bytes=400_000, algo="direct")

    for mode, comp_ms, p_mw, prec, fused in (("fp16", 3.2, 22_000.0, "FP16", False), ("int8", 1.3, 14_000.0, "INT8", True)):
        bdir = out_dir / f"gpu_bundle_{mode}"
        write_nsys_sqlite(bdir / "profile.sqlite", gemm_us=int(comp_ms * 300), comm_us=250)
        write_trtexec_timing(bdir / "times.json", queries=40, compute_ms=comp_ms)
        write_trtexec_layers(bdir / "profile.json", bdir / "engine.json", fused=fused, precision=prec)
        write_jtop_trace(bdir / "jtop.json", duration_s=4, power_mw=p_mw)

    write_kineto_trace(out_dir / "kineto" / "eager.json.gz", attn_us=450.0, gemm_us=600.0, kernel_variant="fp16")
    write_kineto_trace(out_dir / "kineto" / "compiled_fp8.json.gz", attn_us=210.0, gemm_us=280.0, kernel_variant="fp8")
    write_ncu_csv(out_dir / "ncu" / "baseline.csv", sm_pct=35.0, dram_pct=84.0)
    write_ncu_csv(out_dir / "ncu" / "optimized.csv", gemm_dur_ns=260_000.0, sm_pct=82.0, dram_pct=38.0, flops=4e10, dram_bytes=5e8)

    (out_dir / "sweep.yaml").write_text(
        """groups:
  - id: fp16
    label: Jetson_FP16
    path: gpu_bundle_fp16
  - id: int8
    label: Jetson_INT8
    path: gpu_bundle_int8
config:
  warmup_steps: 2
  gates:
    mean_step_ms: 5.0
"""
    )
    console.print(f"[green]✓ Generated synthetic multi-format demo workspace in:[/green] {out_dir}")


@app.command("inspect")
def inspect_cmd(
    source: Annotated[str, typer.Argument(help="Trace file, glob, or RunBundle directory.")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache"),
) -> None:
    """Inspect detected trace formats, world size, and normalized Parquet table schemas."""
    from cta.core.engine import inspect_trace

    info = inspect_trace(source, cache_dir=cache_dir)
    console.print(Pretty(info))


@app.command("list")
def list_cmd() -> None:
    """List registered trace formats (adapters) and comparison aspects, including plugins."""
    from cta.core.registry import ADAPTERS, ASPECTS, load_plugins

    load_plugins()
    tbl = Table(title="Trace formats")
    tbl.add_column("format")
    tbl.add_column("adapter version")
    for name, adapter in sorted(ADAPTERS.items()):
        tbl.add_row(name, adapter.version)
    console.print(tbl)
    tbl = Table(title="Aspects")
    for col in ("name", "title", "requires"):
        tbl.add_column(col)
    for name, aspect in sorted(ASPECTS.items()):
        mode = "any of" if aspect.any_of_formats else "all of"
        tbl.add_row(name, aspect.title, f"{mode}: {', '.join(aspect.required_formats)}")
    console.print(tbl)


_PROFILE_CTX = {"allow_extra_args": True, "ignore_unknown_options": True}
_OutDir = Annotated[Path, typer.Option("--out-dir", "-o", help="Parent directory; the profile goes to <out-dir>/<name>/.")]
_Name = Annotated[str | None, typer.Option("--name", "-n", help="Bundle name (default: script or executable name).")]
_Analyze = Annotated[bool, typer.Option("--analyze", help="Run `cta analyze` on the collected profile afterwards.")]


def _finish(result: "ProfileResult", analyze: bool) -> None:  # noqa: UP037 (ProfileResult is imported lazily)
    for art in result.artifacts:
        console.print(f"[green]✓ {result.tool}:[/green] {art}")
    if not result.artifacts:
        err_console.print(f"[bold red]error:[/bold red] {result.tool} produced no artifacts in {result.bundle_dir}")
        raise typer.Exit(code=result.returncode or 1)
    if result.returncode != 0:
        err_console.print(f"[yellow]warning: the profiled command exited with status {result.returncode}[/yellow]")
    if analyze:
        from cta.core.engine import analyze as run_analyze_sdk
        from cta.exporters.terminal import render_terminal_report

        render_terminal_report(run_analyze_sdk(result.bundle_dir), console=console)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _run_profile(ctx: typer.Context, tool: str, run: "Callable[[list[str], str], ProfileResult]", name: str | None, analyze: bool) -> None:  # noqa: UP037
    from cta.profile.runners import ProfilerError, default_name

    command = list(ctx.args)
    if not command:
        raise typer.BadParameter(f"give the command to profile after '--', e.g. cta profile {tool} -- python train.py")
    try:
        result = run(command, name or default_name(command))
    except ProfilerError as exc:
        err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc
    _finish(result, analyze)


@profile_app.command("nsys", context_settings=_PROFILE_CTX)
def profile_nsys(
    ctx: typer.Context,
    out_dir: _OutDir = Path("profiles"),
    name: _Name = None,
    trace: Annotated[str, typer.Option(help="nsys --trace list.")] = "cuda,nvtx,osrt",
    gpu_metrics: Annotated[bool, typer.Option("--gpu-metrics", help="Sample GPU metrics (needs counter access).")] = False,
    analyze: _Analyze = False,
) -> None:
    """Nsight Systems: `cta profile nsys -- python train.py` (writes .nsys-rep and the .sqlite export)."""
    from cta.profile.runners import run_nsys

    _run_profile(ctx, "nsys", lambda cmd, n: run_nsys(cmd, out_dir, n, trace=trace, gpu_metrics=gpu_metrics), name, analyze)


@profile_app.command("ncu", context_settings=_PROFILE_CTX)
def profile_ncu(
    ctx: typer.Context,
    out_dir: _OutDir = Path("profiles"),
    name: _Name = None,
    ncu_set: Annotated[str, typer.Option("--set", help="ncu section set (basic, detailed, full, roofline).")] = "full",
    metrics: Annotated[str | None, typer.Option(help="Comma-separated metrics instead of a --set.")] = None,
    launch_count: Annotated[int | None, typer.Option(help="Profile only the first N kernel launches.")] = None,
    kernel: Annotated[str | None, typer.Option("--kernel", "-k", help="Regex of kernel names to profile.")] = None,
    cache_control: Annotated[str, typer.Option(help="all (flush between replays) or none.")] = "all",
    clock_control: Annotated[str, typer.Option(help="base (lock clocks) or none.")] = "none",
    analyze: _Analyze = False,
) -> None:
    """Nsight Compute (hardware counters required): `cta profile ncu --launch-count 50 -- python train.py`."""
    from cta.profile.runners import run_ncu

    mlist = tuple(m.strip() for m in metrics.split(",") if m.strip()) if metrics else ()
    _run_profile(
        ctx, "ncu",
        lambda cmd, n: run_ncu(cmd, out_dir, n, ncu_set=ncu_set, metrics=mlist, launch_count=launch_count,
                               kernel_regex=kernel, cache_control=cache_control, clock_control=clock_control),
        name, analyze,
    )  # fmt: skip


@profile_app.command("kineto", context_settings=_PROFILE_CTX)
def profile_kineto(
    ctx: typer.Context,
    out_dir: _OutDir = Path("profiles"),
    name: _Name = None,
    step_on: Annotated[str, typer.Option(help="optimizer: each optimizer.step() starts a ProfilerStep; none: one step.")] = "optimizer",
    analyze: _Analyze = False,
) -> None:
    """PyTorch profiler (Kineto trace): `cta profile kineto -- python train.py` or `-- -m package.module`."""
    from cta.profile.runners import run_kineto

    _run_profile(ctx, "kineto", lambda cmd, n: run_kineto(cmd, out_dir, n, step_on=step_on), name, analyze)


@profile_app.command("trtexec", context_settings=_PROFILE_CTX)
def profile_trtexec(
    ctx: typer.Context,
    out_dir: _OutDir = Path("profiles"),
    name: _Name = None,
    analyze: _Analyze = False,
) -> None:
    """TensorRT: `cta profile trtexec -- --loadEngine=model.plan --iterations=200` (export flags are added)."""
    from cta.profile.runners import run_trtexec

    _run_profile(ctx, "trtexec", lambda args, n: run_trtexec(args, out_dir, n), name or "trtexec", analyze)


@profile_app.command("jtop", context_settings=_PROFILE_CTX)
def profile_jtop(
    ctx: typer.Context,
    out_dir: _OutDir = Path("profiles"),
    name: _Name = None,
    interval: Annotated[float, typer.Option(help="Sampling interval in seconds.")] = 0.5,
    analyze: _Analyze = False,
) -> None:
    """Jetson power/utilisation telemetry while a command runs: `cta profile jtop -- python infer.py`."""
    from cta.profile.runners import run_jtop

    _run_profile(ctx, "jtop", lambda cmd, n: run_jtop(cmd, out_dir, n, interval=interval), name, analyze)


@cache_app.command("status")
def cache_status(cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache")) -> None:
    from cta.core.cache import CacheStore

    st = CacheStore(cache_dir).status()
    console.print(f"Cache entries: [bold]{st['entries']}[/bold], Size: [bold]{st['bytes'] / 1e6:.2f} MB[/bold]")


@cache_app.command("clear")
def cache_clear(cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache")) -> None:
    from cta.core.cache import CacheStore

    removed = CacheStore(cache_dir).clear()
    console.print(f"Cleared [bold]{removed}[/bold] cached trace entries.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
