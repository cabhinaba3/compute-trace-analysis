from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.pretty import Pretty

from cta.core.cache import CacheStore
from cta.core.diagnosis import parse_gate_spec
from cta.core.engine import compare as run_compare_sdk
from cta.core.engine import inspect_trace, run_comparison
from cta.core.manifest import load_manifest
from cta.exporters.terminal import render_terminal_report

app = typer.Typer(
    name="cta",
    help="CTA v1.0: Multi-trace & multi-aspect performance comparison framework.",
    no_args_is_help=True,
)
cache_app = typer.Typer(help="Manage the content-addressed Parquet trace cache.")
app.add_typer(cache_app, name="cache")
console = Console()


@app.command("compare")
def compare_cmd(
    traces: Annotated[list[str] | None, typer.Argument(help="Two or more trace files, globs, or RunBundle directories.")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest", "-m", help="Declarative YAML/TOML sweep manifest.")] = None,
    aspects: Annotated[str | None, typer.Option("--aspects", "-a", help="Comma-separated aspect names.")] = None,
    warmup_steps: Annotated[int, typer.Option("--warmup-steps", help="Skip initial warmup steps/iterations.")] = 0,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Max steps/iterations to analyze.")] = None,
    gate: Annotated[str | None, typer.Option("--gate", "-g", help="Regression gates, e.g. 'p99_latency_ms:+5%'")] = None,
    collective_algo: Annotated[str, typer.Option("--collective-algo", help="Collective decomposition model (ring, tree, direct).")] = "ring",
    labels: Annotated[str | None, typer.Option("--labels", "-l", help="Comma-separated group labels.")] = None,
    out_dir: Annotated[Path | None, typer.Option("--out-dir", "-o", help="Directory to export reports/plots.")] = None,
    export: Annotated[str, typer.Option("--export", help="Comma-separated export formats (markdown,latex,csv,json,pdf).")] = "markdown,csv,json",
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Cache directory.")] = Path(".cta_cache"),
) -> None:
    """Compare two (Baseline vs Candidate) or N traces/bundles across performance aspects."""
    aspect_list = [x.strip() for x in aspects.split(",") if x.strip()] if aspects else None
    gates = parse_gate_spec(gate)
    if manifest is not None:
        groups, cfg = load_manifest(manifest)
        if aspect_list is not None:
            cfg.aspects = tuple(aspect_list)
        if gates:
            cfg.gates.update(gates)
        report = run_comparison(groups, cfg)
    else:
        if not traces or len(traces) < 2:
            raise typer.BadParameter("Provide at least two trace paths/directories or --manifest.")
        lbl_list = [x.strip() for x in labels.split(",") if x.strip()] if labels else None
        report = run_compare_sdk(
            traces[0],
            traces[1],
            *traces[2:],
            aspects=aspect_list,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            gates=gates,
            collective_algo=collective_algo,
            cache_dir=cache_dir,
            labels=lbl_list,
        )

    render_terminal_report(report, console=console)
    if out_dir is not None:
        fmts = [x.strip() for x in export.split(",") if x.strip()]
        written = report.export(out_dir, formats=fmts)
        for fmt_name, fpath in written.items():
            console.print(f"[green]✓ Exported {fmt_name}:[/green] {fpath}")

    if report.violations:
        raise typer.Exit(code=2)


@app.command("inspect")
def inspect_cmd(
    source: Annotated[str, typer.Argument(help="Trace file, glob, or RunBundle directory.")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache"),
) -> None:
    """Inspect detected trace formats, world size, and normalized Parquet table schemas."""
    info = inspect_trace(source, cache_dir=cache_dir)
    console.print(Pretty(info))


@cache_app.command("status")
def cache_status(cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache")) -> None:
    st = CacheStore(cache_dir).status()
    console.print(f"Cache entries: [bold]{st['entries']}[/bold], Size: [bold]{st['bytes'] / 1e6:.2f} MB[/bold]")


@cache_app.command("clear")
def cache_clear(cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cta_cache")) -> None:
    removed = CacheStore(cache_dir).clear()
    console.print(f"Cleared [bold]{removed}[/bold] cached trace entries.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
