"""The ``profile`` command group: collect profiles with vendor tools."""

from __future__ import annotations

import pathlib
from typing import Annotated, TYPE_CHECKING

import typer

from cta.cli import output

if TYPE_CHECKING:
    from collections.abc import Callable

    from cta.profile import runners

profile_app = typer.Typer(
    help=(
        "Collect a profile with the vendor tool (nsys, ncu, kineto, jax, "
        "trtexec, jtop) in a form `cta analyze` reads."
    ),
    no_args_is_help=True,
)


# Extra arguments and unknown options stay in `ctx.args`, which holds the
# command to profile.
_PROFILE_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


_OutDirOption = Annotated[
    pathlib.Path,
    typer.Option(
        "--out-dir",
        "-o",
        help="Parent directory; the profile goes to <out-dir>/<name>/.",
    ),
]


_NameOption = Annotated[
    str | None,
    typer.Option(
        "--name", "-n", help="Bundle name (default: script or executable name)."
    ),
]


_AnalyzeOption = Annotated[
    bool,
    typer.Option(
        "--analyze",
        help="Run `cta analyze` on the collected profile afterwards.",
    ),
]


def _finish_profile(result: runners.ProfileResult, analyze: bool) -> None:
    """Reports the collected artifacts and optionally analyses the bundle.

    Args:
        result: The profiler run to report on.
        analyze: Whether to print a ``cta analyze`` report of the bundle.

    Raises:
        typer.Exit: If no artifact was produced, or with the profiled
            command's exit status when it failed.
    """
    for artifact_path in result.artifacts:
        output.console.print(f"[green]✓ {result.tool}:[/green] {artifact_path}")
    if not result.artifacts:
        output.err_console.print(
            f"[bold red]error:[/bold red] {result.tool} produced no artifacts "
            f"in {result.bundle_dir}"
        )
        raise typer.Exit(code=result.returncode or 1)
    if result.returncode != 0:
        output.err_console.print(
            "[yellow]warning: the profiled command exited with status "
            f"{result.returncode}[/yellow]"
        )
    if analyze:
        from cta.core import engine
        from cta.exporters import terminal

        terminal.render_terminal_report(
            engine.analyze(result.bundle_dir), console=output.console
        )
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _run_profile(
    ctx: typer.Context,
    tool: str,
    run_profiler: Callable[[list[str], str], runners.ProfileResult],
    name: str | None,
    analyze: bool,
) -> None:
    """Profiles the command given after the options and reports the result.

    Args:
        ctx: The Typer context; its extra arguments are the command to run.
        tool: Profiler name, used in messages.
        run_profiler: Runs the profiler on a command and bundle name.
        name: Bundle name; defaults to one derived from the command.
        analyze: Whether to analyse the collected bundle afterwards.

    Raises:
        typer.BadParameter: If no command was given.
        typer.Exit: If the profiler fails or the profiled command fails.
    """
    from cta.profile import runners

    command = list(ctx.args)
    if not command:
        raise typer.BadParameter(
            "give the command to profile after '--', e.g. cta profile "
            f"{tool} -- python train.py"
        )
    try:
        result = run_profiler(command, name or runners.default_name(command))
    except runners.ProfilerError as exc:
        output.err_console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc
    _finish_profile(result, analyze)


@profile_app.command("nsys", context_settings=_PROFILE_CONTEXT_SETTINGS)
def profile_nsys(
    ctx: typer.Context,
    out_dir: _OutDirOption = pathlib.Path("profiles"),
    name: _NameOption = None,
    trace: Annotated[
        str, typer.Option(help="nsys --trace list.")
    ] = "cuda,nvtx,osrt",
    gpu_metrics: Annotated[
        bool,
        typer.Option(
            "--gpu-metrics", help="Sample GPU metrics (needs counter access)."
        ),
    ] = False,
    analyze: _AnalyzeOption = False,
) -> None:
    """Nsight Systems: `cta profile nsys -- python train.py` (writes .nsys-rep and the .sqlite export)."""  # noqa: E501
    from cta.profile import runners

    _run_profile(
        ctx,
        "nsys",
        lambda command, bundle_name: runners.run_nsys(
            command,
            out_dir,
            bundle_name,
            trace=trace,
            gpu_metrics=gpu_metrics,
        ),
        name,
        analyze,
    )


@profile_app.command("ncu", context_settings=_PROFILE_CONTEXT_SETTINGS)
def profile_ncu(
    ctx: typer.Context,
    out_dir: _OutDirOption = pathlib.Path("profiles"),
    name: _NameOption = None,
    ncu_set: Annotated[
        str,
        typer.Option(
            "--set", help="ncu section set (basic, detailed, full, roofline)."
        ),
    ] = "full",
    metrics: Annotated[
        str | None,
        typer.Option(help="Comma-separated metrics instead of a --set."),
    ] = None,
    launch_count: Annotated[
        int | None,
        typer.Option(help="Profile only the first N kernel launches."),
    ] = None,
    kernel: Annotated[
        str | None,
        typer.Option(
            "--kernel", "-k", help="Regex of kernel names to profile."
        ),
    ] = None,
    cache_control: Annotated[
        str, typer.Option(help="all (flush between replays) or none.")
    ] = "all",
    clock_control: Annotated[
        str, typer.Option(help="base (lock clocks) or none.")
    ] = "none",
    analyze: _AnalyzeOption = False,
) -> None:
    """Nsight Compute (hardware counters required): `cta profile ncu --launch-count 50 -- python train.py`."""  # noqa: E501
    from cta.profile import runners

    metric_names = (
        tuple(metric.strip() for metric in metrics.split(",") if metric.strip())
        if metrics
        else ()
    )
    _run_profile(
        ctx,
        "ncu",
        lambda command, bundle_name: runners.run_ncu(
            command,
            out_dir,
            bundle_name,
            ncu_set=ncu_set,
            metrics=metric_names,
            launch_count=launch_count,
            kernel_regex=kernel,
            cache_control=cache_control,
            clock_control=clock_control,
        ),
        name,
        analyze,
    )


@profile_app.command("kineto", context_settings=_PROFILE_CONTEXT_SETTINGS)
def profile_kineto(
    ctx: typer.Context,
    out_dir: _OutDirOption = pathlib.Path("profiles"),
    name: _NameOption = None,
    step_on: Annotated[
        str,
        typer.Option(
            help=(
                "optimizer: each optimizer.step() starts a ProfilerStep; "
                "none: one step."
            )
        ),
    ] = "optimizer",
    analyze: _AnalyzeOption = False,
) -> None:
    """PyTorch profiler (Kineto trace): `cta profile kineto -- python train.py` or `-- -m package.module`."""  # noqa: E501
    from cta.profile import runners

    _run_profile(
        ctx,
        "kineto",
        lambda command, bundle_name: runners.run_kineto(
            command, out_dir, bundle_name, step_on=step_on
        ),
        name,
        analyze,
    )


@profile_app.command("jax", context_settings=_PROFILE_CONTEXT_SETTINGS)
def profile_jax(
    ctx: typer.Context,
    out_dir: _OutDirOption = pathlib.Path("profiles"),
    name: _NameOption = None,
    perfetto: Annotated[
        bool,
        typer.Option(
            help="Also write a Chrome-format trace (readable without xprof)."
        ),
    ] = True,
    analyze: _AnalyzeOption = False,
) -> None:
    """JAX profiler (XProf): `cta profile jax -- python train.py` or `-- -m package.module`."""  # noqa: E501
    from cta.profile import runners

    _run_profile(
        ctx,
        "jax",
        lambda command, bundle_name: runners.run_jax(
            command, out_dir, bundle_name, perfetto=perfetto
        ),
        name,
        analyze,
    )


@profile_app.command("trtexec", context_settings=_PROFILE_CONTEXT_SETTINGS)
def profile_trtexec(
    ctx: typer.Context,
    out_dir: _OutDirOption = pathlib.Path("profiles"),
    name: _NameOption = None,
    analyze: _AnalyzeOption = False,
) -> None:
    """TensorRT: `cta profile trtexec -- --loadEngine=model.plan --iterations=200` (export flags are added)."""  # noqa: E501
    from cta.profile import runners

    _run_profile(
        ctx,
        "trtexec",
        lambda trtexec_args, bundle_name: runners.run_trtexec(
            trtexec_args, out_dir, bundle_name
        ),
        name or "trtexec",
        analyze,
    )


@profile_app.command("jtop", context_settings=_PROFILE_CONTEXT_SETTINGS)
def profile_jtop(
    ctx: typer.Context,
    out_dir: _OutDirOption = pathlib.Path("profiles"),
    name: _NameOption = None,
    interval: Annotated[
        float, typer.Option(help="Sampling interval in seconds.")
    ] = 0.5,
    analyze: _AnalyzeOption = False,
) -> None:
    """Jetson power/utilisation telemetry while a command runs: `cta profile jtop -- python infer.py`."""  # noqa: E501
    from cta.profile import runners

    _run_profile(
        ctx,
        "jtop",
        lambda command, bundle_name: runners.run_jtop(
            command, out_dir, bundle_name, interval=interval
        ),
        name,
        analyze,
    )
