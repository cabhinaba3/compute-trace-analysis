"""Runs a command under Nsight Compute and exports its CSV report."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

from cta.profiling import base


def build_argv(
    command: Sequence[str],
    report: pathlib.Path,
    *,
    ncu_set: str,
    metrics: Sequence[str],
    launch_count: int | None,
    kernel_regex: str | None,
    cache_control: str,
    clock_control: str,
    ncu: str = "ncu",
) -> list[str]:
    """Builds the ``ncu`` command line.

    Args:
        command: Command line to profile.
        report: Report path without the ``.ncu-rep`` suffix.
        ncu_set: Section set to collect when no ``metrics`` are given.
        metrics: Metrics to collect instead of ``ncu_set``.
        launch_count: Number of kernel launches to profile, or None for all.
        kernel_regex: Regex of kernel names to profile, or None for all.
        cache_control: ``all`` (flush caches between replays) or ``none``.
        clock_control: ``base`` (lock clocks) or ``none``.
        ncu: Path of the ncu executable.

    Returns:
        The full command line.
    """
    # Cache/clock defaults follow nsight-python's choices (flush caches
    # between replays, do not lock clocks).
    argv = [
        ncu,
        "--cache-control",
        cache_control,
        "--clock-control",
        clock_control,
        "-f",
        "-o",
        str(report),
    ]
    argv += ["--metrics", ",".join(metrics)] if metrics else ["--set", ncu_set]
    if launch_count is not None:
        argv += ["--launch-count", str(launch_count)]
    if kernel_regex:
        argv += ["-k", f"regex:{kernel_regex}"]
    return [*argv, *command]


def run(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    ncu_set: str = "full",
    metrics: Sequence[str] = (),
    launch_count: int | None = None,
    kernel_regex: str | None = None,
    cache_control: str = "all",
    clock_control: str = "none",
) -> base.ProfileResult:
    """Profiles a command with Nsight Compute.

    Needs access to the GPU hardware counters (see ``ERR_NVGPUCTRPERM``).

    Args:
        command: Command line to profile.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory and report name.
        ncu_set: Section set to collect when no ``metrics`` are given.
        metrics: Metrics to collect instead of ``ncu_set``.
        launch_count: Number of kernel launches to profile, or None for all.
        kernel_regex: Regex of kernel names to profile, or None for all.
        cache_control: ``all`` (flush caches between replays) or ``none``.
        clock_control: ``base`` (lock clocks) or ``none``.

    Returns:
        The run result.

    Raises:
        base.ProfilerError: If ncu is not found.
    """
    ncu = base.find_tool(
        "ncu",
        "NCU_PATH",
        "Install Nsight Compute (CUDA toolkit or conda-forge "
        "'nsight-compute').",
    )
    bundle_dir = base.make_bundle_dir(out_dir, name)
    report = bundle_dir / name
    argv = build_argv(
        command,
        report,
        ncu_set=ncu_set,
        metrics=metrics,
        launch_count=launch_count,
        kernel_regex=kernel_regex,
        cache_control=cache_control,
        clock_control=clock_control,
        ncu=ncu,
    )
    returncode = base.run_echoed(argv)
    artifacts = [
        path for path in (bundle_dir / f"{name}.ncu-rep",) if path.exists()
    ]
    return base.ProfileResult("ncu", bundle_dir, argv, returncode, artifacts)
