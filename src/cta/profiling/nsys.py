"""Runs a command under Nsight Systems and exports its SQLite database."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

from cta.profiling import base


def build_argv(
    command: Sequence[str],
    out_prefix: pathlib.Path,
    *,
    trace: str,
    gpu_metrics: bool,
    nsys: str = "nsys",
) -> list[str]:
    """Builds the ``nsys profile`` command line.

    Args:
        command: Command line to profile.
        out_prefix: Report path without the ``.nsys-rep`` suffix.
        trace: Value of ``nsys --trace``, e.g. ``cuda,nvtx,osrt``.
        gpu_metrics: Whether to sample GPU metrics on all devices.
        nsys: Path of the nsys executable.

    Returns:
        The full command line.
    """
    argv = [
        nsys,
        "profile",
        "-t",
        trace,
        "-s",
        "none",
        "--force-overwrite",
        "true",
        "-o",
        str(out_prefix),
    ]
    if gpu_metrics:
        argv.append("--gpu-metrics-devices=all")
    return [*argv, *command]


def run(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    trace: str = "cuda,nvtx,osrt",
    gpu_metrics: bool = False,
    export: bool = True,
) -> base.ProfileResult:
    """Profiles a command with Nsight Systems.

    Runs ``nsys profile`` and then ``nsys export -t sqlite``, the form CTA
    reads on any machine.

    Args:
        command: Command line to profile.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory and report name.
        trace: Value of ``nsys --trace``.
        gpu_metrics: Whether to sample GPU metrics (needs counter access).
        export: Whether to export the report to SQLite.

    Returns:
        The run result; the SQLite file is listed only if the export
        succeeded.

    Raises:
        base.ProfilerError: If nsys is not found.
    """
    nsys = base.find_tool(
        "nsys",
        "NSYS_PATH",
        "Install Nsight Systems (conda-forge 'nsight-systems' or the CUDA "
        "toolkit).",
    )
    bundle_dir = base.make_bundle_dir(out_dir, name)
    report_prefix = bundle_dir / name
    argv = build_argv(
        command, report_prefix, trace=trace, gpu_metrics=gpu_metrics, nsys=nsys
    )
    returncode = base.run_echoed(argv)
    artifacts = [
        path
        for path in (report_prefix.with_suffix(".nsys-rep"),)
        if path.exists()
    ]
    if export and artifacts:
        sqlite_path = report_prefix.with_suffix(".sqlite")
        export_argv = [
            nsys,
            "export",
            "-t",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            str(sqlite_path),
            str(artifacts[0]),
        ]
        if base.run_echoed(export_argv) == 0:
            artifacts.append(sqlite_path)
    return base.ProfileResult("nsys", bundle_dir, argv, returncode, artifacts)
