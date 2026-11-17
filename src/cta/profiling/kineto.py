"""Runs a Python command under torch.profiler (see ``kineto_runner``)."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import sys

from cta.profiling import base


def build_argv(
    command: Sequence[str], trace_path: pathlib.Path, *, step_on: str
) -> list[str]:
    """Builds the command that runs a script under ``torch.profiler``.

    The script runs through ``cta.profiling.kineto_runner`` in the current
    interpreter; a leading ``python`` executable is dropped.

    Args:
        command: Python command line to profile.
        trace_path: Trace file to write.
        step_on: ``optimizer`` to start a profiler step at each
            ``optimizer.step()``, or ``none``.

    Returns:
        The full command line.

    Raises:
        base.ProfilerError: If nothing is left to run after the interpreter.
    """
    tokens = list(command)
    if tokens and pathlib.Path(tokens[0]).name.startswith("python"):
        tokens = tokens[1:]
    if not tokens:
        raise base.ProfilerError(
            "kineto needs a Python script or '-m module' to run, e.g. -- "
            "python train.py"
        )
    return [
        sys.executable,
        "-m",
        "cta.profiling.kineto_runner",
        "--out",
        str(trace_path),
        "--step-on",
        step_on,
        "--",
        *tokens,
    ]


def run(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    step_on: str = "optimizer",
) -> base.ProfileResult:
    """Profiles a Python command with ``torch.profiler`` (Kineto trace).

    Args:
        command: Python command line to profile.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory and trace name.
        step_on: ``optimizer`` to start a profiler step at each
            ``optimizer.step()``, or ``none``.

    Returns:
        The run result.

    Raises:
        base.ProfilerError: If ``command`` names no script or module.
    """
    bundle_dir = base.make_bundle_dir(out_dir, name)
    trace = bundle_dir / f"{name}.json.gz"
    argv = build_argv(command, trace, step_on=step_on)
    returncode = base.run_echoed(argv)
    return base.ProfileResult(
        "kineto",
        bundle_dir,
        argv,
        returncode,
        [trace] if trace.exists() else [],
    )
