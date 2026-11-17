"""Runs a Python command under jax.profiler (see ``jax_runner``)."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import sys

from cta.profiling import base


def build_argv(
    command: Sequence[str], log_dir: pathlib.Path, *, perfetto: bool = True
) -> list[str]:
    """Builds the command that runs a script under ``jax.profiler.trace``.

    The script runs through ``cta.profiling.jax_runner`` in the current
    interpreter; a leading ``python`` executable is dropped.

    Args:
        command: Python command line to profile.
        log_dir: XProf log directory to write.
        perfetto: Whether to also write a Chrome-format trace.

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
            "jax needs a Python script or '-m module' to run, e.g. -- "
            "python train.py"
        )
    perfetto_flags = [] if perfetto else ["--no-perfetto"]
    return [
        sys.executable,
        "-m",
        "cta.profiling.jax_runner",
        "--out",
        str(log_dir),
        *perfetto_flags,
        "--",
        *tokens,
    ]


def run(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    perfetto: bool = True,
) -> base.ProfileResult:
    """Profiles a Python command with the JAX profiler (XProf).

    Args:
        command: Python command line to profile.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory name; it is also the log directory.
        perfetto: Whether to also write a Chrome-format trace.

    Returns:
        The run result, listing the files under ``plugins/profile/``.

    Raises:
        base.ProfilerError: If ``command`` names no script or module.
    """
    bundle_dir = base.make_bundle_dir(out_dir, name)
    argv = build_argv(command, bundle_dir, perfetto=perfetto)
    returncode = base.run_echoed(argv)
    artifacts = sorted(bundle_dir.glob("plugins/profile/*/*"))
    return base.ProfileResult("jax", bundle_dir, argv, returncode, artifacts)
