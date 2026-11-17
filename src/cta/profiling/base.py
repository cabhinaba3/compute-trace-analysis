"""What every profiler runner shares: errors, results, tool lookup.

Every runner writes into ``<out_dir>/<name>/``, so the directory is a
ready-made run bundle for ``cta analyze``. The real tools do the work; no
runner is imported until a profile command runs.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os
import pathlib
import shlex
import shutil
import subprocess
import sys


class ProfilerError(RuntimeError):
    """Raised when a profiler is missing or the command cannot be profiled."""


@dataclasses.dataclass(slots=True)
class ProfileResult:
    """Outcome of one profiler run.

    Attributes:
        tool: Profiler name, e.g. ``nsys``.
        bundle_dir: Bundle directory the profile was written to.
        command: Full command line that was run.
        returncode: Exit status of that command.
        artifacts: Profile files that were written.
    """

    tool: str
    bundle_dir: pathlib.Path
    command: list[str]
    returncode: int
    artifacts: list[pathlib.Path] = dataclasses.field(default_factory=list)


def find_tool(executable: str, env_var: str, hint: str) -> str:
    """Returns the tool path from ``env_var``, else from ``PATH``.

    Args:
        executable: Executable name to look up on ``PATH``.
        env_var: Environment variable that overrides the lookup.
        hint: Install advice appended to the error message.

    Returns:
        The path of the tool.

    Raises:
        ProfilerError: If the tool is neither set nor on ``PATH``.
    """
    found = os.environ.get(env_var) or shutil.which(executable)
    if not found:
        raise ProfilerError(
            f"'{executable}' not found on PATH (or set {env_var}). {hint}"
        )
    return found


def make_bundle_dir(out_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Creates ``out_dir/name`` if needed and returns it."""
    bundle_dir = out_dir / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def default_name(command: Sequence[str]) -> str:
    """Returns a readable bundle name for the profiled command.

    Args:
        command: Command line to profile.

    Returns:
        The stem of the first ``.py`` argument, else the executable's file
        name, else ``profile`` for an empty command.
    """
    for token in command:
        if token.endswith(".py"):
            return pathlib.Path(token).stem
    return pathlib.Path(command[0]).name if command else "profile"


def run_echoed(argv: list[str]) -> int:
    """Echoes ``argv`` to stderr, runs it and returns its exit status."""
    print("$ " + shlex.join(argv), file=sys.stderr)
    return subprocess.run(argv, check=False).returncode


def execute_target(target: Sequence[str]) -> int:
    """Executes a target module or script file, returning the process exit code.

    Args:
        target: Target command line (e.g. ``["-m", "mymod", ...]`` or
            ``["script.py", ...]``).

    Returns:
        The exit code (0 on success, 1 or SystemExit code on exit/failure).
    """
    import runpy

    try:
        if target[0] == "-m":
            sys.argv = [target[1], *target[2:]]
            runpy.run_module(target[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = list(target)
            runpy.run_path(target[0], run_name="__main__")
        return 0
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(
            f"cta profile: target raised {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
