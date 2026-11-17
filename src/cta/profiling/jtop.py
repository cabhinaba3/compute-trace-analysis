"""Samples Jetson power and utilisation with jtop while a command runs."""

from __future__ import annotations

from collections.abc import Sequence
import json
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any

from cta.profiling import base


def _jsonable(value: Any) -> Any:
    """Returns ``value`` if JSON can hold it, else an ISO date or ``str``."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def run(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    interval: float = 0.5,
) -> base.ProfileResult:
    """Samples Jetson telemetry with ``jtop`` while a command runs.

    Writes the JSON list of samples that CTA's jtop adapter reads. If
    sampling raises (e.g. on Ctrl-C), the command is terminated, killed if
    it is still running after 5 s, and the exception is re-raised.

    Args:
        command: Command line to run.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory name and samples file prefix.
        interval: Seconds between samples.

    Returns:
        The run result; the samples file is listed only if any sample was
        taken.

    Raises:
        base.ProfilerError: If the ``jtop`` module (jetson-stats) is missing.
    """
    try:
        import jtop as jtop_module  # Jetson-only: sudo pip install jetson-stats
    except ImportError as exc:
        raise base.ProfilerError(
            "the 'jtop' module is required (Jetson only): sudo pip install "
            "jetson-stats"
        ) from exc
    bundle_dir = base.make_bundle_dir(out_dir, name)
    samples_path = bundle_dir / f"{name}_jtop.json"
    print("$ " + shlex.join(command), file=sys.stderr)
    samples: list[dict[str, Any]] = []
    with jtop_module.jtop(interval=interval) as jetson:
        proc = subprocess.Popen(list(command))
        try:
            while proc.poll() is None and jetson.ok():
                samples.append(
                    {
                        key: _jsonable(value)
                        for key, value in dict(jetson.stats).items()
                    }
                )
                time.sleep(interval)
            returncode = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    samples_path.write_text(json.dumps(samples, indent=1))
    return base.ProfileResult(
        "jtop",
        bundle_dir,
        list(command),
        returncode,
        [samples_path] if samples else [],
    )
