"""Thin wrappers that run a command under a vendor profiler or sampler.

The supported tools are nsys, ncu, torch.profiler, jax.profiler, trtexec and
a jtop sampler. Every runner writes into ``<out_dir>/<name>/`` so the
directory is a ready-made RunBundle for ``cta analyze``. The heavy lifting is
done by the real tools; nothing here is imported until a profile command
actually runs.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


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


def _find_tool(executable: str, env_var: str, hint: str) -> str:
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


def _make_bundle_dir(out_dir: pathlib.Path, name: str) -> pathlib.Path:
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


def _run(argv: list[str]) -> int:
    """Echoes ``argv`` to stderr, runs it and returns its exit status."""
    print("$ " + shlex.join(argv), file=sys.stderr)
    return subprocess.run(argv, check=False).returncode


def nsys_command(
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


def run_nsys(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    trace: str = "cuda,nvtx,osrt",
    gpu_metrics: bool = False,
    export: bool = True,
) -> ProfileResult:
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
        ProfilerError: If nsys is not found.
    """
    nsys = _find_tool(
        "nsys",
        "NSYS_PATH",
        "Install Nsight Systems (conda-forge 'nsight-systems' or the CUDA "
        "toolkit).",
    )
    bundle_dir = _make_bundle_dir(out_dir, name)
    report_prefix = bundle_dir / name
    argv = nsys_command(
        command, report_prefix, trace=trace, gpu_metrics=gpu_metrics, nsys=nsys
    )
    returncode = _run(argv)
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
        if _run(export_argv) == 0:
            artifacts.append(sqlite_path)
    return ProfileResult("nsys", bundle_dir, argv, returncode, artifacts)


def ncu_command(
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


def run_ncu(
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
) -> ProfileResult:
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
        ProfilerError: If ncu is not found.
    """
    ncu = _find_tool(
        "ncu",
        "NCU_PATH",
        "Install Nsight Compute (CUDA toolkit or conda-forge "
        "'nsight-compute').",
    )
    bundle_dir = _make_bundle_dir(out_dir, name)
    report = bundle_dir / name
    argv = ncu_command(
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
    returncode = _run(argv)
    artifacts = [
        path for path in (bundle_dir / f"{name}.ncu-rep",) if path.exists()
    ]
    return ProfileResult("ncu", bundle_dir, argv, returncode, artifacts)


def kineto_command(
    command: Sequence[str], trace_path: pathlib.Path, *, step_on: str
) -> list[str]:
    """Builds the command that runs a script under ``torch.profiler``.

    The script runs through ``cta.profile.kineto_runner`` in the current
    interpreter; a leading ``python`` executable is dropped.

    Args:
        command: Python command line to profile.
        trace_path: Trace file to write.
        step_on: ``optimizer`` to start a profiler step at each
            ``optimizer.step()``, or ``none``.

    Returns:
        The full command line.

    Raises:
        ProfilerError: If nothing is left to run after the interpreter.
    """
    tokens = list(command)
    if tokens and pathlib.Path(tokens[0]).name.startswith("python"):
        tokens = tokens[1:]
    if not tokens:
        raise ProfilerError(
            "kineto needs a Python script or '-m module' to run, e.g. -- "
            "python train.py"
        )
    return [
        sys.executable,
        "-m",
        "cta.profile.kineto_runner",
        "--out",
        str(trace_path),
        "--step-on",
        step_on,
        "--",
        *tokens,
    ]


def run_kineto(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    step_on: str = "optimizer",
) -> ProfileResult:
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
        ProfilerError: If ``command`` names no script or module.
    """
    bundle_dir = _make_bundle_dir(out_dir, name)
    trace = bundle_dir / f"{name}.json.gz"
    argv = kineto_command(command, trace, step_on=step_on)
    returncode = _run(argv)
    return ProfileResult(
        "kineto",
        bundle_dir,
        argv,
        returncode,
        [trace] if trace.exists() else [],
    )


def jax_command(
    command: Sequence[str], log_dir: pathlib.Path, *, perfetto: bool = True
) -> list[str]:
    """Builds the command that runs a script under ``jax.profiler.trace``.

    The script runs through ``cta.profile.jax_runner`` in the current
    interpreter; a leading ``python`` executable is dropped.

    Args:
        command: Python command line to profile.
        log_dir: XProf log directory to write.
        perfetto: Whether to also write a Chrome-format trace.

    Returns:
        The full command line.

    Raises:
        ProfilerError: If nothing is left to run after the interpreter.
    """
    tokens = list(command)
    if tokens and pathlib.Path(tokens[0]).name.startswith("python"):
        tokens = tokens[1:]
    if not tokens:
        raise ProfilerError(
            "jax needs a Python script or '-m module' to run, e.g. -- "
            "python train.py"
        )
    perfetto_flags = [] if perfetto else ["--no-perfetto"]
    return [
        sys.executable,
        "-m",
        "cta.profile.jax_runner",
        "--out",
        str(log_dir),
        *perfetto_flags,
        "--",
        *tokens,
    ]


def run_jax(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    perfetto: bool = True,
) -> ProfileResult:
    """Profiles a Python command with the JAX profiler (XProf).

    Args:
        command: Python command line to profile.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory name; it is also the log directory.
        perfetto: Whether to also write a Chrome-format trace.

    Returns:
        The run result, listing the files under ``plugins/profile/``.

    Raises:
        ProfilerError: If ``command`` names no script or module.
    """
    bundle_dir = _make_bundle_dir(out_dir, name)
    argv = jax_command(command, bundle_dir, perfetto=perfetto)
    returncode = _run(argv)
    artifacts = sorted(bundle_dir.glob("plugins/profile/*/*"))
    return ProfileResult("jax", bundle_dir, argv, returncode, artifacts)


def trtexec_command(
    args: Sequence[str], bundle_dir: pathlib.Path, *, trtexec: str = "trtexec"
) -> list[str]:
    """Builds the ``trtexec`` command line with CTA's export options.

    Args:
        args: trtexec's own arguments, e.g. ``--onnx=model.onnx``.
        bundle_dir: Directory that receives ``times.json``, ``profile.json`` and
            ``layers.json``.
        trtexec: Path of the trtexec executable.

    Returns:
        The full command line.
    """
    exports = [
        f"--exportTimes={bundle_dir / 'times.json'}",
        f"--exportProfile={bundle_dir / 'profile.json'}",
        f"--exportLayerInfo={bundle_dir / 'layers.json'}",
        "--profilingVerbosity=detailed",
        "--dumpProfile",
    ]
    return [trtexec, *args, *exports]


def run_trtexec(
    args: Sequence[str], out_dir: pathlib.Path, name: str
) -> ProfileResult:
    """Runs TensorRT's trtexec and keeps its timing, profile and layer files.

    Args:
        args: trtexec's own arguments (``--onnx``/``--loadEngine``,
            ``--fp16``, ...); the three export options are added.
        out_dir: Directory that receives the bundle directory.
        name: Bundle directory name.

    Returns:
        The run result.

    Raises:
        ProfilerError: If trtexec is not found.
    """
    trtexec = _find_tool(
        "trtexec",
        "TRTEXEC_PATH",
        "Install TensorRT (on Jetson: /usr/src/tensorrt/bin/trtexec).",
    )
    bundle_dir = _make_bundle_dir(out_dir, name)
    argv = trtexec_command(args, bundle_dir, trtexec=trtexec)
    returncode = _run(argv)
    export_paths = (
        bundle_dir / "times.json",
        bundle_dir / "profile.json",
        bundle_dir / "layers.json",
    )
    artifacts = [path for path in export_paths if path.exists()]
    return ProfileResult("trtexec", bundle_dir, argv, returncode, artifacts)


def _jsonable(value: Any) -> Any:
    """Returns ``value`` if JSON can hold it, else an ISO date or ``str``."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def run_jtop(
    command: Sequence[str],
    out_dir: pathlib.Path,
    name: str,
    *,
    interval: float = 0.5,
) -> ProfileResult:
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
        ProfilerError: If the ``jtop`` module (jetson-stats) is missing.
    """
    try:
        import jtop as jtop_module  # Jetson-only: sudo pip install jetson-stats
    except ImportError as exc:
        raise ProfilerError(
            "the 'jtop' module is required (Jetson only): sudo pip install "
            "jetson-stats"
        ) from exc
    bundle_dir = _make_bundle_dir(out_dir, name)
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
    return ProfileResult(
        "jtop",
        bundle_dir,
        list(command),
        returncode,
        [samples_path] if samples else [],
    )
