"""Thin wrappers that run a command under nsys, ncu, torch.profiler, trtexec or a jtop sampler.

Every runner writes into ``<out_dir>/<name>/`` so the directory is a ready-made RunBundle for ``cta analyze``. The
heavy lifting is done by the real tools; nothing here is imported until a profile command actually runs.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfilerError(RuntimeError):
    """The profiler is missing or failed."""


@dataclass(slots=True)
class ProfileResult:
    tool: str
    bundle_dir: Path
    command: list[str]
    returncode: int
    artifacts: list[Path] = field(default_factory=list)


def _find(exe: str, env: str, hint: str) -> str:
    found = os.environ.get(env) or shutil.which(exe)
    if not found:
        raise ProfilerError(f"'{exe}' not found on PATH (or set {env}). {hint}")
    return found


def _bundle(out_dir: Path, name: str) -> Path:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_name(command: Sequence[str]) -> str:
    """A readable bundle name from the profiled command (script stem, else the executable)."""
    for tok in command:
        if tok.endswith(".py"):
            return Path(tok).stem
    return Path(command[0]).name if command else "profile"


def _run(argv: list[str]) -> int:
    print("$ " + shlex.join(argv), file=sys.stderr)
    return subprocess.run(argv, check=False).returncode


def nsys_command(command: Sequence[str], out_prefix: Path, *, trace: str, gpu_metrics: bool, nsys: str = "nsys") -> list[str]:
    argv = [nsys, "profile", "-t", trace, "-s", "none", "--force-overwrite", "true", "-o", str(out_prefix)]
    if gpu_metrics:
        argv.append("--gpu-metrics-devices=all")
    return [*argv, *command]


def run_nsys(
    command: Sequence[str], out_dir: Path, name: str, *, trace: str = "cuda,nvtx,osrt", gpu_metrics: bool = False, export: bool = True
) -> ProfileResult:
    """Nsight Systems: ``nsys profile`` then ``nsys export -t sqlite`` (the form CTA reads on any machine)."""
    nsys = _find("nsys", "NSYS_PATH", "Install Nsight Systems (conda-forge 'nsight-systems' or the CUDA toolkit).")
    bdir = _bundle(out_dir, name)
    prefix = bdir / name
    argv = nsys_command(command, prefix, trace=trace, gpu_metrics=gpu_metrics, nsys=nsys)
    rc = _run(argv)
    arts = [p for p in (prefix.with_suffix(".nsys-rep"),) if p.exists()]
    if export and arts:
        sqlite = prefix.with_suffix(".sqlite")
        if _run([nsys, "export", "-t", "sqlite", "--force-overwrite", "true", "-o", str(sqlite), str(arts[0])]) == 0:
            arts.append(sqlite)
    return ProfileResult("nsys", bdir, argv, rc, arts)


def ncu_command(
    command: Sequence[str],
    report: Path,
    *,
    ncu_set: str,
    metrics: Sequence[str],
    launch_count: int | None,
    kernel_regex: str | None,
    cache_control: str,
    clock_control: str,
    ncu: str = "ncu",
) -> list[str]:
    # cache/clock defaults follow nsight-python's choices (flush caches between replays, do not lock clocks).
    argv = [ncu, "--cache-control", cache_control, "--clock-control", clock_control, "-f", "-o", str(report)]
    argv += ["--metrics", ",".join(metrics)] if metrics else ["--set", ncu_set]
    if launch_count is not None:
        argv += ["--launch-count", str(launch_count)]
    if kernel_regex:
        argv += ["-k", f"regex:{kernel_regex}"]
    return [*argv, *command]


def run_ncu(
    command: Sequence[str],
    out_dir: Path,
    name: str,
    *,
    ncu_set: str = "full",
    metrics: Sequence[str] = (),
    launch_count: int | None = None,
    kernel_regex: str | None = None,
    cache_control: str = "all",
    clock_control: str = "none",
) -> ProfileResult:
    """Nsight Compute: needs hardware-counter access (see ``ERR_NVGPUCTRPERM``)."""
    ncu = _find("ncu", "NCU_PATH", "Install Nsight Compute (CUDA toolkit or conda-forge 'nsight-compute').")
    bdir = _bundle(out_dir, name)
    report = bdir / name
    argv = ncu_command(
        command, report, ncu_set=ncu_set, metrics=metrics, launch_count=launch_count, kernel_regex=kernel_regex,
        cache_control=cache_control, clock_control=clock_control, ncu=ncu,
    )  # fmt: skip
    rc = _run(argv)
    return ProfileResult("ncu", bdir, argv, rc, [p for p in (bdir / f"{name}.ncu-rep",) if p.exists()])


def kineto_command(command: Sequence[str], trace: Path, *, step_on: str) -> list[str]:
    """Run the script under ``torch.profiler`` through cta.profile.kineto_runner (same interpreter)."""
    toks = list(command)
    if toks and Path(toks[0]).name.startswith("python"):
        toks = toks[1:]
    if not toks:
        raise ProfilerError("kineto needs a Python script or '-m module' to run, e.g. -- python train.py")
    return [sys.executable, "-m", "cta.profile.kineto_runner", "--out", str(trace), "--step-on", step_on, "--", *toks]


def run_kineto(command: Sequence[str], out_dir: Path, name: str, *, step_on: str = "optimizer") -> ProfileResult:
    bdir = _bundle(out_dir, name)
    trace = bdir / f"{name}.json.gz"
    argv = kineto_command(command, trace, step_on=step_on)
    rc = _run(argv)
    return ProfileResult("kineto", bdir, argv, rc, [trace] if trace.exists() else [])


def jax_command(command: Sequence[str], logdir: Path, *, perfetto: bool = True) -> list[str]:
    """Run the script under ``jax.profiler.trace`` through cta.profile.jax_runner (same interpreter)."""
    toks = list(command)
    if toks and Path(toks[0]).name.startswith("python"):
        toks = toks[1:]
    if not toks:
        raise ProfilerError("jax needs a Python script or '-m module' to run, e.g. -- python train.py")
    flag = [] if perfetto else ["--no-perfetto"]
    return [sys.executable, "-m", "cta.profile.jax_runner", "--out", str(logdir), *flag, "--", *toks]


def run_jax(command: Sequence[str], out_dir: Path, name: str, *, perfetto: bool = True) -> ProfileResult:
    bdir = _bundle(out_dir, name)
    argv = jax_command(command, bdir, perfetto=perfetto)
    rc = _run(argv)
    arts = sorted(bdir.glob("plugins/profile/*/*"))
    return ProfileResult("jax", bdir, argv, rc, arts)


def trtexec_command(args: Sequence[str], bundle: Path, *, trtexec: str = "trtexec") -> list[str]:
    exports = [
        f"--exportTimes={bundle / 'times.json'}",
        f"--exportProfile={bundle / 'profile.json'}",
        f"--exportLayerInfo={bundle / 'layers.json'}",
        "--profilingVerbosity=detailed",
        "--dumpProfile",
    ]
    return [trtexec, *args, *exports]


def run_trtexec(args: Sequence[str], out_dir: Path, name: str) -> ProfileResult:
    """TensorRT: pass trtexec's own arguments (--onnx/--loadEngine, --fp16, ...); the three export files are added."""
    trtexec = _find("trtexec", "TRTEXEC_PATH", "Install TensorRT (on Jetson: /usr/src/tensorrt/bin/trtexec).")
    bdir = _bundle(out_dir, name)
    argv = trtexec_command(args, bdir, trtexec=trtexec)
    rc = _run(argv)
    return ProfileResult("trtexec", bdir, argv, rc, [p for p in (bdir / "times.json", bdir / "profile.json", bdir / "layers.json") if p.exists()])


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def run_jtop(command: Sequence[str], out_dir: Path, name: str, *, interval: float = 0.5) -> ProfileResult:
    """Jetson: sample ``jtop`` while the command runs and write the JSON list CTA's jtop adapter reads."""
    try:
        from jtop import jtop  # Jetson-only: sudo pip install jetson-stats
    except ImportError as exc:
        raise ProfilerError("the 'jtop' module is required (Jetson only): sudo pip install jetson-stats") from exc
    bdir = _bundle(out_dir, name)
    out = bdir / f"{name}_jtop.json"
    print("$ " + shlex.join(command), file=sys.stderr)
    samples: list[dict[str, Any]] = []
    with jtop(interval=interval) as jetson:
        proc = subprocess.Popen(list(command))
        while proc.poll() is None and jetson.ok():
            samples.append({k: _jsonable(v) for k, v in dict(jetson.stats).items()})
        rc = proc.wait()
    out.write_text(json.dumps(samples, indent=1))
    return ProfileResult("jtop", bdir, list(command), rc, [out] if samples else [])

