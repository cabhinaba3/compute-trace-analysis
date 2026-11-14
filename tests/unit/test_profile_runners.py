"""Tests for `cta profile`: profiler command lines, runners and outputs.

The profilers (nsys, ncu, trtexec and jtop) are replaced by a fake executable
or a fake module, so the tests check the commands that are run and the files
that are kept without a GPU. The last test checks that the Kineto adapter
counts a step once although PyTorch records it on both the CPU and the GPU.
"""

from __future__ import annotations

import datetime
import gzip
import json
import pathlib
import stat
import sys
import types
from typing import Any

import pytest
from typer import testing

import cta
from cta.cli import app as cli_app
from cta.profile import runners

cli_runner = testing.CliRunner()

# A stand-in for nsys, ncu and trtexec. It appends its arguments to
# `<tool>.calls` next to itself and writes the files the real tool would.
FAKE_TOOL_SCRIPT = """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
tool = pathlib.Path(__file__).name
with pathlib.Path(__file__).with_suffix(".calls").open("a") as fh:
    fh.write(" ".join(args) + "\\n")
if "-o" in args:
    out = pathlib.Path(args[args.index("-o") + 1])
    if tool == "ncu":
        suffix = ".ncu-rep"
    elif args[0] == "export":
        suffix = ".sqlite"
    else:
        suffix = ".nsys-rep"
    if not out.name.endswith(suffix):
        out = out.with_name(out.name + suffix)
    out.write_bytes(b"x")
export_flags = ("--exportTimes=", "--exportProfile=", "--exportLayerInfo=")
for flag in args:
    if flag.startswith(export_flags):
        pathlib.Path(flag.split("=", 1)[1]).write_text("{}")
"""


def _install_fake_tool(bin_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Writes the fake profiler script as an executable file.

    Args:
        bin_dir: Directory to put the executable in.
        name: File name of the tool, such as "nsys".

    Returns:
        Path of the executable.
    """
    tool_path = bin_dir / name
    tool_path.write_text(FAKE_TOOL_SCRIPT)
    tool_path.chmod(tool_path.stat().st_mode | stat.S_IEXEC)
    return tool_path


def test_default_name() -> None:
    assert runners.default_name(["python", "train.py", "--x"]) == "train"
    assert runners.default_name(["./bin/app"]) == "app"


def test_command_construction() -> None:
    output_path = pathlib.Path("out/x")
    nsys_argv = runners.nsys_command(
        ["python", "a.py"], output_path, trace="cuda,nvtx", gpu_metrics=True
    )
    assert nsys_argv[:2] == ["nsys", "profile"]
    assert "--gpu-metrics-devices=all" in nsys_argv
    assert nsys_argv[-2:] == ["python", "a.py"]
    ncu_argv = runners.ncu_command(
        ["python", "a.py"],
        output_path,
        ncu_set="full",
        metrics=(),
        launch_count=24,
        kernel_regex="gemm",
        cache_control="all",
        clock_control="none",
    )
    assert ncu_argv[ncu_argv.index("--set") + 1] == "full"
    assert ncu_argv[ncu_argv.index("--launch-count") + 1] == "24"
    assert "regex:gemm" in ncu_argv
    ncu_metrics_argv = runners.ncu_command(
        ["a"],
        output_path,
        ncu_set="full",
        metrics=("m1", "m2"),
        launch_count=None,
        kernel_regex=None,
        cache_control="none",
        clock_control="base",
    )
    assert "--set" not in ncu_metrics_argv
    metrics_index = ncu_metrics_argv.index("--metrics")
    assert ncu_metrics_argv[metrics_index + 1] == "m1,m2"
    trtexec_argv = runners.trtexec_command(
        ["--loadEngine=m.plan"], pathlib.Path("b")
    )
    assert any(arg.startswith("--exportTimes=") for arg in trtexec_argv)
    assert "--profilingVerbosity=detailed" in trtexec_argv
    jax_argv = runners.jax_command(
        ["python", "train.py", "--lr", "1"], pathlib.Path("logdir")
    )
    assert jax_argv[1:3] == ["-m", "cta.profile.jax_runner"]
    assert jax_argv[-3:] == ["train.py", "--lr", "1"]
    assert "--no-perfetto" not in jax_argv
    assert "--no-perfetto" in runners.jax_command(
        ["train.py"], pathlib.Path("logdir"), perfetto=False
    )
    kineto_argv = runners.kineto_command(
        ["python", "train.py", "--lr", "1"],
        pathlib.Path("t.json.gz"),
        step_on="optimizer",
    )
    assert kineto_argv[1:3] == ["-m", "cta.profile.kineto_runner"]
    assert kineto_argv[-3:] == ["train.py", "--lr", "1"]
    assert "python" not in kineto_argv[-4:]


def test_nsys_cli_writes_report_and_sqlite(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_fake_tool(bin_dir, "nsys")
    monkeypatch.setenv("NSYS_PATH", str(bin_dir / "nsys"))
    result = cli_runner.invoke(
        cli_app.app,
        [
            "profile",
            "nsys",
            "-o",
            str(tmp_path / "p"),
            "--name",
            "run1",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "p" / "run1"
    assert (run_dir / "run1.nsys-rep").exists()
    assert (run_dir / "run1.sqlite").exists()
    calls = (bin_dir / "nsys.calls").read_text().splitlines()
    assert calls[0].startswith("profile -t cuda,nvtx,osrt")
    assert calls[1].startswith("export -t sqlite")


def test_ncu_and_trtexec_cli(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("NCU_PATH", str(_install_fake_tool(bin_dir, "ncu")))
    monkeypatch.setenv(
        "TRTEXEC_PATH", str(_install_fake_tool(bin_dir, "trtexec"))
    )
    ncu_result = cli_runner.invoke(
        cli_app.app,
        [
            "profile",
            "ncu",
            "-o",
            str(tmp_path / "p"),
            "--launch-count",
            "3",
            "--",
            "python",
            "a.py",
        ],
    )
    assert ncu_result.exit_code == 0, ncu_result.output
    assert (tmp_path / "p" / "a" / "a.ncu-rep").exists()
    trtexec_result = cli_runner.invoke(
        cli_app.app,
        [
            "profile",
            "trtexec",
            "-o",
            str(tmp_path / "p"),
            "--",
            "--loadEngine=m.plan",
        ],
    )
    assert trtexec_result.exit_code == 0, trtexec_result.output
    trtexec_dir = tmp_path / "p" / "trtexec"
    assert {path.name for path in trtexec_dir.iterdir()} == {
        "times.json",
        "profile.json",
        "layers.json",
    }


def test_missing_tool_is_a_clean_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NSYS_PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = cli_runner.invoke(
        cli_app.app,
        ["profile", "nsys", "-o", str(tmp_path), "--", "echo", "hi"],
    )
    assert result.exit_code == 2
    assert "not found" in result.output
    # No command to profile is an error too.
    without_command = cli_runner.invoke(cli_app.app, ["profile", "nsys"])
    assert without_command.exit_code != 0


def test_failing_profiled_command_propagates_status(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_trtexec = _install_fake_tool(bin_dir, "trtexec")
    fake_trtexec.write_text(FAKE_TOOL_SCRIPT + "\nsys.exit(7)\n")
    monkeypatch.setenv("TRTEXEC_PATH", str(fake_trtexec))
    result = cli_runner.invoke(
        cli_app.app,
        ["profile", "trtexec", "-o", str(tmp_path / "p"), "--", "--x"],
    )
    # The artifacts are still written and the exit status is passed on.
    assert result.exit_code == 7


def test_jtop_sampler_writes_json_list(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeJtop:
        """Stand-in for `jtop.jtop` that always reports the same sample."""

        def __init__(self, interval: float = 0.5) -> None:
            self.stats = {
                "time": datetime.datetime(2026, 1, 1),
                "GPU": 50.0,
                "Power TOT": 9000,
            }
            self.calls = 0

        def __enter__(self) -> FakeJtop:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def ok(self) -> bool:
            """Counts the call and reports that the service still runs."""
            self.calls += 1
            return True

    fake_jtop_module = types.ModuleType("jtop")
    fake_jtop_module.jtop = FakeJtop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jtop", fake_jtop_module)
    result = runners.run_jtop(
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        tmp_path,
        "j",
        interval=0.01,
    )
    assert result.returncode == 0
    assert result.artifacts
    samples = json.loads(result.artifacts[0].read_text())
    assert samples
    assert samples[0]["GPU"] == 50.0
    assert samples[0]["time"].startswith("2026-01-01")


def test_kineto_gpu_step_annotations_are_not_extra_steps(
    tmp_path: pathlib.Path,
) -> None:
    """Counts each ProfilerStep#N once, though PyTorch emits it twice.

    The CPU copy has category `user_annotation`, the GPU copy
    `gpu_user_annotation`.
    """
    events: list[dict[str, Any]] = []
    for i in range(3):
        step_start_us = 1000.0 + i * 500
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": f"ProfilerStep#{i}",
                "ts": step_start_us,
                "dur": 400.0,
                "pid": 1,
                "tid": 1,
            }
        )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "ts": step_start_us + 10,
                "dur": 100.0,
                "pid": 0,
                "tid": 7,
                "args": {"stream": 7, "correlation": i},
            }
        )
    events.append(
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": "ProfilerStep#0",
            "ts": 1010.0,
            "dur": 100.0,
            "pid": 0,
            "tid": 7,
        }
    )
    trace_path = tmp_path / "t.json.gz"
    with gzip.open(trace_path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)
    report = cta.analyze(trace_path, label="t", cache_dir=tmp_path / "c")
    latency = report.to_polars("latency_throughput")
    rows_by_metric = {
        row["metric"]: row for row in latency.iter_rows(named=True)
    }
    assert rows_by_metric["total_steps"]["t"] == 3.0
