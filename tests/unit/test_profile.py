from __future__ import annotations

import stat
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cta.cli.app import app
from cta.profile import runners

runner = CliRunner()

FAKE_TOOL = """#!/usr/bin/env python3
import pathlib, sys
a = sys.argv[1:]
log = pathlib.Path(__file__).with_suffix(".calls")
with log.open("a") as fh:
    fh.write(" ".join(a) + "\\n")
if "-o" in a:
    out = pathlib.Path(a[a.index("-o") + 1])
    suffix = ".ncu-rep" if pathlib.Path(__file__).name == "ncu" else (".sqlite" if a[0] == "export" else ".nsys-rep")
    out.with_name(out.name + suffix if not out.name.endswith(suffix) else out.name).write_bytes(b"x")
for flag in a:
    if flag.startswith(("--exportTimes=", "--exportProfile=", "--exportLayerInfo=")):
        pathlib.Path(flag.split("=", 1)[1]).write_text("{}")
"""


def _fake(bin_dir: Path, name: str) -> Path:
    p = bin_dir / name
    p.write_text(FAKE_TOOL)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_default_name() -> None:
    assert runners.default_name(["python", "train.py", "--x"]) == "train"
    assert runners.default_name(["./bin/app"]) == "app"


def test_command_construction() -> None:
    p = Path("out/x")
    nsys = runners.nsys_command(["python", "a.py"], p, trace="cuda,nvtx", gpu_metrics=True)
    assert nsys[:2] == ["nsys", "profile"] and "--gpu-metrics-devices=all" in nsys and nsys[-2:] == ["python", "a.py"]
    ncu = runners.ncu_command(["python", "a.py"], p, ncu_set="full", metrics=(), launch_count=24, kernel_regex="gemm", cache_control="all", clock_control="none")
    assert ncu[ncu.index("--set") + 1] == "full" and ncu[ncu.index("--launch-count") + 1] == "24" and "regex:gemm" in ncu
    ncu_m = runners.ncu_command(["a"], p, ncu_set="full", metrics=("m1", "m2"), launch_count=None, kernel_regex=None, cache_control="none", clock_control="base")
    assert "--set" not in ncu_m and ncu_m[ncu_m.index("--metrics") + 1] == "m1,m2"
    trt = runners.trtexec_command(["--loadEngine=m.plan"], Path("b"))
    assert any(a.startswith("--exportTimes=") for a in trt) and "--profilingVerbosity=detailed" in trt
    k = runners.kineto_command(["python", "train.py", "--lr", "1"], Path("t.json.gz"), step_on="optimizer")
    assert k[1:3] == ["-m", "cta.profile.kineto_runner"] and k[-3:] == ["train.py", "--lr", "1"] and "python" not in k[-4:]


def test_nsys_cli_writes_report_and_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake(bindir, "nsys")
    monkeypatch.setenv("NSYS_PATH", str(bindir / "nsys"))
    res = runner.invoke(app, ["profile", "nsys", "-o", str(tmp_path / "p"), "--name", "run1", "--", "echo", "hi"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "p" / "run1" / "run1.nsys-rep").exists() and (tmp_path / "p" / "run1" / "run1.sqlite").exists()
    calls = (bindir / "nsys.calls").read_text().splitlines()
    assert calls[0].startswith("profile -t cuda,nvtx,osrt") and calls[1].startswith("export -t sqlite")


def test_ncu_and_trtexec_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    monkeypatch.setenv("NCU_PATH", str(_fake(bindir, "ncu")))
    monkeypatch.setenv("TRTEXEC_PATH", str(_fake(bindir, "trtexec")))
    r1 = runner.invoke(app, ["profile", "ncu", "-o", str(tmp_path / "p"), "--launch-count", "3", "--", "python", "a.py"])
    assert r1.exit_code == 0, r1.output
    assert (tmp_path / "p" / "a" / "a.ncu-rep").exists()
    r2 = runner.invoke(app, ["profile", "trtexec", "-o", str(tmp_path / "p"), "--", "--loadEngine=m.plan"])
    assert r2.exit_code == 0, r2.output
    assert {f.name for f in (tmp_path / "p" / "trtexec").iterdir()} == {"times.json", "profile.json", "layers.json"}


def test_missing_tool_is_a_clean_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NSYS_PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    res = runner.invoke(app, ["profile", "nsys", "-o", str(tmp_path), "--", "echo", "hi"])
    assert res.exit_code == 2 and "not found" in res.output
    assert runner.invoke(app, ["profile", "nsys"]).exit_code != 0  # no command given


def test_failing_profiled_command_propagates_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = _fake(bindir, "trtexec")
    fake.write_text(FAKE_TOOL + "\nsys.exit(7)\n")
    monkeypatch.setenv("TRTEXEC_PATH", str(fake))
    res = runner.invoke(app, ["profile", "trtexec", "-o", str(tmp_path / "p"), "--", "--x"])
    assert res.exit_code == 7  # artifacts were still written, the status is passed on


def test_jtop_sampler_writes_json_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeJtop:
        def __init__(self, interval: float = 0.5) -> None:
            self.stats = {"time": __import__("datetime").datetime(2026, 1, 1), "GPU": 50.0, "Power TOT": 9000}
            self.calls = 0

        def __enter__(self) -> FakeJtop:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def ok(self) -> bool:
            self.calls += 1
            return True

    mod = types.ModuleType("jtop")
    mod.jtop = FakeJtop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jtop", mod)
    res = runners.run_jtop([sys.executable, "-c", "import time; time.sleep(0.2)"], tmp_path, "j", interval=0.01)
    assert res.returncode == 0 and res.artifacts
    import json

    data = json.loads(res.artifacts[0].read_text())
    assert data and data[0]["GPU"] == 50.0 and data[0]["time"].startswith("2026-01-01")


def test_kineto_gpu_step_annotations_are_not_extra_steps(tmp_path: Path) -> None:
    """PyTorch emits ProfilerStep#N twice (cpu `user_annotation` + `gpu_user_annotation`); count it once."""
    import gzip
    import json

    import cta

    ev = []
    for i in range(3):
        t = 1000.0 + i * 500
        ev.append({"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}", "ts": t, "dur": 400.0, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "kernel", "name": "gemm_kernel", "ts": t + 10, "dur": 100.0, "pid": 0, "tid": 7, "args": {"stream": 7, "correlation": i}})
    ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": "ProfilerStep#0", "ts": 1010.0, "dur": 100.0, "pid": 0, "tid": 7})
    trace = tmp_path / "t.json.gz"
    with gzip.open(trace, "wt") as fh:
        json.dump({"traceEvents": ev}, fh)
    rep = cta.analyze(trace, label="t", cache_dir=tmp_path / "c")
    rows = {r["metric"]: r for r in rep.to_polars("latency_throughput").iter_rows(named=True)}
    assert rows["total_steps"]["t"] == 3.0
