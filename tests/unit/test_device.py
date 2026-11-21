"""Device selection: `accelerator` is auto, cpu, cuda or metal.

The CPU is always the fallback.

Asking for a device that is not there is not an error: the analysis runs on
the CPU and says so once. Only a name that is not a device at all is refused.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest
from typer import testing

from cta import _device
from cta import models
from cta.cli import app as cli_app
from cta.ingest import manifest


@pytest.fixture(autouse=True)
def _fresh_warnings() -> None:
    _device.reset_warnings()


def _no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_device, "backends", lambda: [])


def _with_gpu(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(_device, "backends", lambda: list(names))


def test_cpu_is_always_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_gpu(monkeypatch, "cuda")

    device = _device.resolve("cpu")

    assert (device.name, device.requested, device.note) == ("cpu", "cpu", "")


def test_a_missing_device_falls_back_to_the_cpu_with_one_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_gpu(monkeypatch)

    with pytest.warns(UserWarning, match="cuda is not available"):
        first = _device.resolve("cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the second time says nothing
        second = _device.resolve("cuda")

    assert first.name == second.name == "cpu"
    assert first.requested == "cuda"
    assert "not available" in first.note


def test_a_present_device_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_gpu(monkeypatch, "metal")

    device = _device.resolve("metal")

    assert device.name == "metal"
    assert device.note == ""


def test_auto_picks_a_gpu_only_for_enough_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_gpu(monkeypatch, "cuda")

    small = _device.resolve("auto", work=10)
    large = _device.resolve("auto", work=10**9)
    unknown = _device.resolve("auto")

    assert small.name == "cpu"
    assert large.name == "cuda"
    assert unknown.name == "cpu"  # no size given: do not pay for a transfer


def test_auto_does_not_pick_metal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metal is fp32 and no faster than the CPU on an M1: only on request."""
    _with_gpu(monkeypatch, "metal")

    assert _device.resolve("auto", work=10**10).name == "cpu"
    assert _device.resolve("metal").name == "metal"


def test_auto_without_a_gpu_is_the_cpu_and_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_gpu(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        device = _device.resolve("auto", work=10**9)

    assert device.name == "cpu"


def test_a_name_that_is_not_a_device_is_refused() -> None:
    with pytest.raises(ValueError, match="auto, cpu, cuda or metal"):
        _device.resolve("tpu")


def test_the_current_device_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_gpu(monkeypatch, "cuda")
    assert _device.requested() == "auto"

    with _device.use("cuda"):
        assert _device.requested() == "cuda"
        with _device.use("cpu"):
            assert _device.requested() == "cpu"
        assert _device.requested() == "cuda"

    assert _device.requested() == "auto"


def test_the_config_default_and_the_manifest_key(
    tmp_path: pathlib.Path,
) -> None:
    assert models.AnalysisConfig().accelerator == "auto"

    trace = tmp_path / "t.goal"
    trace.write_text("num_ranks 1\nrank 0 {\nl1: calc 1\n}\n")
    path = tmp_path / "m.yaml"
    path.write_text(
        "groups:\n"
        f"  - {{id: a, path: {trace.name}}}\n"
        "config:\n  accelerator: cpu\n"
    )

    _, config = manifest.load_manifest(path)

    assert config.accelerator == "cpu"


def test_the_manifest_refuses_a_device_that_does_not_exist(
    tmp_path: pathlib.Path,
) -> None:
    trace = tmp_path / "t.goal"
    trace.write_text("num_ranks 1\nrank 0 {\nl1: calc 1\n}\n")
    path = tmp_path / "m.yaml"
    path.write_text(
        f"groups:\n  - {{id: a, path: {trace.name}}}\n"
        "config:\n  accelerator: tpu\n"
    )

    with pytest.raises(ValueError, match="accelerator"):
        manifest.load_manifest(path)


def test_the_cli_takes_a_device_and_lists_what_it_found(
    tmp_path: pathlib.Path,
) -> None:
    from cta.synth import goal

    trace = tmp_path / "t.goal"
    goal.write_allreduce(trace)
    runner = testing.CliRunner()

    ok = runner.invoke(
        cli_app.app,
        [
            "analyze",
            str(trace),
            "--accelerator",
            "cpu",
            "--cache-dir",
            str(tmp_path / "c"),
        ],
    )
    bad = runner.invoke(
        cli_app.app, ["analyze", str(trace), "--accelerator", "tpu"]
    )
    listing = runner.invoke(cli_app.app, ["list"])

    assert ok.exit_code == 0, ok.output
    assert bad.exit_code != 0
    assert "Devices" in listing.output and "cpu" in listing.output
