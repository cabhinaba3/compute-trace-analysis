"""Tests that a core install works and names the extra that a feature needs.

`duckdb`, `mlc-chakra` and `protobuf` are optional. A subprocess blocks them
(and the other optional packages), so the real command line runs as it would
on a machine that installed only the default dependencies.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import polars as pl
import pytest

from cta.cli import output
from cta.synth import chakra_proto
from cta.viz import context
from cta.viz import registry as viz_registry

_OPTIONAL = (
    "duckdb",
    "chakra",
    "google",
    "google.protobuf",
    "plotly",
    "pandas",
)
_CLI = f"""
import sys
for name in {_OPTIONAL!r}:
    sys.modules[name] = None
from cta.cli import app
sys.argv = ["cta", *sys.argv[1:]]
app.main()
"""


def _cta(*args: str, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CLI, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_core_install_runs_the_command_line(tmp_path: pathlib.Path) -> None:
    demo = tmp_path / "demo"
    assert _cta("synth", str(demo), cwd=tmp_path).returncode == 0
    assert _cta("list", cwd=tmp_path).returncode == 0
    trace = str(demo / "gpu_bundle_fp16")
    cache = str(tmp_path / "cache")
    analyse = _cta("analyze", trace, "--cache-dir", cache, cwd=tmp_path)
    assert analyse.returncode == 0, analyse.stderr
    plots = tmp_path / "plots"
    plot = _cta(
        "plot",
        trace,
        "--kind",
        "breakdown",
        "--format",
        "svg",
        "--out-dir",
        str(plots),
        "--cache-dir",
        cache,
        cwd=tmp_path,
    )
    assert plot.returncode == 0, plot.stderr
    assert (plots / "breakdown.svg").exists()


def test_core_install_names_the_extra_for_sql(tmp_path: pathlib.Path) -> None:
    demo = tmp_path / "demo"
    assert _cta("synth", str(demo), cwd=tmp_path).returncode == 0
    result = _cta(
        "plot",
        str(demo / "gpu_bundle_fp16"),
        "--sql",
        "SELECT 1 AS x",
        "--mark",
        "scatter",
        "--x",
        "x",
        "--y",
        "x",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out-dir",
        str(tmp_path / "plots"),
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "compute-trace-analysis[sql]" in result.stdout + result.stderr


def test_sql_and_where_name_the_sql_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(viz_registry.NotApplicableError, match=r"\[sql\]"):
        context.DataContext(None).sql("SELECT 1")
    with pytest.raises(viz_registry.NotApplicableError, match=r"\[sql\]"):
        context._sql_filter(pl.DataFrame({"a": [1]}), "a > 0")


def test_writing_et_files_names_the_chakra_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chakra_proto.et_def_pb2.cache_clear()
    # Other tests may have imported the schema already, so block every level.
    for name in (
        "chakra",
        "chakra.schema",
        "chakra.schema.protobuf",
        "chakra.schema.protobuf.et_def_pb2",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    try:
        with pytest.raises(ImportError, match=r"\[chakra\]"):
            chakra_proto.et_def_pb2()
    finally:
        chakra_proto.et_def_pb2.cache_clear()


def test_errors_keep_the_brackets_of_install_hints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output.print_error("pip install 'compute-trace-analysis[sql]'")
    output.print_warning("see [bold]this[/bold]")
    err = capsys.readouterr().err
    assert "error: pip install 'compute-trace-analysis[sql]'" in err
    assert "warning: see [bold]this[/bold]" in err
