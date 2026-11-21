"""Tests of the export format registry and its command-line check."""

from __future__ import annotations

import pathlib

import pytest
from typer import testing

from cta import _optional
from cta.cli import app as cli_app
from cta.export import writers
from cta.synth import kineto
from cta.viz import registry


def test_formats_resolve_aliases_in_the_order_given() -> None:
    assert writers.resolve_formats(["JSON", "md", "json", "tex"]) == [
        "json",
        "markdown",
        "latex",
    ]


def test_unknown_formats_name_the_closest_one() -> None:
    with pytest.raises(ValueError, match="Did you mean md"):
        writers.resolve_formats(["mdd"])


def test_static_figures_draw_the_exported_kinds() -> None:
    assert set(registry.exported_kinds()) == {
        "breakdown",
        "topk",
        "traffic_delta",
        "skew",
        "roofline",
        "scaling",
        "sim_attribution",
        "sim_breakdown",
        "sim_phases",
        "sim_ranks",
    }


def test_cli_rejects_an_unknown_format_before_analysing(
    tmp_path: pathlib.Path,
) -> None:
    trace = tmp_path / "trace.json"
    kineto.write_trace(trace)
    result = testing.CliRunner().invoke(
        cli_app.app,
        [
            "analyze",
            str(trace),
            "--out-dir",
            str(tmp_path / "out"),
            "--export",
            "json,mdd",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 2
    assert "mdd" in result.output
    assert not (tmp_path / "cache").exists()  # nothing was ingested


def test_install_hints_name_the_distribution() -> None:
    error = _optional.missing("plotly.io", extra="viz", purpose="HTML plots")
    assert str(error) == (
        "plotly is needed for HTML plots: "
        "pip install 'compute-trace-analysis[viz]'"
    )
    with pytest.raises(ImportError, match=r"compute-trace-analysis\[sql\]"):
        _optional.require("cta_no_such_module", extra="sql", purpose="SQL")
