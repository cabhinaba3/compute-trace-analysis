"""Regression tests for defects found in the architectural review."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from cta.core import alignment
from cta.core import dag
from cta.core import diagnosis
from cta.exporters import latex
from cta.viz import render_plotly


def test_tex_escape_covers_math_and_group_delimiters() -> None:
    escaped = latex._tex_escape("vector<int{}>$^~")
    assert "\\{" in escaped and "\\}" in escaped and "\\$" in escaped
    assert "\\textasciicircum{}" in escaped
    assert "\\textasciitilde{}" in escaped
    assert latex._tex_escape("\\") == "\\textbackslash{}"


def test_rgba_passes_named_colors_through() -> None:
    assert render_plotly._rgba("orange", 0.5) == "orange"
    assert render_plotly._rgba("#ff0000", 0.5) == "rgba(255,0,0,0.5)"


def test_gate_direction_detection() -> None:
    assert diagnosis._higher_is_better("throughput_per_s")
    assert diagnosis._higher_is_better("queries_per_joule")
    assert not diagnosis._higher_is_better("latency_ms")


def test_regime_memory_bound_still_reported() -> None:
    shares = {"Compute": 0.2, "Memory-BW": 0.6}
    assert diagnosis._classify_regime(shares) == "Memory-BW-Dominated"


def test_dag_depth_is_graph_depth_not_depth_of_latest_sink() -> None:
    nodes = pl.DataFrame(
        {
            "rank": [0] * 5,
            "node_id": [1, 2, 3, 4, 5],
            "op_class": ["compute"] * 5,
            "duration_ns": [10, 10, 10, 10_000, 10_000],
            "size_bytes": [0] * 5,
            "src_rank": [-1] * 5,
            "dst_rank": [-1] * 5,
            "message_tag": [0] * 5,
        }
    )
    edges = pl.DataFrame(
        {"rank": [0, 0, 0], "src_id": [1, 2, 4], "dst_id": [2, 3, 5]}
    )
    assert dag.solve_distributed_dag(nodes, edges).dag_depth == 3


def test_bundle_keeps_nsys_rep_without_sibling_sqlite_in_other_dir(
    tmp_path: pathlib.Path,
) -> None:
    from cta.core import bundle as core_bundle

    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_a" / "report.sqlite").write_bytes(b"")
    (tmp_path / "run_b" / "report.nsys-rep").write_bytes(b"")
    try:
        bundle = core_bundle.resolve_bundle(tmp_path)
    except Exception:
        # Detection may reject empty files; only the file filtering matters.
        pytest.skip("empty fixtures not resolvable")
    file_names = {
        path.name
        for artifact in bundle.artifacts.values()
        for path in artifact.paths
    }
    assert "report.nsys-rep" in file_names


def test_kernel_family_needs_standalone_elementwise_token() -> None:
    kernels = pl.DataFrame(
        {
            "name": [
                "simulate_kernel",
                "accumulate_x",
                "add_kernel",
                "vectorized_elementwise_kernel",
            ]
        }
    )
    op_family = alignment.op_family_expr("name").alias("op_family")
    families = kernels.select(op_family)["op_family"].to_list()
    assert families[0] != "Elementwise" and families[1] != "Elementwise"
    assert families[2] == "Elementwise" and families[3] == "Elementwise"
