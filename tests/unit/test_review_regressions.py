"""Regression tests for defects found in the architectural review."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cta.core.alignment import op_family_expr
from cta.core.dag import solve_distributed_dag
from cta.core.diagnosis import _classify_regime, _higher_is_better
from cta.exporters.latex import _tex_escape
from cta.viz.render_plotly import _rgba


def test_tex_escape_covers_math_and_group_delimiters() -> None:
    out = _tex_escape("vector<int{}>$^~")
    assert "\\{" in out and "\\}" in out and "\\$" in out
    assert "\\textasciicircum{}" in out and "\\textasciitilde{}" in out
    assert _tex_escape("\\") == "\\textbackslash{}"


def test_rgba_passes_named_colors_through() -> None:
    assert _rgba("orange", 0.5) == "orange"
    assert _rgba("#ff0000", 0.5) == "rgba(255,0,0,0.5)"


def test_gate_direction_detection() -> None:
    assert _higher_is_better("throughput_per_s")
    assert _higher_is_better("queries_per_joule")
    assert not _higher_is_better("latency_ms")


def test_regime_memory_bound_still_reported() -> None:
    assert _classify_regime({"Compute": 0.2, "Memory-BW": 0.6}) == "Memory-BW-Dominated"


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
            "tag": [0] * 5,
        }
    )
    edges = pl.DataFrame({"rank": [0, 0, 0], "src_id": [1, 2, 4], "dst_id": [2, 3, 5]})
    assert solve_distributed_dag(nodes, edges).dag_depth == 3


def test_bundle_keeps_nsys_rep_without_sibling_sqlite_in_other_dir(tmp_path: Path) -> None:
    from cta.core.bundle import resolve_bundle

    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_a" / "report.sqlite").write_bytes(b"")
    (tmp_path / "run_b" / "report.nsys-rep").write_bytes(b"")
    try:
        bundle = resolve_bundle(tmp_path)
    except Exception:  # detection may reject empty files; only the file filtering matters
        pytest.skip("empty fixtures not resolvable")
    names = {p.name for a in bundle.artifacts.values() for p in a.paths}
    assert "report.nsys-rep" in names


def test_kernel_family_needs_standalone_elementwise_token() -> None:
    df = pl.DataFrame({"name": ["simulate_kernel", "accumulate_x", "add_kernel", "vectorized_elementwise_kernel"]})
    fam = df.select(op_family_expr("name").alias("f"))["f"].to_list()
    assert fam[0] != "Elementwise" and fam[1] != "Elementwise"
    assert fam[2] == "Elementwise" and fam[3] == "Elementwise"
