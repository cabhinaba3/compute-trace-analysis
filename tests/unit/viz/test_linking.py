"""Tests of the links between charts (`cta.viz.linking`).

A step picked for a run zooms the charts with a time axis, so the window of a
step has to be in the time base those charts draw in; a kernel family picked
for a run dims the others in all charts that tag them. All of it is Python
acting on the chart IR: there is no page script to test.
"""

from __future__ import annotations

import copy
import json
import pathlib

import polars as pl
import pytest
from typer import testing

import cta
from cta.cli import app as cli_app
from cta.synth import nsys as synth_nsys
from cta.viz import context
from cta.viz import dashboard
from cta.viz import linking
from cta.viz import plots
from cta.viz import server

EPS = 1e-6


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> cta.Report:
    """Two groups of eight steps of the generated Nsight Systems trace."""
    root = tmp_path_factory.mktemp("link")
    base = synth_nsys.write_sqlite(root / "base.sqlite", steps=8)
    cand = synth_nsys.write_sqlite(root / "cand.sqlite", steps=8, gemm_us=600)
    return cta.compare(base, cand, cache_dir=root / "cache")


@pytest.fixture(scope="module")
def ctx(report: cta.Report) -> context.DataContext:
    return context.DataContext(report)


def _family_of(ctx: context.DataContext) -> str:
    """Returns the op family of the first timeline series."""
    chart, _ = plots.build_chart(
        ctx, "timeline", {"by": "op_family"}, warn=False
    )
    return chart.panels[0].traces[0].meta["family"]


# --- the data the selection reads -----------------------------------------


def test_step_windows_are_the_steps_table_in_the_time_base(
    ctx: context.DataContext,
) -> None:
    windows = linking.step_windows(ctx)
    steps = ctx.frame("steps")

    assert [(w["label"], w["rank"]) for w in windows] == [
        ("base", 0),
        ("cand", 0),
    ]
    for window in windows:
        rows = steps.filter(pl.col("label") == window["label"]).sort("idx")
        assert window["idx"] == rows["idx"].to_list() == list(range(8))
        assert window["t0"] == pytest.approx(rows["t_ms"].to_list(), abs=1e-4)
        assert window["t1"] == pytest.approx(rows["end_ms"].to_list(), abs=1e-4)
        # Steps follow each other without overlap.
        assert all(
            a <= b + EPS
            for a, b in zip(window["t1"], window["t0"][1:], strict=False)
        )


def test_the_kernels_of_a_step_lie_in_its_window(
    ctx: context.DataContext,
) -> None:
    windows = {w["label"]: w for w in linking.step_windows(ctx)}
    kernels = ctx.frame("kernels")

    for row in kernels.iter_rows(named=True):
        window = windows[row["label"]]
        k = window["idx"].index(row["idx"])
        assert row["t_ms"] >= window["t0"][k] - EPS
        assert row["end_ms"] <= window["t1"][k] + EPS


@pytest.mark.parametrize(
    "kind", ["timeline", "queue_depth", "runtime_timeline"]
)
def test_time_axis_charts_draw_in_the_time_base_of_the_windows(
    ctx: context.DataContext, kind: str
) -> None:
    windows = {w["label"]: w for w in linking.step_windows(ctx)}

    chart, _ = plots.build_chart(ctx, kind, {}, warn=False)

    assert chart.panels
    for panel in chart.panels:
        assert panel.xkind == "time"
        label = panel.traces[0].meta["label"]
        first, last = windows[label]["t0"][0], windows[label]["t1"][-1]
        for trace in panel.traces:
            for values in (trace.x, trace.x1):
                for value in values or []:
                    assert first - EPS <= value <= last + EPS
    # The default timeline shows steps 1-3: those windows hold every bar.
    if kind == "timeline":
        for panel in chart.panels:
            window = windows[panel.traces[0].meta["label"]]
            lo, hi = window["t0"][1], window["t1"][3]
            for trace in panel.traces:
                assert trace.x is not None and trace.x1 is not None
                assert min(trace.x) >= lo - EPS
                assert max(trace.x1) <= hi + EPS


def test_op_family_series_and_axes_carry_the_family(
    ctx: context.DataContext,
) -> None:
    by_family, _ = plots.build_chart(
        ctx, "timeline", {"by": "op_family"}, warn=False
    )
    by_category, _ = plots.build_chart(
        ctx, "timeline", {"by": "category"}, warn=False
    )
    share, _ = plots.build_chart(
        ctx, "family_share", {"by": "op_family"}, warn=False
    )
    share_by_kernel, _ = plots.build_chart(
        ctx, "family_share", {"by": "category"}, warn=False
    )
    pareto, _ = plots.build_chart(
        ctx, "pareto", {"by": "op_family"}, warn=False
    )
    pareto_by_kernel, _ = plots.build_chart(
        ctx, "pareto", {"by": "canonical_name"}, warn=False
    )

    for panel in by_family.panels:
        for trace in panel.traces:
            assert trace.meta["family"] == trace.name
    assert all(
        "family" not in t.meta for p in by_category.panels for t in p.traces
    )
    assert all(t.meta["family"] == t.name for t in share.panels[0].traces)
    assert all("family" not in t.meta for t in share_by_kernel.panels[0].traces)
    bars = [t for p in pareto.panels for t in p.traces if t.mark == "bar"]
    assert bars and all(t.meta["family_axis"] == "x" for t in bars)
    assert all(
        "family_axis" not in t.meta
        for p in pareto_by_kernel.panels
        for t in p.traces
    )


# --- parsing and finding windows -------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", linking.StepPick(4)),
        (" 4@1 ", linking.StepPick(4, rank=1)),
        ("0@0", linking.StepPick(0, rank=0)),
    ],
)
def test_a_step_is_read_as_idx_or_idx_at_rank(
    text: str, expected: linking.StepPick
) -> None:
    assert linking.parse_step(text) == expected


@pytest.mark.parametrize("text", ["", "x", "4@", "4@x", "-1", "4@-2", "1.5"])
def test_a_step_that_is_not_idx_or_idx_at_rank_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="step"):
        linking.parse_step(text)


def test_the_window_comes_from_the_most_exact_series() -> None:
    windows = [
        {
            "label": "a",
            "trial": 0,
            "rank": 0,
            "idx": [0, 1],
            "t0": [0, 10],
            "t1": [10, 20],
        },
        {
            "label": "a",
            "trial": 1,
            "rank": 1,
            "idx": [0, 1, 2],
            "t0": [5, 15, 25],
            "t1": [15, 25, 35],
        },
        {
            "label": "b",
            "trial": 0,
            "rank": 0,
            "idx": [0, 1, 2],
            "t0": [1, 2, 3],
            "t1": [2, 3, 4],
        },
    ]
    find = linking.find_window

    # Rank and trial match exactly, then the rank, then any series with it.
    assert find(windows, linking.StepPick(1, rank=1, trial=1), "a") == (15, 25)
    assert find(windows, linking.StepPick(1, rank=0), "a") == (10, 20)
    assert find(windows, linking.StepPick(2, rank=0), "a") == (25, 35)
    # With no rank the first series that has the step wins.
    assert find(windows, linking.StepPick(1), "a") == (10, 20)
    # The chart's label beats the pick's, which beats "any label".
    assert find(windows, linking.StepPick(1, label="a"), "b") == (2, 3)
    assert find(windows, linking.StepPick(1, label="b")) == (2, 3)
    assert find(windows, linking.StepPick(2)) == (25, 35)
    assert find(windows, linking.StepPick(9)) is None
    assert find([], linking.StepPick(0)) is None


def test_family_names_cut_to_36_characters_still_match() -> None:
    long_name = "a" * 40
    assert linking.same_family(long_name, long_name[:36])
    assert linking.same_family("gemm", "gemm")
    assert not linking.same_family("gemm", "attn")


# --- applying a selection ---------------------------------------------------


def test_an_empty_selection_changes_nothing(ctx: context.DataContext) -> None:
    chart, _ = plots.build_chart(ctx, "timeline", {}, warn=False)
    before = copy.deepcopy(chart)

    assert not linking.Selection()
    assert linking.apply(chart, linking.Selection(), ctx) is chart
    assert chart == before


def test_a_picked_step_zooms_every_time_axis_to_its_window(
    ctx: context.DataContext,
) -> None:
    windows = {w["label"]: w for w in linking.step_windows(ctx)}
    picked = context.DataContext(
        ctx.report, selection=linking.Selection(step=linking.StepPick(2))
    )

    for kind in ("timeline", "queue_depth", "runtime_timeline"):
        chart, _ = plots.build_chart(picked, kind, {}, warn=False)
        assert "Selection: step 2." in chart.notes
        for panel in chart.panels:
            window = windows[panel.traces[0].meta["label"]]
            wanted = (window["t0"][2], window["t1"][2])
            if panel.xrange is not None and panel.xrange == pytest.approx(
                wanted
            ):
                continue
            # A chart keeps its own range only when it did not draw the step,
            # and then says so (runtime_timeline draws its own slice).
            assert kind == "runtime_timeline", kind
            assert "is not in what this chart drew" in chart.notes
        if kind != "runtime_timeline":
            assert all(
                panel.xrange == pytest.approx((w["t0"][2], w["t1"][2]))
                for panel in chart.panels
                for w in [windows[panel.traces[0].meta["label"]]]
            ), kind


def test_a_step_the_chart_did_not_draw_is_left_alone_and_named(
    ctx: context.DataContext,
) -> None:
    """The timeline draws steps 1-3 by default; step 6 is not in it."""
    plain, _ = plots.build_chart(ctx, "timeline", {}, warn=False)
    picked = context.DataContext(
        ctx.report, selection=linking.Selection(step=linking.StepPick(6))
    )

    chart, _ = plots.build_chart(picked, "timeline", {}, warn=False)

    assert [p.xrange for p in chart.panels] == [p.xrange for p in plain.panels]
    assert chart.notes.count("Step 6 is not in what this chart drew") == len(
        chart.panels
    )


def test_the_other_families_are_dimmed(ctx: context.DataContext) -> None:
    family = _family_of(ctx)
    picked = context.DataContext(
        ctx.report, selection=linking.Selection(family=family)
    )

    chart, _ = plots.build_chart(
        picked, "timeline", {"by": "op_family"}, warn=False
    )

    traces = [t for p in chart.panels for t in p.traces]
    assert {t.meta["family"] for t in traces} > {family}
    for trace in traces:
        if trace.meta["family"] == family:
            assert trace.style.get("alpha", 1.0) == 1.0
        else:
            assert trace.style["alpha"] == pytest.approx(linking.DIM)
    # A chart that does not tag families is not touched.
    steps_chart, _ = plots.build_chart(picked, "step_trace", {}, warn=False)
    plain, _ = plots.build_chart(ctx, "step_trace", {}, warn=False)
    assert [t.style for t in steps_chart.panels[0].traces] == [
        t.style for t in plain.panels[0].traces
    ]


def test_a_category_axis_of_families_dims_bars_one_by_one(
    ctx: context.DataContext,
) -> None:
    family = _family_of(ctx)
    picked = context.DataContext(
        ctx.report, selection=linking.Selection(family=family)
    )

    chart, resolved = plots.build_chart(
        picked, "pareto", {"by": "op_family"}, warn=False
    )

    bars = [t for p in chart.panels for t in p.traces if t.mark == "bar"]
    assert bars
    for bar in bars:
        alpha = bar.style["alpha"]
        assert bar.x is not None
        assert len(alpha) == len(bar.x)
        for category, a in zip(bar.x, alpha, strict=True):
            expected = (
                1.0 if linking.same_family(category, family) else linking.DIM
            )
            assert a == pytest.approx(expected)
    # Both backends draw it.
    fig = plots.to_plotly(chart, resolved)
    plotly_bars = [t for t in fig.data if t.type == "bar"]
    assert plotly_bars
    assert list(plotly_bars[0].marker.opacity) == pytest.approx(
        bars[0].style["alpha"]
    )
    plots.to_matplotlib(chart, resolved)


def test_a_scalar_bar_alpha_reaches_both_backends(
    ctx: context.DataContext,
) -> None:
    family = _family_of(ctx)
    picked = context.DataContext(
        ctx.report, selection=linking.Selection(family=family)
    )

    chart, resolved = plots.build_chart(
        picked, "family_share", {"by": "op_family"}, warn=False
    )

    fig = plots.to_plotly(chart, resolved)
    opacities = {t.name: t.marker.opacity for t in fig.data if t.type == "bar"}
    assert opacities[family] in (None, 1.0)
    assert any(o == pytest.approx(linking.DIM) for o in opacities.values())
    plots.to_matplotlib(chart, resolved)


def test_a_step_and_a_family_together(ctx: context.DataContext) -> None:
    selection = linking.Selection(
        step=linking.StepPick(2, rank=0), family=_family_of(ctx)
    )

    assert (
        selection.describe() == f"step 2 (rank 0) · family {selection.family}"
    )
    chart, _ = plots.build_chart(
        context.DataContext(ctx.report, selection=selection),
        "timeline",
        {"by": "op_family"},
        warn=False,
    )

    assert chart.notes.endswith(f"Selection: {selection.describe()}.")
    assert all(p.xrange is not None for p in chart.panels)


def test_check_refuses_a_step_the_report_does_not_have(
    ctx: context.DataContext,
) -> None:
    linking.check(ctx, linking.Selection())
    linking.check(ctx, linking.Selection(step=linking.StepPick(7)))
    with pytest.raises(ValueError, match="Step 99 is not in the report"):
        linking.check(ctx, linking.Selection(step=linking.StepPick(99)))
    with pytest.raises(ValueError, match="for rank 3"):
        linking.check(ctx, linking.Selection(step=linking.StepPick(0, rank=3)))


# --- the entry points ---------------------------------------------------------


def test_report_plot_follows_a_selection(
    report: cta.Report, tmp_path: pathlib.Path
) -> None:
    picked = linking.Selection(step=linking.StepPick(2))

    fig = report.plot("timeline", tmp_path / "t.html", selection=picked)

    assert any(
        axis.range is not None for axis in (fig.layout.xaxis, fig.layout.xaxis2)
    )


def test_a_dashboard_follows_a_selection_without_a_page_script(
    report: cta.Report, tmp_path: pathlib.Path
) -> None:
    plain = dashboard.build_dashboard(
        context.DataContext(report), None, tmp_path / "a.html", cdn=True
    ).path.read_text()
    picked = dashboard.build_dashboard(
        context.DataContext(
            report, selection=linking.Selection(step=linking.StepPick(2))
        ),
        None,
        tmp_path / "b.html",
        cdn=True,
    ).path.read_text()

    # The selection is in the figures, not in extra script: the page is the
    # same code either way.
    assert picked != plain
    assert "Selection: step 2." in picked
    for page in (plain, picked):
        assert "cta-links" not in page
        assert "plotly_click" not in page
        assert "ctaFind" not in page


def test_the_live_server_applies_the_selection_to_every_slice(
    report: cta.Report,
) -> None:
    selection = linking.Selection(step=linking.StepPick(2))
    session = server.LiveSession(report, None, selection=selection)
    timeline = next(c for c in session.cards if c.spec.kind == "timeline")

    whole = session.plot(timeline, server.Filters())
    sliced = session.plot(
        timeline, server.Filters.from_query({"steps": ["1:4"]})
    )

    for response in (whole, sliced):
        assert "error" not in response
        assert "Selection: step 2." in response["notes"]
        layout = json.loads(response["figure"])["layout"]
        assert layout["xaxis"]["range"][0] < layout["xaxis"]["range"][1]


def _kineto_args(root: pathlib.Path) -> list[str]:
    base = synth_nsys.write_sqlite(root / "b.sqlite", steps=8)
    return [str(base), "--cache-dir", str(root / "c")]


def test_cli_pick_step_and_family(tmp_path: pathlib.Path) -> None:
    runner = testing.CliRunner()
    args = _kineto_args(tmp_path)

    ok = runner.invoke(
        cli_app.app,
        [
            "plot",
            *args,
            "--kind",
            "timeline",
            "--pick-step",
            "2",
            "--pick-family",
            "gemm",
            "--out-dir",
            str(tmp_path / "f"),
            "--format",
            "html",
        ],
    )
    bad = runner.invoke(
        cli_app.app,
        ["plot", *args, "--kind", "timeline", "--pick-step", "99"],
    )
    malformed = runner.invoke(
        cli_app.app,
        ["plot", *args, "--kind", "timeline", "--pick-step", "two"],
    )

    assert ok.exit_code == 0, ok.output
    assert (tmp_path / "f" / "timeline.html").exists()
    assert bad.exit_code != 0
    assert "Step 99 is not in the report" in bad.output
    assert malformed.exit_code != 0
    assert "IDX or IDX@RANK" in malformed.output
