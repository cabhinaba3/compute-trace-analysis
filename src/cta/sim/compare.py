"""Compares simulation runs: GOAL with GOAL, Chakra with Chakra, and across.

``compare`` simulates each input and compares every run with the first (the
baseline). A Chakra trace with collectives runs twice by default, lowered and
analytical, so a GOAL schedule can be set against both: the lowered Chakra
run is like for like with GOAL (the same kind of messages on the same
network), and the gap between Chakra's two levels is what contention and
serialisation add to the analytical collective costs.

The comparison's tables:

* ``overview``: one row per run with its makespan, its difference from the
  baseline, critical-path shares and volumes.
* ``breakdown``: critical-path time per run and segment.
* ``attribution``: per run, the makespan gap to the baseline split by
  critical-path segment. Each critical path tiles [0, makespan], so the
  deltas add up to the gap exactly.
* ``ranks``: per run and rank, finish time and where the time went.
* ``phases``: per run, rank and phase (``cta.sim.alignment``), the span,
  compute and communication of the phase in the baseline and in the run, and
  the bytes and collectives of its region.
* ``ops``: the matched compute ops whose start moved the most.
* ``structure``: graph-level counts of each run (ops by kind, edges, compute,
  messages and bytes on the wire) and their differences from the baseline.
* ``traffic``: bytes per (src, dst) pair in the baseline and each run.
* ``issues``: every run's validation issues.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import pathlib
from typing import Any

import polars as pl

from cta import models
from cta.sim import alignment
from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib
from cta.sim import result as result_lib

LEVEL_CHOICES = ("auto", "lowered", "analytical", "both")


@dataclasses.dataclass(slots=True)
class SimComparison:
    """Runs and the tables comparing them with the first (see the module)."""

    results: list[result_lib.SimResult]
    overview: pl.DataFrame
    breakdown: pl.DataFrame
    attribution: pl.DataFrame
    ranks: pl.DataFrame
    phases: pl.DataFrame
    ops: pl.DataFrame
    structure: pl.DataFrame
    traffic: pl.DataFrame
    issues: pl.DataFrame

    @property
    def baseline(self) -> result_lib.SimResult:
        """The run the others are compared with."""
        return self.results[0]

    def tables(self) -> dict[str, pl.DataFrame]:
        """Returns every table by name."""
        return {
            name: getattr(self, name)
            for name in (
                "overview",
                "breakdown",
                "attribution",
                "ranks",
                "phases",
                "ops",
                "structure",
                "traffic",
                "issues",
            )
        }

    def to_report(self) -> models.Report:
        """Returns the comparison as a Report, for the exporters and plots.

        One group per run; the ``simulation`` aspect holds a summary of
        headline metrics and every table of the comparison as details.
        """
        groups = tuple(
            models.TrialGroup(group_id=r.name, label=r.name, trials=())
            for r in self.results
        )
        return models.Report(
            mode="single"
            if len(groups) == 1
            else "pair"
            if len(groups) == 2
            else "multi",
            groups=groups,
            aspects={"simulation": aspect_result(self)},
        )

    def write(
        self, out_dir: str | pathlib.Path, *, csv: bool = False
    ) -> list[pathlib.Path]:
        """Writes the non-empty tables (Parquet or CSV); returns the files."""
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, table in self.tables().items():
            if table.is_empty():
                continue
            path = out / f"{name}.{'csv' if csv else 'parquet'}"
            if csv:
                table.write_csv(path)
            else:
                table.write_parquet(path)
            written.append(path)
        return written


# Headline metrics of the ``simulation`` aspect: name, unit, lower is better.
SUMMARY_METRICS = (
    ("makespan_ms", "ms", True),
    ("cp_compute_pct", "%", False),
    ("cp_comm_pct", "%", True),
    ("cp_contention_pct", "%", True),
    ("rank_finish_spread_ms", "ms", True),
    ("mean_utilization", "", False),
    ("wire_gb", "GB", True),
)


def aspect_result(comparison: SimComparison) -> models.AspectResult:
    """Returns the comparison as the ``simulation`` aspect's result."""
    overview = comparison.overview
    labels = overview["run"].to_list()
    rows = []
    for metric, unit, lower_is_better in SUMMARY_METRICS:
        row: dict[str, Any] = {"metric": metric, "unit": unit}
        values = overview[metric].to_list()
        for label, value in zip(labels, values, strict=True):
            row[label] = value
        base = values[0] if values else None
        for label, value in zip(labels[1:], values[1:], strict=True):
            row[f"{label}_delta_pct"] = (
                100.0 * (value - base) / base if base else None
            )
        row["lower_is_better"] = lower_is_better
        rows.append(row)
    return models.AspectResult(
        name="simulation",
        title="Simulation",
        summary=pl.DataFrame(rows),
        details=comparison.tables(),
        scalars={
            metric: dict(zip(labels, overview[metric].to_list(), strict=True))
            for metric, _, _ in SUMMARY_METRICS
        },
    )


def run_sources(
    sources: Sequence[str | pathlib.Path | graph_lib.SimGraph],
    settings: config_lib.SimConfig,
    *,
    labels: Sequence[str] | None = None,
    levels: str = "auto",
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> list[result_lib.SimResult]:
    """Simulates each source at the requested levels.

    Args:
        sources: Traces or graphs, the baseline first.
        settings: The config every run uses.
        labels: Names of the sources, in order.
        levels: ``auto`` runs graphs with collectives both lowered and
            analytical when another input has none (a GOAL schedule) and
            at the config's level otherwise; ``both`` always runs both;
            ``lowered`` or ``analytical`` runs that level only.
        cache_dir: Directory of the Parquet ingest cache.

    Returns:
        The results, in source order (lowered before analytical).
    """
    if levels not in LEVEL_CHOICES:
        raise ValueError(f"levels must be one of {', '.join(LEVEL_CHOICES)}")
    graphs = []
    for i, source in enumerate(sources):
        label = labels[i] if labels and i < len(labels) else None
        if isinstance(source, graph_lib.SimGraph):
            graphs.append(
                dataclasses.replace(source, label=label) if label else source
            )
        else:
            graphs.append(
                graph_lib.load_graph(
                    source,
                    label=label,
                    groups=settings.collectives.groups,
                    world_size=settings.world_size,
                    cache_dir=cache_dir,
                )
            )
    names = [g.label for g in graphs]
    if len(set(names)) != len(names):
        graphs = [
            dataclasses.replace(g, label=f"{g.label}_{i}")
            for i, g in enumerate(graphs)
        ]
    mixed = any(not g.has_collectives for g in graphs) and any(
        g.has_collectives for g in graphs
    )
    results = []
    for g in graphs:
        if not g.has_collectives:
            results.append(engine.run_graph(g, settings))
            continue
        if levels == "both" or (levels == "auto" and mixed):
            chosen: tuple[str, ...] = ("lowered", "analytical")
        elif levels == "auto":
            chosen = (settings.collectives.level,)
        else:
            chosen = (levels,)
        results.extend(engine.run_graph(g, settings, level=lv) for lv in chosen)
    return results


def compare(
    sources: Sequence[str | pathlib.Path | graph_lib.SimGraph],
    settings: config_lib.SimConfig | None = None,
    *,
    labels: Sequence[str] | None = None,
    levels: str = "auto",
    cache_dir: str | pathlib.Path = ".cta_cache",
    top: int = 20,
    tolerance: float = 0.01,
) -> SimComparison:
    """Simulates the sources and compares every run with the first.

    Args:
        sources: Two or more traces (GOAL, Chakra, or both) or graphs.
        settings: The config; None uses the defaults.
        labels: Names of the sources.
        levels: See ``run_sources``.
        cache_dir: Directory of the Parquet ingest cache.
        top: How many divergent ops to list.
        tolerance: Duration tolerance of anchor alignment.

    Returns:
        The comparison.
    """
    settings = settings or config_lib.SimConfig()
    results = run_sources(
        sources, settings, labels=labels, levels=levels, cache_dir=cache_dir
    )
    return compare_results(results, top=top, tolerance=tolerance)


def compare_results(
    results: Sequence[result_lib.SimResult],
    *,
    top: int = 20,
    tolerance: float = 0.01,
) -> SimComparison:
    """Compares runs (which must keep their op tables) with the first."""
    if not results:
        raise ValueError("nothing to compare")
    results = list(results)
    phases, ops = _phases_and_ops(results, top, tolerance)
    return SimComparison(
        results=results,
        overview=_overview(results),
        breakdown=pl.concat(
            [
                r.breakdown().select(pl.lit(r.name).alias("run"), pl.all())
                for r in results
            ]
        ),
        attribution=_attribution(results),
        ranks=_ranks(results),
        phases=phases,
        ops=ops,
        structure=_structure(results),
        traffic=_traffic(results),
        issues=pl.concat(
            [
                r.issues.select(pl.lit(r.name).alias("run"), pl.all())
                for r in results
            ]
        ),
    )


def _wire_bytes(r: result_lib.SimResult) -> float:
    return float(r.ranks["bytes_sent"].sum()) if r.ranks.height else 0.0


def _overview(results: Sequence[result_lib.SimResult]) -> pl.DataFrame:
    base = results[0].makespan_ns
    rows = []
    for r in results:
        s = r.summary()
        rows.append(
            {
                "run": r.name,
                "label": r.label,
                "format": r.trace_format,
                "level": r.level,
                "model": r.config.model,
                "completed": r.completed,
                "makespan_ms": s["makespan_ms"],
                "delta_ms": (r.makespan_ns - base) / 1e6,
                "delta_pct": 100.0 * (r.makespan_ns - base) / base
                if base
                else 0.0,
                "cp_compute_pct": s["cp_compute_pct"],
                "cp_comm_pct": s["cp_comm_pct"],
                "cp_contention_pct": s["cp_contention_pct"],
                "cp_collective_pct": s["cp_collective_pct"],
                "rank_finish_spread_ms": s.get("rank_finish_spread_ns", 0.0)
                / 1e6,
                "mean_utilization": s.get("mean_utilization", 0.0),
                "ops": int(r.stats["ops"]),
                "messages": int(s.get("total_messages", 0.0)),
                "wire_gb": _wire_bytes(r) / 1e9,
                "events": int(r.stats["events"]),
                "wall_s": float(r.stats["wall_seconds"]),
            }
        )
    return pl.DataFrame(rows)


def _relation(base: result_lib.SimResult, other: result_lib.SimResult) -> str:
    if other is base:
        return "baseline"
    if other.label == base.label:
        return "same trace, other level"
    if other.trace_format.startswith("chakra") == base.trace_format.startswith(
        "chakra"
    ):
        return "same format"
    return "cross format"


def _attribution(results: Sequence[result_lib.SimResult]) -> pl.DataFrame:
    base = results[0]
    base_cp = dict(base.breakdown().select("segment", "ns").iter_rows())
    rows = []
    for r in results[1:]:
        cp = dict(r.breakdown().select("segment", "ns").iter_rows())
        gap = r.makespan_ns - base.makespan_ns
        for segment in result_lib.SEGMENTS:
            delta = cp.get(segment, 0) - base_cp.get(segment, 0)
            rows.append(
                {
                    "run": r.name,
                    "relation": _relation(base, r),
                    "segment": segment,
                    "baseline_ns": base_cp.get(segment, 0),
                    "run_ns": cp.get(segment, 0),
                    "delta_ns": delta,
                    "share_of_gap_pct": 100.0 * delta / gap if gap else 0.0,
                }
            )
    schema = {
        "run": pl.String,
        "relation": pl.String,
        "segment": pl.String,
        "baseline_ns": pl.Int64,
        "run_ns": pl.Int64,
        "delta_ns": pl.Int64,
        "share_of_gap_pct": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _ranks(results: Sequence[result_lib.SimResult]) -> pl.DataFrame:
    base = results[0].ranks.select(
        "rank", pl.col("finish_ns").alias("baseline_finish_ns")
    )
    frames = [
        r.ranks.select(
            pl.lit(r.name).alias("run"),
            "rank",
            "finish_ns",
            "compute_ns",
            "overhead_ns",
            "wait_ns",
            "sync_ns",
            "idle_ns",
            "utilization",
            "bytes_sent",
        )
        .join(base, on="rank", how="left")
        .with_columns(
            (pl.col("finish_ns") - pl.col("baseline_finish_ns")).alias(
                "delta_ns"
            )
        )
        for r in results
    ]
    return pl.concat(frames)


def _phases_and_ops(
    results: Sequence[result_lib.SimResult], top: int, tolerance: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = results[0]
    if base.ops.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    base_anchors = alignment.anchors(base.ops)
    phase_frames, op_frames = [], []
    for r in results[1:]:
        if r.ops.is_empty():
            continue
        matched = alignment.match(
            base_anchors, alignment.anchors(r.ops), tolerance=tolerance
        )
        if matched.is_empty():
            continue
        base_phases = alignment.phases(
            base.ops,
            base.ranks,
            matched.select("rank", "phase", pl.col("op_base").alias("op")),
        )
        run_phases = alignment.phases(
            r.ops,
            r.ranks,
            matched.select("rank", "phase", pl.col("op_other").alias("op")),
        )
        joined = base_phases.join(
            run_phases, on=["rank", "phase"], how="inner", suffix="_run"
        )
        phase_frames.append(
            joined.select(
                pl.lit(r.name).alias("run"),
                "rank",
                "phase",
                "anchor",
                pl.col("start_ns").alias("baseline_start_ns"),
                pl.col("start_ns_run").alias("run_start_ns"),
                pl.col("span_ns").alias("baseline_span_ns"),
                pl.col("span_ns_run").alias("run_span_ns"),
                (pl.col("span_ns_run") - pl.col("span_ns")).alias(
                    "span_delta_ns"
                ),
                pl.col("comm_ns").alias("baseline_comm_ns"),
                pl.col("comm_ns_run").alias("run_comm_ns"),
                pl.col("bytes").alias("baseline_bytes"),
                pl.col("bytes_run").alias("run_bytes"),
                (
                    pl.col("bytes_run").cast(pl.Int64)
                    - pl.col("bytes").cast(pl.Int64)
                ).alias("bytes_delta"),
                pl.col("collectives").alias("baseline_collectives"),
                pl.col("collectives_run").alias("run_collectives"),
            )
        )
        moved = (
            matched.join(
                base.ops.select(
                    pl.col("op").alias("op_base"),
                    "name",
                    pl.col("start_ns").alias("baseline_start_ns"),
                    (pl.col("end_ns") - pl.col("start_ns")).alias(
                        "baseline_duration_ns"
                    ),
                ),
                on="op_base",
            )
            .join(
                r.ops.select(
                    pl.col("op").alias("op_other"),
                    pl.col("start_ns").alias("run_start_ns"),
                    (pl.col("end_ns") - pl.col("start_ns")).alias(
                        "run_duration_ns"
                    ),
                ),
                on="op_other",
            )
            .with_columns(
                (pl.col("run_start_ns") - pl.col("baseline_start_ns")).alias(
                    "start_delta_ns"
                ),
                (
                    pl.col("run_duration_ns") - pl.col("baseline_duration_ns")
                ).alias("duration_delta_ns"),
            )
            .sort(pl.col("start_delta_ns").abs(), descending=True)
            .head(top)
            .select(
                pl.lit(r.name).alias("run"),
                "rank",
                "phase",
                "name",
                "baseline_start_ns",
                "run_start_ns",
                "start_delta_ns",
                "baseline_duration_ns",
                "run_duration_ns",
                "duration_delta_ns",
            )
        )
        op_frames.append(moved)
    return (
        pl.concat(phase_frames) if phase_frames else pl.DataFrame(),
        pl.concat(op_frames) if op_frames else pl.DataFrame(),
    )


def _structure(results: Sequence[result_lib.SimResult]) -> pl.DataFrame:
    rows = []
    for r in results:
        row: dict[str, Any] = {"run": r.name}
        if not r.ops.is_empty():
            kinds = dict(r.ops.group_by("kind").len().iter_rows())
            for kind in graph_lib.KIND_NAMES:
                row[f"{kind}_ops"] = int(kinds.get(kind, 0))
            calcs = r.ops.filter(
                (pl.col("kind") == "calc") & (pl.col("role") == "original")
            )
            row["compute_total_ns"] = int(calcs["duration_ns"].sum())
            colls = r.ops.filter(
                (pl.col("kind") == "collective") | (pl.col("role") == "entry")
            )
            row["collective_ops"] = colls.height
        row["messages"] = (
            int(r.ranks["messages"].sum()) if r.ranks.height else 0
        )
        row["wire_bytes"] = int(_wire_bytes(r))
        row["ranks"] = r.ranks.height
        row["issues"] = int(
            r.issues.filter(pl.col("severity") == "error")["count"].sum()
        )
        rows.append(row)
    table = pl.DataFrame(rows)
    numeric = [
        c for c, t in table.schema.items() if c != "run" and t.is_numeric()
    ]
    return table.with_columns(
        [(pl.col(c) - pl.col(c).first()).alias(f"{c}_delta") for c in numeric]
    )


def _traffic(results: Sequence[result_lib.SimResult]) -> pl.DataFrame:
    frames = []
    for r in results:
        if r.messages.is_empty():
            continue
        frames.append(
            r.messages.group_by("src", "dst")
            .agg(
                pl.col("bytes").sum().alias("bytes"), pl.len().alias("messages")
            )
            .select(pl.lit(r.name).alias("run"), pl.all())
        )
    if not frames:
        return pl.DataFrame()
    table = pl.concat(frames)
    base = table.filter(pl.col("run") == results[0].name).select(
        "src", "dst", pl.col("bytes").alias("baseline_bytes")
    )
    return (
        table.join(base, on=["src", "dst"], how="full", coalesce=True)
        .with_columns(
            pl.col("run").fill_null("(missing)"),
            pl.col("bytes").fill_null(0),
            pl.col("baseline_bytes").fill_null(0),
        )
        .with_columns(
            (
                pl.col("bytes").cast(pl.Int64)
                - pl.col("baseline_bytes").cast(pl.Int64)
            ).alias("delta_bytes")
        )
        .sort("run", "src", "dst")
    )


def traffic_mismatch(comparison: SimComparison) -> dict[str, float]:
    """Per run, the bytes its traffic matrix differs from the baseline's.

    Runs at the analytical level are left out: their collectives send no
    messages, so their matrices hold point-to-point traffic only.

    Returns:
        Run name -> sum of |delta| over (src, dst) pairs divided by the
        baseline's total, in percent. Pairs only one side has count fully.
    """
    traffic = comparison.traffic
    if traffic.is_empty():
        return {}
    base_name = comparison.baseline.name
    base_total = float(
        traffic.filter(pl.col("run") == base_name)["bytes"].sum() or 0.0
    )
    out: dict[str, float] = {}
    for r in comparison.results[1:]:
        if r.level == "analytical":
            continue
        mine = traffic.filter(pl.col("run") == r.name).select(
            "src", "dst", "bytes"
        )
        theirs = traffic.filter(pl.col("run") == base_name).select(
            "src", "dst", pl.col("bytes").alias("base")
        )
        diff = mine.join(
            theirs, on=["src", "dst"], how="full", coalesce=True
        ).select(
            (
                pl.col("bytes").fill_null(0).cast(pl.Int64)
                - pl.col("base").fill_null(0).cast(pl.Int64)
            )
            .abs()
            .sum()
        )
        out[r.name] = (
            100.0 * float(diff.item()) / base_total if base_total else 0.0
        )
    return out


def summarize(comparison: SimComparison) -> Mapping[str, Any]:
    """Returns the comparison's key findings as plain data (for JSON)."""
    return {
        "baseline": comparison.baseline.name,
        "runs": comparison.overview.to_dicts(),
        "attribution": comparison.attribution.to_dicts(),
        "traffic_mismatch_pct": traffic_mismatch(comparison),
        "issues": comparison.issues.to_dicts(),
    }
