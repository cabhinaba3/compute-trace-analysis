"""Compares simulation runs of GOAL and Chakra traces, in any mix.

Every run is compared with the first. When a graph with collectives meets one
without (Chakra against GOAL), it runs both lowered (like for like with GOAL)
and analytical; the gap between those is what contention and serialisation
add to the collectives' analytical costs.

Tables: ``overview`` (one row per run), ``breakdown`` (critical path per run
and segment), ``attribution`` (each run's makespan gap to the baseline split
by critical-path segment; the deltas add up to the gap exactly, as every
critical path tiles [0, makespan]), ``ranks``, ``phases``
(``cta.sim.alignment``) and ``traffic`` (bytes per rank pair).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import pathlib
from typing import Any, cast

import polars as pl

from cta import models
from cta.sim import alignment
from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib
from cta.sim import result as result_lib

LEVEL_CHOICES = ("auto", "both", "lowered", "analytical")
# The simulation aspect's headline metrics: name, unit, lower is better.
METRICS = (
    ("makespan_ms", "ms", True),
    ("cp_compute_pct", "%", False),
    ("cp_comm_pct", "%", True),
    ("cp_contention_pct", "%", True),
    ("rank_finish_spread_ms", "ms", True),
    ("mean_utilization", "", False),
    ("wire_gb", "GB", True),
)
TABLES = (
    "overview",
    "breakdown",
    "attribution",
    "ranks",
    "phases",
    "traffic",
    "issues",
)


@dataclasses.dataclass(slots=True)
class SimComparison:
    """The runs, baseline first, and the tables comparing them."""

    results: list[result_lib.SimResult]
    overview: pl.DataFrame
    breakdown: pl.DataFrame
    attribution: pl.DataFrame
    ranks: pl.DataFrame
    phases: pl.DataFrame
    traffic: pl.DataFrame
    issues: pl.DataFrame

    def tables(self) -> dict[str, pl.DataFrame]:
        """Every table by name."""
        return {name: getattr(self, name) for name in TABLES}

    def write(
        self, out_dir: str | pathlib.Path, *, csv: bool = False
    ) -> list[pathlib.Path]:
        """Writes the non-empty tables; returns the files."""
        return result_lib.write_tables(self.tables(), out_dir, csv=csv)

    def traffic_mismatch(self) -> dict[str, float]:
        """Per run, how much its rank-pair traffic differs, in % of baseline.

        Analytical runs are left out: their collectives send no messages.
        """
        if self.traffic.is_empty():
            return {}
        base = self.traffic.filter(pl.col("run") == self.results[0].name)
        total = float(cast(Any, base["bytes"].sum()))
        return {
            r.name: 100
            * float(
                self.traffic.filter(pl.col("run") == r.name)["delta_bytes"]
                .abs()
                .sum()
            )
            / max(total, 1)
            for r in self.results[1:]
            if r.level != "analytical"
        }

    def to_report(self) -> models.Report:
        """A Report with one group per run and a ``simulation`` aspect."""
        names = self.overview["run"].to_list()
        rows = []
        for metric, unit, lower_is_better in METRICS:
            values = self.overview[metric].to_list()
            rows.append(
                {
                    "metric": metric,
                    "unit": unit,
                    **dict(zip(names, values, strict=True)),
                    "lower_is_better": lower_is_better,
                }
            )
        aspect = models.AspectResult(
            name="simulation",
            title="Simulation",
            summary=pl.DataFrame(rows),
            details=self.tables(),
            scalars={
                m: dict(zip(names, self.overview[m].to_list(), strict=True))
                for m, _, _ in METRICS
            },
        )
        return models.Report(
            mode="single"
            if len(names) == 1
            else "pair"
            if len(names) == 2
            else "multi",
            groups=tuple(
                models.TrialGroup(group_id=n, label=n, trials=()) for n in names
            ),
            aspects={"simulation": aspect},
        )


def run_sources(
    sources: Sequence[str | pathlib.Path | graph_lib.SimGraph],
    settings: config_lib.SimConfig,
    *,
    labels: Sequence[str] | None = None,
    levels: str = "auto",
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> list[result_lib.SimResult]:
    """Simulates each source; ``levels``: see LEVEL_CHOICES and the module."""
    if levels not in LEVEL_CHOICES:
        raise ValueError(f"levels must be one of {', '.join(LEVEL_CHOICES)}")
    graphs = []
    for i, source in enumerate(sources):
        label = labels[i] if labels and i < len(labels) else None
        if isinstance(source, graph_lib.SimGraph):
            graphs.append(
                dataclasses.replace(source, label=label or source.label)
            )
        else:
            graphs.append(
                graph_lib.load_graph(
                    source,
                    label=label,
                    groups=settings["collectives.groups"],
                    world_size=settings["world_size"],
                    cache_dir=cache_dir,
                )
            )
    if len({g.label for g in graphs}) < len(graphs):
        graphs = [
            dataclasses.replace(g, label=f"{g.label}_{i}")
            for i, g in enumerate(graphs)
        ]
    mixed = len({g.has_collectives for g in graphs}) == 2
    both = levels == "both" or (levels == "auto" and mixed)
    chosen = (
        ("lowered", "analytical")
        if both
        else (None if levels == "auto" else levels,)
    )
    return [
        engine.run_graph(g, settings, level=level)
        for g in graphs
        for level in (chosen if g.has_collectives else (None,))
    ]


def compare(
    sources: Sequence[str | pathlib.Path | graph_lib.SimGraph],
    settings: config_lib.SimConfig | None = None,
    *,
    labels: Sequence[str] | None = None,
    levels: str = "auto",
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> SimComparison:
    """Simulates the sources and compares every run with the first."""
    settings = settings or config_lib.SimConfig()
    return compare_results(
        run_sources(
            sources, settings, labels=labels, levels=levels, cache_dir=cache_dir
        )
    )


def _per_run(
    results: Sequence[result_lib.SimResult],
    table: Callable[[result_lib.SimResult], pl.DataFrame],
) -> pl.DataFrame:
    frames = [
        t.select(pl.lit(r.name).alias("run"), pl.all())
        for r in results
        if not (t := table(r)).is_empty()
    ]
    return (
        pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    )


def compare_results(
    results: Sequence[result_lib.SimResult], *, tolerance: float = 0.01
) -> SimComparison:
    """Compares runs (which kept their op tables) with the first."""
    results = list(results)
    base = results[0]
    breakdown = _per_run(results, lambda r: r.breakdown())
    gap = pl.col("run").replace_strict(
        {r.name: r.makespan_ns - base.makespan_ns for r in results},
        return_dtype=pl.Int64,
    )
    attribution = (
        breakdown.filter(pl.col("run") != base.name)
        .join(
            base.breakdown().select(
                "segment", pl.col("ns").alias("baseline_ns")
            ),
            on="segment",
        )
        .select(
            "run",
            "segment",
            "baseline_ns",
            pl.col("ns").alias("run_ns"),
            delta_ns=pl.col("ns") - pl.col("baseline_ns"),
        )
        .with_columns(
            share_of_gap_pct=pl.when(gap != 0)
            .then(100 * pl.col("delta_ns") / gap)
            .otherwise(0.0)
        )
    )
    ranks = _per_run(results, lambda r: r.ranks).join(
        base.ranks.select(
            "rank", pl.col("finish_ns").alias("baseline_finish_ns")
        ),
        on="rank",
    )
    pairs = _per_run(
        results,
        lambda r: (
            r.messages.group_by("src", "dst").agg(pl.col("bytes").sum())
            if not r.messages.is_empty()
            else pl.DataFrame()
        ),
    )
    traffic = pl.DataFrame()
    if not pairs.is_empty():
        base_pairs = pairs.filter(pl.col("run") == base.name).select(
            "src", "dst", pl.col("bytes").alias("baseline_bytes")
        )
        traffic = pl.concat(
            [
                pairs.filter(pl.col("run") == r.name)
                .drop("run")
                .join(base_pairs, on=["src", "dst"], how="full", coalesce=True)
                .select(pl.lit(r.name).alias("run"), pl.all())
                for r in results
            ]
        ).with_columns(
            pl.col("bytes", "baseline_bytes").fill_null(0),
            delta_bytes=pl.col("bytes").fill_null(0).cast(pl.Int64)
            - pl.col("baseline_bytes").fill_null(0).cast(pl.Int64),
        )
    return SimComparison(
        results=results,
        overview=pl.DataFrame([_overview_row(r, base) for r in results]),
        breakdown=breakdown,
        attribution=attribution,
        ranks=ranks.with_columns(
            delta_ns=pl.col("finish_ns") - pl.col("baseline_finish_ns")
        ),
        phases=_phases(results, tolerance),
        traffic=traffic.sort("run", "src", "dst")
        if not traffic.is_empty()
        else traffic,
        issues=_per_run(results, lambda r: r.issues),
    )


def _overview_row(
    r: result_lib.SimResult, base: result_lib.SimResult
) -> dict[str, Any]:
    s = r.summary()
    gap = r.makespan_ns - base.makespan_ns
    return {
        "run": r.name,
        "format": r.trace_format,
        "level": r.level,
        "model": r.config["model"],
        "completed": r.completed,
        "makespan_ms": s["makespan_ms"],
        "delta_ms": gap / 1e6,
        "delta_pct": 100 * gap / max(base.makespan_ns, 1),
        **{
            f"cp_{k}_pct": s[f"cp_{k}_pct"]
            for k in ("compute", "comm", "contention", "collective")
        },
        "rank_finish_spread_ms": s["rank_finish_spread_ns"] / 1e6,
        "mean_utilization": s["mean_utilization"],
        "compute_total_ms": s["total_compute_ns"] / 1e6,
        "ops": int(s["ops"]),
        "messages": int(s["total_messages"]),
        "wire_gb": s["total_bytes_sent"] / 1e9,
        "wall_s": s["wall_seconds"],
    }


def _phases(
    results: Sequence[result_lib.SimResult], tolerance: float
) -> pl.DataFrame:
    base = results[0]
    if base.ops.is_empty():
        return pl.DataFrame()
    base_anchors = alignment.anchors(base.ops)
    frames = []
    for r in results[1:]:
        matched = alignment.match(
            base_anchors, alignment.anchors(r.ops), tolerance=tolerance
        )
        mine = alignment.phases(
            base.ops,
            base.ranks,
            matched.select("rank", "phase", pl.col("op_base").alias("op")),
        )
        theirs = alignment.phases(
            r.ops,
            r.ranks,
            matched.select("rank", "phase", pl.col("op_other").alias("op")),
        )
        frames.append(
            mine.join(theirs, on=["rank", "phase"], suffix="_run").select(
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
                (
                    pl.col("bytes_run").cast(pl.Int64)
                    - pl.col("bytes").cast(pl.Int64)
                ).alias("bytes_delta"),
                pl.col("collectives").alias("baseline_collectives"),
                pl.col("collectives_run").alias("run_collectives"),
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()
