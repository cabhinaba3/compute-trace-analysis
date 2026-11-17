"""The ``simulation`` aspect, off by default.

Simulates each group's GOAL or Chakra trials with the default settings.
"""

from __future__ import annotations

import pathlib

import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats
from cta.sim import compare
from cta.sim import engine
from cta.sim import graph as graph_lib

_METRICS = [
    ("makespan_ms", "ms", True),
    ("cp_compute_pct", "%", False),
    ("cp_comm_pct", "%", True),
    ("cp_contention_pct", "%", True),
]


def _graph(
    table_paths: dict[str, pathlib.Path], label: str
) -> graph_lib.SimGraph | None:
    for trace_format in graph_lib.GRAPH_FORMATS:
        if f"{trace_format}.nodes" in table_paths:
            return graph_lib.from_tables(
                pl.read_parquet(table_paths[f"{trace_format}.nodes"]),
                pl.read_parquet(table_paths[f"{trace_format}.edges"]),
                trace_format=trace_format,
                label=label,
            )
    return None


@plugins.register_aspect(
    "simulation",
    title="Discrete-Event Simulation",
    required_formats=graph_lib.GRAPH_FORMATS,
    any_of_formats=True,
    default_enabled=False,
)
class SimulationAspect(plugins.Aspect):
    """Simulated metrics per group, and ``cta.sim.compare``'s tables.

    The tables compare each group's first trial.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
    ) -> models.AspectResult:
        """Simulates every trial."""
        trials, firsts = [], []
        for group in groups:
            values: dict[str, list[float]] = {m: [] for m, _, _ in _METRICS}
            for i, table_paths in enumerate(tables[group.group_id]):
                if (graph := _graph(table_paths, group.label)) is None:
                    continue
                result = engine.run_graph(graph, keep_ops=i == 0)
                for metric, _, _ in _METRICS:
                    values[metric].append(result.summary()[metric])
                if i == 0:
                    firsts.append(result)
            trials.append(values)
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=stats.compare_scalar_metrics(
                [(f"sim_{m}", u, low) for m, u, low in _METRICS],
                [g.label for g in groups],
                [{f"sim_{k}": v for k, v in t.items()} for t in trials],
                alpha=config.alpha,
            ),
            details=compare.compare_results(firsts).tables() if firsts else {},
        )
