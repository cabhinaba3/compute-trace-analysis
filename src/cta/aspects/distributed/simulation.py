"""The ``simulation`` aspect: simulates each group's GOAL or Chakra graph.

Off by default (``--aspects simulation`` turns it on), because a large graph
takes a while to simulate. Every trial runs with the default settings
(LogGOPSim's parameters, collectives lowered); ``cta simulate compare``
offers the other models and settings. The summary compares the simulated
makespan and critical-path shares across groups; the details are the tables
of ``cta.sim.compare`` for each group's first trial.
"""

from __future__ import annotations

import pathlib

import polars as pl

from cta import models
from cta import plugins
from cta.analysis import stats
from cta.sim import compare
from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib
from cta.sim import result as result_lib

_METRICS = [
    ("sim_makespan_ms", "ms", True),
    ("sim_cp_compute_pct", "%", False),
    ("sim_cp_comm_pct", "%", True),
    ("sim_cp_contention_pct", "%", True),
    ("sim_rank_finish_spread_ms", "ms", True),
]


def _trial_graph(
    table_paths: dict[str, pathlib.Path], label: str
) -> graph_lib.SimGraph | None:
    """Builds the graph of the first GOAL or Chakra trace of a trial."""
    for trace_format in graph_lib.GRAPH_FORMATS:
        nodes = table_paths.get(f"{trace_format}.nodes")
        edges = table_paths.get(f"{trace_format}.edges")
        if nodes is not None and edges is not None:
            return graph_lib.from_tables(
                pl.read_parquet(nodes),
                pl.read_parquet(edges),
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
    """Simulates every trial and compares the groups' simulated runs.

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: GOAL or Chakra.
        any_of_formats: One of the formats is enough.
        default_enabled: Off unless requested.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
    ) -> models.AspectResult:
        """Simulates each trial; compares the groups.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings (the significance level).

        Returns:
            The summary metrics, and the simulation comparison tables of the
            groups' first trials as details.
        """
        settings = config_lib.SimConfig()
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        firsts: list[result_lib.SimResult] = []
        for group in groups:
            values: dict[str, list[float]] = {m: [] for m, _, _ in _METRICS}
            for i, table_paths in enumerate(tables[group.group_id]):
                graph = _trial_graph(table_paths, group.label)
                if graph is None:
                    continue
                result = engine.run_graph(graph, settings, keep_ops=i == 0)
                summary = result.summary()
                values["sim_makespan_ms"].append(summary["makespan_ms"])
                values["sim_cp_compute_pct"].append(summary["cp_compute_pct"])
                values["sim_cp_comm_pct"].append(summary["cp_comm_pct"])
                values["sim_cp_contention_pct"].append(
                    summary["cp_contention_pct"]
                )
                values["sim_rank_finish_spread_ms"].append(
                    summary.get("rank_finish_spread_ns", 0.0) / 1e6
                )
                if i == 0:
                    firsts.append(result)
            trials.append(values)
        summary_table = stats.compare_scalar_metrics(
            _METRICS, labels, trials, alpha=config.alpha
        )
        details = compare.compare_results(firsts).tables() if firsts else {}
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary_table,
            details=details,
        )
