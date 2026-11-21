"""What-if prediction aspect (Chakra and GOAL execution graphs)."""

from __future__ import annotations

import pathlib

import polars as pl

from cta import models
from cta import plugins
from cta.analysis import interconnect
from cta.analysis import stats
from cta.analysis import streaming
from cta.analysis import whatif
from cta.aspects.distributed import graph_tables
from cta.ingest import cache


@plugins.register_aspect(
    "whatif",
    title="What-If Prediction",
    required_formats=graph_tables.FORMATS,
    any_of_formats=True,
    default_enabled=False,
)
class WhatIfAspect(plugins.Aspect):
    """Predicts a run from the baseline and scores the prediction.

    The scenario in ``AnalysisConfig.predict`` (for example ``"comm:1.6x"``)
    is applied to the baseline's execution graph, and the predicted critical
    path is compared with the measured one of every candidate. Gate the
    metric ``whatif_residual`` to fail when reality is slower than predicted.
    The baseline's op families are also swept: how much faster would the run
    get if each one were faster?

    Attributes:
        name: Aspect key used by ``--aspects`` and in reports.
        title: Report heading.
        required_formats: Formats that provide an execution graph.
        any_of_formats: Whether one of ``required_formats`` is enough.
    """

    def run(
        self,
        groups: tuple[models.TrialGroup, ...],
        tables: dict[str, list[dict[str, pathlib.Path]]],
        config: models.AnalysisConfig,
    ) -> models.AspectResult:
        """Predicts the candidates from the baseline and sweeps the families.

        Args:
            groups: Trial groups to compare, baseline first.
            tables: Per group id, one mapping of table key to Parquet path
                per trial.
            config: Comparison settings; ``predict`` and the significance
                level are used.

        Returns:
            Summary metrics ``whatif_makespan_us`` (the critical path of each
            group) and, with a prediction, ``whatif_residual`` (the
            baseline column is the predicted critical path, the candidate
            columns the measured ones, so ``delta_pct`` is how much slower
            than predicted the run was). Detail tables ``sweep`` and
            ``ceiling`` come from the first trial of the baseline.

        Raises:
            ValueError: ``config.predict`` is malformed.
        """
        scenario = (
            whatif.parse_scenario(config.predict, label="predicted")
            if config.predict
            else None
        )
        new_net = (
            interconnect.parse(config.interconnect)
            if config.interconnect
            else None
        )
        base_net = (
            interconnect.parse(config.baseline_interconnect)
            if config.baseline_interconnect
            else None
        )
        if scenario is None and new_net is not None:
            scenario = whatif.Scenario("repriced", ())
        labels = [group.label for group in groups]
        trials: list[dict[str, list[float]]] = []
        details: dict[str, pl.DataFrame] = {}
        for index, group in enumerate(groups):
            values: dict[str, list[float]] = {
                "whatif_makespan_us": [],
                "whatif_residual": [],
            }
            store = cache.CacheStore(config.cache_dir)
            for trial_index, (trial, table_paths) in enumerate(
                zip(group.trials, tables[group.group_id], strict=True)
            ):
                artifact = streaming.pick_graph_artifact(trial)
                has_parquet = any(
                    f"{fmt}.nodes" in table_paths
                    for fmt in graph_tables.FORMATS
                )
                use_stream = artifact is not None and (
                    not has_parquet
                    or streaming.resolve_streaming_tier(artifact, config)
                    != "off"
                )
                model: whatif.WhatIfModel | whatif.StreamingWhatIfModel
                if use_stream and artifact is not None:
                    gc_path = table_paths.get(
                        f"{artifact.format}.graph_cache"
                    ) or table_paths.get("graph_cache")
                    if gc_path is None:
                        gc_path = store.graph_cache_path(
                            artifact,
                            goal_semantics=config.goal_semantics,
                            collective_sync=config.collective_sync,
                        )
                    model = whatif.build_streaming_model_from_artifact(
                        artifact,
                        cache_path=gc_path,
                        goal_semantics=config.goal_semantics,
                        collective_sync=config.collective_sync,
                        memory_budget_bytes=config.memory_budget_bytes,
                        scratch_dir=str(config.scratch_dir)
                        if config.scratch_dir
                        else "",
                    )
                else:
                    nodes, edges, _ = graph_tables.read_graph_tables(
                        table_paths
                    )
                    model = whatif.build_model(nodes, edges)
                is_baseline = index == 0
                # With no new interconnect the baseline and the scenario
                # share one solve; with one, the scenario runs on the
                # repriced graph and the baseline's own makespan stays the
                # measured one.
                reprices = is_baseline and new_net is not None
                run = whatif.run_scenarios(
                    model,
                    [scenario]
                    if scenario and is_baseline and not reprices
                    else [],
                    memory_budget_bytes=config.memory_budget_bytes,
                )
                makespan_us = run.table["makespan_ns"].to_numpy() / 1e3
                values["whatif_makespan_us"].append(float(makespan_us[0]))
                if scenario is None:
                    pass
                elif reprices:
                    assert new_net is not None
                    predicted = whatif.run_scenarios(
                        whatif.reprice(model, new_net, base_net),
                        [scenario],
                        memory_budget_bytes=config.memory_budget_bytes,
                    )
                    predicted_us = (
                        predicted.table["makespan_ns"].to_numpy() / 1e3
                    )
                    values["whatif_residual"].append(float(predicted_us[1]))
                    if trial_index == 0:
                        details["predicted"] = predicted.table
                elif is_baseline:
                    values["whatif_residual"].append(float(makespan_us[1]))
                    if trial_index == 0:
                        details["predicted"] = run.table
                else:
                    values["whatif_residual"].append(float(makespan_us[0]))
                if is_baseline and trial_index == 0:
                    result = whatif.sweep(
                        model, memory_budget_bytes=config.memory_budget_bytes
                    )
                    details["sweep"] = result.table
                    details["ceiling"] = result.ceiling
            trials.append(values)
        summary = stats.compare_scalar_metrics(
            [
                ("whatif_makespan_us", "us", True),
                ("whatif_residual", "us", True),
            ],
            labels,
            trials,
            alpha=config.alpha,
        )
        return models.AspectResult(
            name=self.name,
            title=self.title,
            summary=summary,
            details=details,
        )
