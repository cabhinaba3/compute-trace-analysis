"""The engine: ingests trial groups, runs the aspects, gates and diagnoses.

``run_comparison`` turns pre-built groups and a config into a report; the
entry points in ``cta.api`` build those from paths.
"""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
from typing import Literal

from cta import advisor
from cta import diagnosis
from cta import gating
from cta import models
from cta import plugins
from cta.ingest import cache


def ingest_group_tables(
    groups: Sequence[models.TrialGroup],
    store: cache.CacheStore,
    config: models.AnalysisConfig | None = None,
) -> dict[str, list[dict[str, pathlib.Path]]]:
    """Ingests (or fetches from the cache) every artifact of every group.

    Each trial's tables from all of its formats are merged into one map.
    Every table appears under ``<format>.<table>`` and under its bare name;
    when several formats have a table with the same bare name, the first
    format's table keeps it. When streaming Parquet bypass is active for a
    distributed execution graph, full ``nodes``/``edges`` Parquet files are
    skipped and a ``.cta_graph`` cache is built on demand instead.

    Args:
        groups: The trial groups to ingest.
        store: The cache that holds the ingested Parquet tables.
        config: Optional analysis config controlling streaming Parquet bypass.

    Returns:
        Group id -> one map of table name -> cached path per trial.
    """
    from cta.analysis import streaming

    tables_by_group: dict[str, list[dict[str, pathlib.Path]]] = {}
    for group in groups:
        trial_tables: list[dict[str, pathlib.Path]] = []
        for trial in group.trials:
            merged: dict[str, pathlib.Path] = {}
            for trace_format, artifact in trial.artifacts.items():
                if config is not None and streaming.should_bypass_parquet(
                    artifact, config
                ):
                    tier = streaming.resolve_streaming_tier(artifact, config)
                    if tier == "exact" and config.graph_engine == "gridgraph":
                        gc_path = store.get_or_build_graph_cache(
                            artifact,
                            goal_semantics=config.goal_semantics,
                            collective_sync=config.collective_sync,
                            memory_budget_bytes=config.memory_budget_bytes,
                            fail_on_cycle=config.fail_on_cycle,
                            scratch_dir=str(config.scratch_dir)
                            if config.scratch_dir
                            else "",
                        )
                        merged[f"{trace_format}.graph_cache"] = gc_path
                        merged.setdefault("graph_cache", gc_path)
                    merged[f"{trace_format}.streaming"] = artifact.paths[0]
                    continue
                adapter = plugins.ADAPTERS[trace_format]
                format_tables = store.get_or_ingest(artifact, adapter)
                for table_name, table_path in format_tables.items():
                    merged[f"{trace_format}.{table_name}"] = table_path
                    merged.setdefault(table_name, table_path)
            trial_tables.append(merged)
        tables_by_group[group.group_id] = trial_tables
    return tables_by_group


def select_aspects(
    common_formats: set[str], requested: tuple[str, ...] | None
) -> list[str]:
    """Returns the registered aspects to run, in registration order.

    An aspect applies when the common formats include all of its required
    formats, or any of them for an ``any_of_formats`` aspect. Without a
    request, only default-enabled aspects are selected.

    Raises:
        ValueError: A requested aspect is unknown or does not apply.
    """
    selected: list[str] = []
    for name, aspect in plugins.ASPECTS.items():
        if requested is not None and name not in requested:
            continue
        if requested is None and not aspect.default_enabled:
            continue
        required = set(aspect.required_formats)
        is_applicable = (
            bool(common_formats & required)
            if aspect.any_of_formats
            else required.issubset(common_formats)
        )
        if is_applicable:
            selected.append(name)
    if requested is not None:
        missing = [name for name in requested if name not in selected]
        if missing:
            raise ValueError(
                f"Requested aspects {missing} not applicable to "
                f"common formats {sorted(common_formats)}"
            )
    return selected


def run_comparison(
    groups: Sequence[models.TrialGroup],
    config: models.AnalysisConfig | None = None,
) -> models.Report:
    """Analyses one group or compares several groups.

    Ingests every artifact, runs each applicable aspect and keeps the
    results that have a non-empty summary or any details table. Then it
    diagnoses bottleneck regimes, recommends optimizations and, with two or
    more groups, evaluates the regression gates.

    Args:
        groups: The groups to analyse; the first one is the baseline.
        config: Analysis options; None uses the defaults.

    Returns:
        The report.

    Raises:
        ValueError: There are no groups, the groups share no trace format,
            a requested aspect does not apply, or a gate metric is not
            reported and ``config.allow_missing_gates`` is False.
    """
    if not groups:
        raise ValueError("Analysis requires at least 1 trace group")
    config = config or models.AnalysisConfig()
    group_tuple = tuple(groups)
    common_formats = set.intersection(*(g.formats for g in group_tuple))
    if not common_formats:
        formats_by_group = {
            group.label: sorted(group.formats) for group in group_tuple
        }
        raise ValueError(
            "Traces share no supported/common trace format. "
            f"Found: {formats_by_group}"
        )

    store = cache.CacheStore(config.cache_dir)
    tables = ingest_group_tables(group_tuple, store, config=config)
    aspect_names = select_aspects(common_formats, config.aspects)
    if config.streaming == "aggregate":
        if config.aspects is None:
            aspect_names = [
                name
                for name in aspect_names
                if name not in ("critical_path", "whatif")
            ]
        elif any(name in ("critical_path", "whatif") for name in aspect_names):
            raise ValueError(
                "Aspects 'critical_path' and 'whatif' require DAG traversal "
                "and cannot run under streaming='aggregate'. Use "
                "streaming='frontier', 'windowed', 'exact', or 'auto'."
            )

    results: dict[str, models.AspectResult] = {}
    for name in aspect_names:
        result = plugins.ASPECTS[name].run(group_tuple, tables, config)
        if not result.summary.is_empty() or result.details:
            results[name] = result

    mode: Literal["single", "pair", "multi"]
    if len(group_tuple) == 1:
        mode = "single"
    elif len(group_tuple) == 2:
        mode = "pair"
    else:
        mode = "multi"
    diagnoses = diagnosis.diagnose_bottleneck_shifts(group_tuple, results)
    recommendations = advisor.advise(group_tuple, tables, config, results)
    violations: list[models.GateViolation] = []
    if len(group_tuple) >= 2:
        missing = gating.missing_gate_metrics(results, config.gates)
        if missing and not config.allow_missing_gates:
            raise ValueError(
                f"Gate metrics not found in report: {missing} "
                "(typo, or the aspect that reports them did not run). "
                "Use --allow-missing-gates / allow_missing_gates=True "
                "to ignore."
            )
        violations = gating.evaluate_gates(results, config.gates)
    return models.Report(
        mode=mode,
        groups=group_tuple,
        aspects=results,
        diagnoses=diagnoses,
        recommendations=recommendations,
        violations=violations,
        sources=tables,
        config=config,
    )
