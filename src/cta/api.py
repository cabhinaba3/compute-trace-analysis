"""SDK entry points: analyse one input, compare several, inspect a trace.

Each input is a trace file, directory, glob, run bundle, trial group or a
list of trials (see ``cta.ingest.discovery.resolve_trial_group``). The
functions resolve the inputs, build an ``AnalysisConfig`` from their keyword
arguments and hand both to ``cta.engine.run_comparison``.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
from typing import Any

import polars as pl

from cta import engine
from cta import models
from cta import plugins
from cta.ingest import cache
from cta.ingest import discovery


def _config(
    *,
    aspects: Sequence[str] | None,
    warmup_steps: int,
    max_steps: int | None,
    collective_algorithm: str,
    alpha: float,
    scope_depth: int | None,
    ranks: Sequence[int] | None,
    devices: Sequence[int] | None,
    predict: str | None,
    interconnect: str | None,
    baseline_interconnect: str | None,
    cache_dir: str | pathlib.Path,
    gates: dict[str, float] | None = None,
    allow_missing_gates: bool = False,
    streaming: str = "auto",
    graph_engine: str = "gridgraph",
    quantile_sketch: str = "ddsketch",
    memory_budget_bytes: int = 0,
    collective_sync: str = "none",
    goal_semantics: str = "cta_legacy",
    window_size: int = 262144,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    delay_attribution: bool = False,
    scratch_dir: str | pathlib.Path | None = None,
    parquet_bypass_bytes: int = 2 * 1024**3,
) -> models.AnalysisConfig:
    """Builds the config; empty sequences mean "all", like None."""
    return models.AnalysisConfig(
        aspects=tuple(aspects) if aspects else None,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gates=dict(gates or {}),
        allow_missing_gates=allow_missing_gates,
        collective_algorithm=collective_algorithm,
        alpha=alpha,
        scope_depth=scope_depth,
        ranks=tuple(ranks) if ranks else None,
        devices=tuple(devices) if devices else None,
        predict=predict,
        interconnect=interconnect,
        baseline_interconnect=baseline_interconnect,
        cache_dir=pathlib.Path(cache_dir),
        streaming=streaming,
        graph_engine=graph_engine,
        quantile_sketch=quantile_sketch,
        memory_budget_bytes=memory_budget_bytes,
        collective_sync=collective_sync,
        goal_semantics=goal_semantics,
        window_size=window_size,
        fail_on_cycle=fail_on_cycle,
        extract_critical_path_ops=extract_critical_path_ops,
        delay_attribution=delay_attribution,
        scratch_dir=pathlib.Path(scratch_dir) if scratch_dir else None,
        parquet_bypass_bytes=parquet_bypass_bytes,
    )


def _with_distinct_labels(
    groups: list[models.TrialGroup],
) -> list[models.TrialGroup]:
    """Appends each group's position to every label if any two collide.

    The groups are copied, not modified: callers may reuse their own.
    """
    labels = [group.label for group in groups]
    if len(set(labels)) == len(labels):
        return groups
    return [
        dataclasses.replace(
            group, label=f"{group.label}_{i}", group_id=f"{group.group_id}_{i}"
        )
        for i, group in enumerate(groups)
    ]


def analyze(
    source: discovery.TrialSource,
    *,
    aspects: Sequence[str] | None = None,
    warmup_steps: int = 0,
    max_steps: int | None = None,
    collective_algorithm: str = "ring",
    alpha: float = 0.05,
    scope_depth: int | None = None,
    ranks: Sequence[int] | None = None,
    devices: Sequence[int] | None = None,
    predict: str | None = None,
    interconnect: str | None = None,
    baseline_interconnect: str | None = None,
    cache_dir: str | pathlib.Path = ".cta_cache",
    label: str | None = None,
    streaming: str = "auto",
    graph_engine: str = "gridgraph",
    quantile_sketch: str = "ddsketch",
    memory_budget_bytes: int = 0,
    collective_sync: str = "none",
    goal_semantics: str = "cta_legacy",
    window_size: int = 262144,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    delay_attribution: bool = False,
    scratch_dir: str | pathlib.Path | None = None,
    parquet_bypass_bytes: int = 2 * 1024**3,
) -> models.Report:
    """Analyses a single trace, run bundle directory or multi-trial group.

    Args:
        source: The trace(s) to analyse.
        aspects: Names of the aspects to run; None or empty runs every
            default-enabled aspect that applies.
        warmup_steps: Number of leading steps to drop as warm-up.
        max_steps: Maximum number of steps to keep after warm-up.
        collective_algorithm: Collective algorithm assumed for collectives.
        alpha: Significance level of the hypothesis tests.
        scope_depth: Scope level kernels are attributed to (see
            ``AnalysisConfig``).
        ranks: Analyse only these ranks; None or empty means all.
        devices: Analyse only these GPU devices; None or empty means all.
        predict: A what-if scenario for the ``whatif`` aspect (see
            ``AnalysisConfig``).
        interconnect: Interconnect the ``whatif`` aspect prices the baseline's
            communication on (see ``AnalysisConfig``).
        baseline_interconnect: The interconnect the baseline was measured on.
        cache_dir: Directory of the Parquet ingest cache.
        label: Label of the group; defaults to the bundle's label.
        streaming: Execution-graph streaming mode (``auto``, ``exact``,
            ``frontier``, ``windowed``, ``aggregate``, ``off``).
        graph_engine: Out-of-core / streaming DAG engine (``gridgraph``,
            ``graphchi``, ``xstream``, ``frontier``).
        quantile_sketch: Streaming quantile sketch (``ddsketch``, ``kll``,
            ``gk``, ``exact``).
        memory_budget_bytes: RAM budget in bytes (0 = auto 60% free).
        collective_sync: Cross-rank collective sync (``none`` or ``barrier``).
        goal_semantics: GOAL interpretation (``cta_legacy`` or ``strict``).
        window_size: Frontier / windowed cache size in nodes.
        fail_on_cycle: Raise ``RuntimeError`` when a DAG cycle is detected.
        extract_critical_path_ops: Include exact critical-path op chain table.
        delay_attribution: Enable Scalasca backward wait-state attribution.
        scratch_dir: Optional directory for out-of-core scratch files.
        parquet_bypass_bytes: Raw size threshold for Parquet bypass in auto.

    Returns:
        The report, in ``"single"`` mode.

    Raises:
        FileNotFoundError: A trace path does not exist.
        ValueError: The source holds no supported trace, or a requested
            aspect does not apply.
    """
    group = discovery.resolve_trial_group(source, label=label)
    config = _config(
        aspects=aspects,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        collective_algorithm=collective_algorithm,
        alpha=alpha,
        scope_depth=scope_depth,
        ranks=ranks,
        devices=devices,
        predict=predict,
        interconnect=interconnect,
        baseline_interconnect=baseline_interconnect,
        cache_dir=cache_dir,
        streaming=streaming,
        graph_engine=graph_engine,
        quantile_sketch=quantile_sketch,
        memory_budget_bytes=memory_budget_bytes,
        collective_sync=collective_sync,
        goal_semantics=goal_semantics,
        window_size=window_size,
        fail_on_cycle=fail_on_cycle,
        extract_critical_path_ops=extract_critical_path_ops,
        delay_attribution=delay_attribution,
        scratch_dir=scratch_dir,
        parquet_bypass_bytes=parquet_bypass_bytes,
    )
    return engine.run_comparison((group,), config)


def compare(
    baseline: discovery.TrialSource,
    candidate: discovery.TrialSource,
    *others: discovery.TrialSource,
    aspects: Sequence[str] | None = None,
    warmup_steps: int = 0,
    max_steps: int | None = None,
    gates: dict[str, float] | None = None,
    allow_missing_gates: bool = False,
    collective_algorithm: str = "ring",
    alpha: float = 0.05,
    scope_depth: int | None = None,
    ranks: Sequence[int] | None = None,
    devices: Sequence[int] | None = None,
    predict: str | None = None,
    interconnect: str | None = None,
    baseline_interconnect: str | None = None,
    cache_dir: str | pathlib.Path = ".cta_cache",
    labels: Sequence[str] | None = None,
    streaming: str = "auto",
    graph_engine: str = "gridgraph",
    quantile_sketch: str = "ddsketch",
    memory_budget_bytes: int = 0,
    collective_sync: str = "none",
    goal_semantics: str = "cta_legacy",
    window_size: int = 262144,
    fail_on_cycle: bool = False,
    extract_critical_path_ops: bool = False,
    delay_attribution: bool = False,
    scratch_dir: str | pathlib.Path | None = None,
    parquet_bypass_bytes: int = 2 * 1024**3,
) -> models.Report:
    """Compares a baseline with one or more candidates.

    If any two inputs end up with the same label, every group's label and id
    get its position appended (``<label>_<i>``).

    Args:
        baseline: The baseline trace(s).
        candidate: The first candidate trace(s).
        *others: Further candidates.
        aspects: Names of the aspects to run; None or empty runs every
            default-enabled aspect that applies.
        warmup_steps: Number of leading steps to drop as warm-up.
        max_steps: Maximum number of steps to keep after warm-up.
        gates: Regression gates: metric name -> allowed regression in %.
        allow_missing_gates: If True, gate metrics that no aspect reports
            are ignored instead of raising an error.
        collective_algorithm: Collective algorithm assumed for collectives.
        alpha: Significance level of the hypothesis tests.
        scope_depth: Scope level kernels are attributed to (see
            ``AnalysisConfig``).
        ranks: Analyse only these ranks; None or empty means all.
        devices: Analyse only these GPU devices; None or empty means all.
        predict: A what-if scenario for the ``whatif`` aspect (see
            ``AnalysisConfig``).
        interconnect: Interconnect the ``whatif`` aspect prices the baseline's
            communication on (see ``AnalysisConfig``).
        baseline_interconnect: The interconnect the baseline was measured on.
        cache_dir: Directory of the Parquet ingest cache.
        labels: Labels for the inputs, in order; inputs past the end of
            the list keep their default label.
        streaming: Execution-graph streaming mode.
        graph_engine: Out-of-core / streaming DAG engine.
        quantile_sketch: Streaming quantile sketch algorithm.
        memory_budget_bytes: RAM budget in bytes (0 = auto).
        collective_sync: Cross-rank collective sync (``none`` or ``barrier``).
        goal_semantics: GOAL interpretation (``cta_legacy`` or ``strict``).
        window_size: Frontier / windowed cache size in nodes.
        fail_on_cycle: Raise ``RuntimeError`` when a DAG cycle is detected.
        extract_critical_path_ops: Include exact critical-path op chain table.
        delay_attribution: Enable Scalasca backward wait-state attribution.
        scratch_dir: Optional directory for out-of-core scratch files.
        parquet_bypass_bytes: Raw size threshold for Parquet bypass in auto.

    Returns:
        The report, in ``"pair"`` or ``"multi"`` mode.

    Raises:
        FileNotFoundError: A trace path does not exist.
        ValueError: An input holds no supported trace, the inputs share no
            trace format, a requested aspect does not apply, or a gate
            metric is not reported and ``allow_missing_gates`` is False.
    """
    sources = [baseline, candidate, *others]
    groups = [
        discovery.resolve_trial_group(
            source, label=(labels[i] if labels and i < len(labels) else None)
        )
        for i, source in enumerate(sources)
    ]
    config = _config(
        aspects=aspects,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gates=gates,
        allow_missing_gates=allow_missing_gates,
        collective_algorithm=collective_algorithm,
        alpha=alpha,
        scope_depth=scope_depth,
        ranks=ranks,
        devices=devices,
        predict=predict,
        interconnect=interconnect,
        baseline_interconnect=baseline_interconnect,
        cache_dir=cache_dir,
        streaming=streaming,
        graph_engine=graph_engine,
        quantile_sketch=quantile_sketch,
        memory_budget_bytes=memory_budget_bytes,
        collective_sync=collective_sync,
        goal_semantics=goal_semantics,
        window_size=window_size,
        fail_on_cycle=fail_on_cycle,
        extract_critical_path_ops=extract_critical_path_ops,
        delay_attribution=delay_attribution,
        scratch_dir=scratch_dir,
        parquet_bypass_bytes=parquet_bypass_bytes,
    )
    return engine.run_comparison(_with_distinct_labels(groups), config)


def _describe_table(table_path: pathlib.Path) -> dict[str, Any]:
    """Returns the row count and column names of a Parquet table."""
    row_count = pl.scan_parquet(table_path).select(pl.len()).collect().item()
    return {
        "rows": row_count,
        "columns": list(pl.read_parquet_schema(table_path).keys()),
    }


def inspect_trace(
    source: str | pathlib.Path | models.RunBundle,
    *,
    cache_dir: str | pathlib.Path = ".cta_cache",
) -> dict[str, Any]:
    """Ingests a trace and describes the tables it produced.

    Args:
        source: A trace file, directory, glob or run bundle.
        cache_dir: Directory of the Parquet ingest cache.

    Returns:
        ``{"label", "metadata", "formats"}``, where ``formats`` maps each
        trace format to its ``world_size``, source ``files`` and
        ``tables`` (table name -> ``{"rows", "columns"}``).

    Raises:
        FileNotFoundError: The trace path does not exist.
        ValueError: The source holds no supported trace.
    """
    run_bundle = (
        source
        if isinstance(source, models.RunBundle)
        else discovery.resolve_bundle(source)
    )
    store = cache.CacheStore(pathlib.Path(cache_dir))
    info_by_format: dict[str, dict[str, Any]] = {}
    for trace_format, artifact in run_bundle.artifacts.items():
        table_paths = store.get_or_ingest(
            artifact, plugins.ADAPTERS[trace_format]
        )
        info_by_format[trace_format] = {
            "world_size": artifact.world_size,
            "files": [str(path) for path in artifact.paths],
            "tables": {
                table_name: _describe_table(table_path)
                for table_name, table_path in table_paths.items()
            },
        }
    return {
        "label": run_bundle.label,
        "metadata": run_bundle.metadata,
        "formats": info_by_format,
    }
