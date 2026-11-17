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
    cache_dir: str | pathlib.Path,
    gates: dict[str, float] | None = None,
    allow_missing_gates: bool = False,
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
        cache_dir=pathlib.Path(cache_dir),
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
    cache_dir: str | pathlib.Path = ".cta_cache",
    label: str | None = None,
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
        cache_dir: Directory of the Parquet ingest cache.
        label: Label of the group; defaults to the bundle's label.

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
        cache_dir=cache_dir,
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
    cache_dir: str | pathlib.Path = ".cta_cache",
    labels: Sequence[str] | None = None,
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
        cache_dir: Directory of the Parquet ingest cache.
        labels: Labels for the inputs, in order; inputs past the end of
            the list keep their default label.

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
        cache_dir=cache_dir,
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
