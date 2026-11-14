"""Comparison engine: ingests trial groups, runs aspects, gates and diagnoses.

``analyze``, ``compare`` and ``inspect_trace`` are the SDK entry points;
``run_comparison`` does the work for pre-built groups and config.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import pathlib
from typing import Any, Literal

import polars as pl

from cta.core import bundle as core_bundle
from cta.core import cache
from cta.core import diagnosis
from cta.core import models
from cta.core import registry


def ingest_group_tables(
    groups: Sequence[models.TrialGroup],
    store: cache.CacheStore,
) -> dict[str, list[dict[str, pathlib.Path]]]:
    """Ingests (or fetches from the cache) every artifact of every group.

    Each trial's tables from all of its formats are merged into one map.
    Every table appears under ``<format>.<table>`` and under its bare name;
    when several formats have a table with the same bare name, the first
    format's table keeps it.

    Args:
        groups: The trial groups to ingest.
        store: The cache that holds the ingested Parquet tables.

    Returns:
        Group id -> one map of table name -> Parquet path per trial.
    """
    registry.load_plugins()
    tables_by_group: dict[str, list[dict[str, pathlib.Path]]] = {}
    for group in groups:
        trial_tables: list[dict[str, pathlib.Path]] = []
        for trial in group.trials:
            merged: dict[str, pathlib.Path] = {}
            for trace_format, artifact in trial.artifacts.items():
                adapter = registry.ADAPTERS[trace_format]
                format_tables = store.get_or_ingest(artifact, adapter)
                for table_name, table_path in format_tables.items():
                    merged[f"{trace_format}.{table_name}"] = table_path
                    merged.setdefault(table_name, table_path)
            trial_tables.append(merged)
        tables_by_group[group.group_id] = trial_tables
    return tables_by_group


def _select_aspects(
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
    for name, aspect in registry.ASPECTS.items():
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
    config: models.CompareConfig | None = None,
) -> models.ComparisonReport:
    """Analyses one group or compares several groups.

    Ingests every artifact, runs each applicable aspect and keeps the
    results that have a non-empty summary or any details table. Then it
    diagnoses bottleneck regimes and, with two or more groups, evaluates
    the regression gates.

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
    registry.load_plugins()
    if not groups:
        raise ValueError("Analysis requires at least 1 trace group")
    config = config or models.CompareConfig()
    group_tuple = tuple(groups)
    common_formats = set(group_tuple[0].formats)
    for group in group_tuple[1:]:
        common_formats &= group.formats
    if not common_formats:
        formats_by_group = {
            group.label: sorted(group.formats) for group in group_tuple
        }
        raise ValueError(
            "Traces share no supported/common trace format. "
            f"Found: {formats_by_group}"
        )

    store = cache.CacheStore(config.cache_dir)
    tables = ingest_group_tables(group_tuple, store)
    aspect_names = _select_aspects(common_formats, config.aspects)

    results: dict[str, models.AspectResult] = {}
    for name in aspect_names:
        result = registry.ASPECTS[name].run(group_tuple, tables, config)
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
    violations: list[models.RegressionViolation] = []
    if len(group_tuple) >= 2:
        missing = diagnosis.missing_gate_metrics(results, config.gates)
        if missing and not config.allow_missing_gates:
            raise ValueError(
                f"Gate metrics not found in report: {missing} "
                "(typo, or the aspect that reports them did not run). "
                "Use --allow-missing-gates / allow_missing_gates=True "
                "to ignore."
            )
        violations = diagnosis.evaluate_gates(results, config.gates)
    return models.ComparisonReport(
        mode=mode,
        groups=group_tuple,
        aspects=results,
        diagnoses=diagnoses,
        violations=violations,
        sources=tables,
        config=config,
    )


def analyze(
    source: str
    | pathlib.Path
    | models.RunBundle
    | models.TrialGroup
    | Iterable[str | pathlib.Path | models.RunBundle],
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
) -> models.ComparisonReport:
    """Analyses a single trace, run bundle directory or multi-trial group.

    Args:
        source: A trace file, directory, glob, run bundle, trial group or
            several trials (see ``cta.core.bundle.resolve_trial_group``).
        aspects: Names of the aspects to run; None or empty runs every
            default-enabled aspect that applies.
        warmup_steps: Number of leading steps to drop as warm-up.
        max_steps: Maximum number of steps to keep after warm-up.
        collective_algorithm: Collective algorithm assumed for collectives.
        alpha: Significance level of the hypothesis tests.
        scope_depth: Scope level kernels are attributed to (see
            ``CompareConfig``).
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
    group = core_bundle.resolve_trial_group(source, label=label)
    config = models.CompareConfig(
        aspects=tuple(aspects) if aspects else None,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        collective_algorithm=collective_algorithm,
        alpha=alpha,
        scope_depth=scope_depth,
        ranks=tuple(ranks) if ranks else None,
        devices=tuple(devices) if devices else None,
        cache_dir=pathlib.Path(cache_dir),
    )
    return run_comparison((group,), config)


def compare(
    baseline: str
    | pathlib.Path
    | models.RunBundle
    | models.TrialGroup
    | Iterable[str | pathlib.Path | models.RunBundle],
    candidate: str
    | pathlib.Path
    | models.RunBundle
    | models.TrialGroup
    | Iterable[str | pathlib.Path | models.RunBundle],
    *others: str
    | pathlib.Path
    | models.RunBundle
    | models.TrialGroup
    | Iterable[str | pathlib.Path | models.RunBundle],
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
) -> models.ComparisonReport:
    """Compares a baseline with one or more candidates.

    Each input is resolved into a trial group (see
    ``cta.core.bundle.resolve_trial_group``). If any two groups end up with
    the same label, every group's label and id get its position appended
    (``<label>_<i>``); the caller's groups are copied, not modified.

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
            ``CompareConfig``).
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
        core_bundle.resolve_trial_group(
            source, label=(labels[i] if labels and i < len(labels) else None)
        )
        for i, source in enumerate(sources)
    ]
    # Ensure distinct labels when comparing identically named files in
    # different directories.
    label_counts: dict[str, int] = {}
    for group in groups:
        label_counts[group.label] = label_counts.get(group.label, 0) + 1
    if any(count > 1 for count in label_counts.values()):
        # Copy instead of mutating: callers may pass in (and reuse) their own
        # TrialGroups.
        groups = [
            dataclasses.replace(
                group,
                label=f"{group.label}_{i}",
                group_id=f"{group.group_id}_{i}",
            )
            for i, group in enumerate(groups)
        ]

    config = models.CompareConfig(
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
    return run_comparison(groups, config)


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
    registry.load_plugins()
    run_bundle = (
        source
        if isinstance(source, models.RunBundle)
        else core_bundle.resolve_bundle(source)
    )
    store = cache.CacheStore(pathlib.Path(cache_dir))
    info_by_format: dict[str, dict[str, Any]] = {}
    for trace_format, artifact in run_bundle.artifacts.items():
        table_paths = store.get_or_ingest(
            artifact, registry.ADAPTERS[trace_format]
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
