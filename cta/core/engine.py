from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl

from cta.core.bundle import resolve_bundle, resolve_trial_group
from cta.core.cache import CacheStore
from cta.core.diagnosis import diagnose_bottleneck_shifts, evaluate_gates
from cta.core.models import (
    AspectResult,
    CompareConfig,
    ComparisonReport,
    RunBundle,
    TrialGroup,
)
from cta.core.registry import ADAPTERS, ASPECTS, load_plugins


def ingest_group_tables(
    groups: Sequence[TrialGroup],
    store: CacheStore,
) -> dict[str, list[dict[str, Path]]]:
    """Ingest (or fetch cached Parquet tables for) all artifacts across all TrialGroups.

    Returns mapping: group_id -> [merged table_name -> parquet_path per trial].
    """
    load_plugins()
    out: dict[str, list[dict[str, Path]]] = {}
    for grp in groups:
        trial_tables: list[dict[str, Path]] = []
        for bundle in grp.trials:
            merged: dict[str, Path] = {}
            for fmt, artifact in bundle.artifacts.items():
                adapter = ADAPTERS[fmt]
                fmt_tables = store.get_or_ingest(artifact, adapter)
                for tname, tpath in fmt_tables.items():
                    merged[f"{fmt}.{tname}"] = tpath
                    merged.setdefault(tname, tpath)
            trial_tables.append(merged)
        out[grp.group_id] = trial_tables
    return out


def _select_aspects(common_formats: set[str], requested: tuple[str, ...] | None) -> list[str]:
    selected: list[str] = []
    for name, aspect in ASPECTS.items():
        if requested is not None and name not in requested:
            continue
        reqs = set(aspect.required_formats)
        ok = bool(common_formats & reqs) if aspect.any_of_formats else reqs.issubset(common_formats)
        if ok:
            selected.append(name)
    if requested is not None:
        missing = [r for r in requested if r not in selected]
        if missing:
            raise ValueError(
                f"Requested aspects {missing} not applicable to common formats {sorted(common_formats)}"
            )
    return selected


def run_comparison(
    groups: Sequence[TrialGroup],
    config: CompareConfig | None = None,
) -> ComparisonReport:
    load_plugins()
    if len(groups) < 2:
        raise ValueError("Comparison requires at least 2 trace groups (baseline and candidate)")
    cfg = config or CompareConfig()
    grp_tuple = tuple(groups)
    common_formats = set(grp_tuple[0].formats)
    for g in grp_tuple[1:]:
        common_formats &= g.formats
    if not common_formats:
        per_grp = {g.label: sorted(g.formats) for g in grp_tuple}
        raise ValueError(
            f"Compared traces share no common trace format (same-format rule). Found: {per_grp}"
        )

    store = CacheStore(cfg.cache_dir)
    tables = ingest_group_tables(grp_tuple, store)
    aspect_names = _select_aspects(common_formats, cfg.aspects)

    results: dict[str, AspectResult] = {}
    for name in aspect_names:
        res = ASPECTS[name].run(grp_tuple, tables, cfg)
        if not res.summary.is_empty() or res.details:
            results[name] = res

    mode: Literal["pair", "multi"] = "pair" if len(grp_tuple) == 2 else "multi"
    diagnoses = diagnose_bottleneck_shifts(grp_tuple, results)
    violations = evaluate_gates(results, cfg.gates)
    return ComparisonReport(
        mode=mode,
        groups=grp_tuple,
        aspects=results,
        diagnoses=diagnoses,
        violations=violations,
    )


def compare(
    baseline: str | Path | RunBundle | TrialGroup | Iterable[str | Path | RunBundle],
    candidate: str | Path | RunBundle | TrialGroup | Iterable[str | Path | RunBundle],
    *others: str | Path | RunBundle | TrialGroup | Iterable[str | Path | RunBundle],
    aspects: Sequence[str] | None = None,
    warmup_steps: int = 0,
    max_steps: int | None = None,
    gates: dict[str, float] | None = None,
    collective_algo: str = "ring",
    alpha: float = 0.05,
    cache_dir: str | Path = ".cta_cache",
    labels: Sequence[str] | None = None,
) -> ComparisonReport:
    raw_inputs = [baseline, candidate, *others]
    groups = [
        resolve_trial_group(src, label=(labels[i] if labels and i < len(labels) else None))
        for i, src in enumerate(raw_inputs)
    ]
    # Ensure distinct labels when comparing identically named files in different directories
    seen: dict[str, int] = {}
    for g in groups:
        seen[g.label] = seen.get(g.label, 0) + 1
    if any(c > 1 for c in seen.values()):
        for idx, g in enumerate(groups):
            g.label = f"{g.label}_{idx}"
            g.group_id = f"{g.group_id}_{idx}"

    cfg = CompareConfig(
        aspects=tuple(aspects) if aspects else None,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gates=dict(gates or {}),
        collective_algo=collective_algo,
        alpha=alpha,
        cache_dir=Path(cache_dir),
    )
    return run_comparison(groups, cfg)


def inspect_trace(
    source: str | Path | RunBundle,
    *,
    cache_dir: str | Path = ".cta_cache",
) -> dict[str, Any]:
    load_plugins()
    bundle = source if isinstance(source, RunBundle) else resolve_bundle(source)
    store = CacheStore(Path(cache_dir))
    table_info: dict[str, dict[str, Any]] = {}
    for fmt, art in bundle.artifacts.items():
        tpaths = store.get_or_ingest(art, ADAPTERS[fmt])
        table_info[fmt] = {
            "world_size": art.world_size,
            "files": [str(p) for p in art.paths],
            "tables": {
                tname: {
                    "rows": pl.scan_parquet(tp).select(pl.len()).collect().item(),
                    "columns": list(pl.read_parquet_schema(tp).keys()),
                }
                for tname, tp in tpaths.items()
            },
        }
    return {"label": bundle.label, "metadata": bundle.metadata, "formats": table_info}
