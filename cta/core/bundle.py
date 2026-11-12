from __future__ import annotations

import glob
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from cta.core.models import RunBundle, TraceArtifact, TrialGroup
from cta.core.registry import ADAPTERS, load_plugins

_RANK_RE = re.compile(r"(?:rank[_.-]?|\.r?)(\d+)(?:\.[^.]+)+$", re.IGNORECASE)


def detect_format(path: Path) -> str | None:
    load_plugins()
    if not path.is_file() or path.name.startswith("."):
        return None
    for fmt, adapter in ADAPTERS.items():
        if adapter.detect(path):
            return fmt
    return None


def _infer_world_size(paths: list[Path]) -> int:
    if len(paths) <= 1:
        return 1
    ranks = set()
    for p in paths:
        m = _RANK_RE.search(p.name)
        if m:
            ranks.add(int(m.group(1)))
    return max(len(ranks), len(paths))


def resolve_bundle(
    source: str | Path | Iterable[str | Path],
    *,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunBundle:
    load_plugins()
    meta: dict[str, Any] = dict(metadata or {})

    if isinstance(source, (str, Path)):
        src_str = str(source)
        p = Path(src_str)
        if any(ch in src_str for ch in "*?[]") and not p.exists():
            matched = sorted(Path(x) for x in glob.glob(src_str, recursive=True) if Path(x).is_file())
            if not matched:
                raise FileNotFoundError(f"No trace files matched pattern: {src_str}")
            run_label = label or Path(src_str).parent.name or "glob"
            return _bundle_from_files(matched, run_id=run_label, label=run_label, meta=meta)
        if not p.exists():
            raise FileNotFoundError(f"Trace path does not exist: {p}")
        if p.is_file():
            fmt = detect_format(p)
            if fmt is None:
                raise ValueError(f"Unsupported trace file format: {p}")
            # Avoid reading `.nsys-rep` AND its already-exported `.sqlite` twin twice only in dir mode
            run_label = label or p.stem
            return RunBundle(
                run_id=run_label,
                label=run_label,
                artifacts={fmt: TraceArtifact(format=fmt, paths=(p.resolve(),), world_size=1)},
                metadata=meta,
            )
        # Directory RunBundle discovery
        for meta_name in ("bundle.yaml", "bundle.yml", "bundle.json"):
            mp = p / meta_name
            if mp.exists():
                loaded = yaml.safe_load(mp.read_text()) if mp.suffix != ".json" else json.loads(mp.read_text())
                if isinstance(loaded, dict):
                    meta = {**loaded, **meta}
        files = sorted(f for f in p.rglob("*") if f.is_file() and not f.name.startswith("."))
        # Prefer .sqlite over .nsys-rep when both exist for the same stem in a directory
        sqlite_stems = {f.with_suffix("").name for f in files if f.suffix == ".sqlite"}
        files = [f for f in files if not (f.suffix == ".nsys-rep" and f.with_suffix("").name in sqlite_stems)]
        run_label = label or str(meta.get("label", p.name))
        return _bundle_from_files(files, run_id=p.name, label=run_label, meta=meta)

    files = sorted(Path(x).resolve() for x in source)
    if not files:
        raise ValueError("Empty trace file sequence provided")
    run_label = label or files[0].stem
    return _bundle_from_files(files, run_id=run_label, label=run_label, meta=meta)


def _bundle_from_files(
    files: list[Path],
    *,
    run_id: str,
    label: str,
    meta: dict[str, Any],
) -> RunBundle:
    by_fmt: dict[str, list[Path]] = {}
    for f in files:
        if f.name in ("bundle.yaml", "bundle.yml", "bundle.json"):
            continue
        fmt = detect_format(f)
        if fmt is not None:
            by_fmt.setdefault(fmt, []).append(f.resolve())
    if not by_fmt:
        raise ValueError(f"No recognized trace files found for bundle '{label}'")
    artifacts = {
        fmt: TraceArtifact(format=fmt, paths=tuple(paths), world_size=_infer_world_size(paths))
        for fmt, paths in by_fmt.items()
    }
    return RunBundle(run_id=run_id, label=label, artifacts=artifacts, metadata=meta)


def resolve_trial_group(
    source: str | Path | RunBundle | TrialGroup | Iterable[str | Path | RunBundle],
    *,
    label: str | None = None,
    tags: dict[str, Any] | None = None,
) -> TrialGroup:
    if isinstance(source, TrialGroup):
        return source
    if isinstance(source, RunBundle):
        lbl = label or source.label
        return TrialGroup(group_id=source.run_id, label=lbl, trials=(source,), tags=dict(tags or source.metadata))
    if isinstance(source, (str, Path)):
        b = resolve_bundle(source, label=label, metadata=tags)
        return TrialGroup(group_id=b.run_id, label=b.label, trials=(b,), tags=dict(tags or b.metadata))
    bundles = tuple(
        x if isinstance(x, RunBundle) else resolve_bundle(x)
        for x in source
    )
    if not bundles:
        raise ValueError("TrialGroup requires at least one RunBundle")
    lbl = label or bundles[0].label
    return TrialGroup(group_id=lbl, label=lbl, trials=bundles, tags=dict(tags or bundles[0].metadata))
