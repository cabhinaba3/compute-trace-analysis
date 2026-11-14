"""Resolves trace paths into run bundles and trial groups.

Format detection asks every registered adapter in turn, so the plugins are
loaded first.
"""

from __future__ import annotations

from collections.abc import Iterable
import glob
import json
import pathlib
import re
from typing import Any

import yaml

from cta.core import models
from cta.core import registry

_RANK_RE = re.compile(r"(?:rank[_.-]?|\.r?)(\d+)(?:\.[^.]+)+$", re.IGNORECASE)
_BUNDLE_METADATA_NAMES = ("bundle.yaml", "bundle.yml", "bundle.json")


def detect_format(path: pathlib.Path) -> str | None:
    """Returns the trace format of a file, as recognised by the adapters.

    Args:
        path: The file to check.

    Returns:
        The name of the first registered adapter that accepts the file, or
        None for unknown formats, hidden files and paths that are not files.
    """
    registry.load_plugins()
    if not path.is_file() or path.name.startswith("."):
        return None
    for trace_format, adapter in registry.ADAPTERS.items():
        if adapter.detect(path):
            return trace_format
    return None


def infer_rank(path: pathlib.Path, fallback: int) -> int:
    """Returns the rank encoded in a per-rank trace file name.

    Recognises names such as ``..._rank3.sqlite`` or ``trace.r3.json``.

    Args:
        path: The trace file.
        fallback: Rank to return when the name encodes none.

    Returns:
        The encoded rank, else ``fallback``.
    """
    match = _RANK_RE.search(path.name)
    return int(match.group(1)) if match else fallback


def _infer_world_size(paths: list[pathlib.Path]) -> int:
    """Returns the larger of the file count and the distinct encoded ranks."""
    if len(paths) <= 1:
        return 1
    ranks = set()
    for path in paths:
        match = _RANK_RE.search(path.name)
        if match:
            ranks.add(int(match.group(1)))
    return max(len(ranks), len(paths))


def resolve_bundle(
    source: str | pathlib.Path | Iterable[str | pathlib.Path],
    *,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> models.RunBundle:
    """Builds a run bundle from a trace file, directory, glob or file list.

    - A file becomes a one-artifact bundle with world size 1.
    - A directory is searched recursively. A ``bundle.yaml``,
      ``bundle.yml`` or ``bundle.json`` in it is merged into the metadata
      (``metadata`` wins), and a ``.nsys-rep`` file is skipped when the
      ``.sqlite`` export with the same stem is present.
    - A string or path with glob characters (``*?[]``) that does not exist
      as a path is expanded recursively.
    - Any other iterable is taken as a list of trace files.

    In every case but a single file, the files are grouped into one artifact
    per detected format and files of unknown format are skipped.

    Args:
        source: A trace file, a directory, a glob pattern or trace files.
        label: Label of the bundle. Defaults to the file stem, the
            directory's ``label`` metadata or its name, the glob's parent
            directory name, or the stem of the first file in the list.
        metadata: Metadata to attach to the bundle.

    Returns:
        The run bundle.

    Raises:
        FileNotFoundError: The path does not exist or the glob matches no
            file.
        ValueError: The file's format is not supported, no file of a
            directory, glob or list is recognised, or the list is empty.
    """
    registry.load_plugins()
    bundle_metadata: dict[str, Any] = dict(metadata or {})

    if isinstance(source, (str, pathlib.Path)):
        source_text = str(source)
        source_path = pathlib.Path(source_text)
        is_pattern = any(char in source_text for char in "*?[]")
        if is_pattern and not source_path.exists():
            return _bundle_from_glob(source_text, label, bundle_metadata)
        if not source_path.exists():
            raise FileNotFoundError(f"Trace path does not exist: {source_path}")
        if source_path.is_file():
            return _bundle_from_single_file(source_path, label, bundle_metadata)
        return _bundle_from_directory(source_path, label, bundle_metadata)

    files = sorted(pathlib.Path(item).resolve() for item in source)
    if not files:
        raise ValueError("Empty trace file sequence provided")
    run_label = label or files[0].stem
    return _bundle_from_files(
        files, run_id=run_label, label=run_label, metadata=bundle_metadata
    )


def _bundle_from_glob(
    pattern: str, label: str | None, metadata: dict[str, Any]
) -> models.RunBundle:
    """Builds a bundle from the files a recursive glob pattern matches."""
    matched_paths = sorted(
        pathlib.Path(name)
        for name in glob.glob(pattern, recursive=True)
        if pathlib.Path(name).is_file()
    )
    if not matched_paths:
        raise FileNotFoundError(f"No trace files matched pattern: {pattern}")
    run_label = label or pathlib.Path(pattern).parent.name or "glob"
    return _bundle_from_files(
        matched_paths, run_id=run_label, label=run_label, metadata=metadata
    )


def _bundle_from_single_file(
    file_path: pathlib.Path, label: str | None, metadata: dict[str, Any]
) -> models.RunBundle:
    """Builds a one-artifact bundle from one trace file."""
    trace_format = detect_format(file_path)
    if trace_format is None:
        raise ValueError(f"Unsupported trace file format: {file_path}")
    # The `.nsys-rep` / exported `.sqlite` twin de-duplication only applies
    # in directory mode: an explicitly named file is always read.
    run_label = label or file_path.stem
    artifact = models.TraceArtifact(
        format=trace_format, paths=(file_path.resolve(),), world_size=1
    )
    return models.RunBundle(
        run_id=run_label,
        label=run_label,
        artifacts={trace_format: artifact},
        metadata=metadata,
    )


def _bundle_from_directory(
    bundle_dir: pathlib.Path, label: str | None, metadata: dict[str, Any]
) -> models.RunBundle:
    """Builds a bundle from every trace file below a directory."""
    for metadata_name in _BUNDLE_METADATA_NAMES:
        metadata_path = bundle_dir / metadata_name
        if metadata_path.exists():
            metadata_text = metadata_path.read_text()
            loaded = (
                yaml.safe_load(metadata_text)
                if metadata_path.suffix != ".json"
                else json.loads(metadata_text)
            )
            if isinstance(loaded, dict):
                metadata = {**loaded, **metadata}
    files = sorted(
        file_path
        for file_path in bundle_dir.rglob("*")
        if file_path.is_file() and not file_path.name.startswith(".")
    )
    # Prefer .sqlite over .nsys-rep when both exist for the same stem.
    sqlite_stems = {
        file_path.with_suffix("")
        for file_path in files
        if file_path.suffix == ".sqlite"
    }
    files = [
        file_path
        for file_path in files
        if not (
            file_path.suffix == ".nsys-rep"
            and file_path.with_suffix("") in sqlite_stems
        )
    ]
    run_label = label or str(metadata.get("label", bundle_dir.name))
    return _bundle_from_files(
        files, run_id=bundle_dir.name, label=run_label, metadata=metadata
    )


def _bundle_from_files(
    files: list[pathlib.Path],
    *,
    run_id: str,
    label: str,
    metadata: dict[str, Any],
) -> models.RunBundle:
    """Groups trace files by detected format into one bundle.

    Bundle metadata files and files of unknown format are skipped.

    Raises:
        ValueError: None of the files is a recognised trace file.
    """
    paths_by_format: dict[str, list[pathlib.Path]] = {}
    for file_path in files:
        if file_path.name in _BUNDLE_METADATA_NAMES:
            continue
        trace_format = detect_format(file_path)
        if trace_format is not None:
            paths_by_format.setdefault(trace_format, []).append(
                file_path.resolve()
            )
    if not paths_by_format:
        raise ValueError(
            f"No recognized trace files found for bundle '{label}'"
        )
    artifacts = {
        trace_format: models.TraceArtifact(
            format=trace_format,
            paths=tuple(paths),
            world_size=_infer_world_size(paths),
        )
        for trace_format, paths in paths_by_format.items()
    }
    return models.RunBundle(
        run_id=run_id, label=label, artifacts=artifacts, metadata=metadata
    )


def resolve_trial_group(
    source: str
    | pathlib.Path
    | models.RunBundle
    | models.TrialGroup
    | Iterable[str | pathlib.Path | models.RunBundle],
    *,
    label: str | None = None,
    tags: dict[str, Any] | None = None,
) -> models.TrialGroup:
    """Builds a trial group from a group, a bundle, a path or several trials.

    Args:
        source: A trial group (returned unchanged), a run bundle (one
            trial), a path or glob for ``resolve_bundle`` (one trial), or an
            iterable of paths and bundles (one trial each; paths are
            resolved with ``resolve_bundle``).
        label: Group label. Defaults to the label of the (first) bundle.
        tags: Group tags. Defaults to the metadata of the (first) bundle. A
            path ``source`` also receives them as bundle metadata.

    Returns:
        The trial group. Its id is the bundle's run id for a single bundle
        or path, and the group label for an iterable.

    Raises:
        FileNotFoundError: A path does not exist or a glob matches nothing.
        ValueError: The iterable is empty, or a path holds no supported
            trace file.
    """
    if isinstance(source, models.TrialGroup):
        return source
    if isinstance(source, models.RunBundle):
        group_label = label or source.label
        return models.TrialGroup(
            group_id=source.run_id,
            label=group_label,
            trials=(source,),
            tags=dict(tags or source.metadata),
        )
    if isinstance(source, (str, pathlib.Path)):
        run_bundle = resolve_bundle(source, label=label, metadata=tags)
        return models.TrialGroup(
            group_id=run_bundle.run_id,
            label=run_bundle.label,
            trials=(run_bundle,),
            tags=dict(tags or run_bundle.metadata),
        )
    bundles = tuple(
        item if isinstance(item, models.RunBundle) else resolve_bundle(item)
        for item in source
    )
    if not bundles:
        raise ValueError("TrialGroup requires at least one RunBundle")
    group_label = label or bundles[0].label
    return models.TrialGroup(
        group_id=group_label,
        label=group_label,
        trials=bundles,
        tags=dict(tags or bundles[0].metadata),
    )
