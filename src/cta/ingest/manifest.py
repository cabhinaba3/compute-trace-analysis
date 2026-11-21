"""Loads a YAML or TOML experiment manifest into trial groups and config."""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

import yaml

from cta import models
from cta.ingest import discovery


def _optional_int(section: dict[str, Any], key: str) -> int | None:
    """Returns ``section[key]`` as an int, or None if absent or null."""
    return int(section[key]) if section.get(key) is not None else None


def _optional_int_tuple(
    section: dict[str, Any], key: str
) -> tuple[int, ...] | None:
    """Returns ``section[key]`` as a tuple of ints, or None if absent/null."""
    if section.get(key) is None:
        return None
    return tuple(int(value) for value in section[key])


def _flag(section: dict[str, Any], key: str) -> bool:
    """Reads a yes/no key of a manifest section (YAML or TOML booleans only)."""
    value = section.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(
            f"Manifest config '{key}' must be true or false, got {value!r}"
        )
    return value


def _streaming_options(
    section: dict[str, Any], base_dir: pathlib.Path
) -> dict[str, Any]:
    """Reads the execution-graph streaming keys of a manifest's ``config``.

    Keys that are absent keep the ``AnalysisConfig`` default; the same names
    as the ``cta compare`` flags can still override them.

    Args:
        section: The manifest's ``config`` mapping.
        base_dir: Directory a relative ``scratch_dir`` is taken from.

    Returns:
        ``AnalysisConfig`` keyword arguments for the keys that are present.

    Raises:
        ValueError: If a key has a value of the wrong type.
    """
    options: dict[str, Any] = {}
    for key in (
        "streaming",
        "graph_engine",
        "quantile_sketch",
        "collective_sync",
        "goal_semantics",
    ):
        if key in section:
            options[key] = str(section[key]).strip().lower()
    for key in ("memory_budget_bytes", "window_size", "parquet_bypass_bytes"):
        value = _optional_int(section, key)
        if value is not None:
            options[key] = value
    for key in (
        "fail_on_cycle",
        "extract_critical_path_ops",
        "delay_attribution",
    ):
        if key in section:
            options[key] = _flag(section, key)
    if section.get("scratch_dir"):
        options["scratch_dir"] = base_dir / str(section["scratch_dir"])
    return options


def _trial_path(base_dir: pathlib.Path, trial_spec: Any) -> pathlib.Path:
    """Returns a trial path, resolving relative ones against ``base_dir``."""
    return base_dir / str(trial_spec)


def load_manifest(
    path: str | pathlib.Path,
) -> tuple[tuple[models.TrialGroup, ...], models.AnalysisConfig]:
    """Loads a manifest that lists the groups to compare and the options.

    The file is TOML when its suffix is ``.toml``, else YAML. Each entry of
    ``groups`` names its traces with ``path`` (one trial) or ``trials``
    (several); relative paths and ``config.cache_dir`` are resolved
    against the manifest's directory. With several trials, trial ``i`` is
    labelled ``<label>#<i>``.

    Args:
        path: The manifest file.

    Returns:
        The trial groups, in manifest order, and the compare config.

    Raises:
        ValueError: The manifest has no top-level ``groups`` list, or a
            group has neither ``path`` nor ``trials``.
    """
    manifest_path = pathlib.Path(path).resolve()
    base_dir = manifest_path.parent
    raw_text = manifest_path.read_text(encoding="utf-8")
    manifest: dict[str, Any] = (
        tomllib.loads(raw_text)
        if manifest_path.suffix.lower() == ".toml"
        else yaml.safe_load(raw_text)
    )
    if not isinstance(manifest, dict) or "groups" not in manifest:
        raise ValueError(
            f"Manifest {manifest_path} must define a top-level 'groups' list"
        )

    groups: list[models.TrialGroup] = []
    for group_idx, entry in enumerate(manifest["groups"]):
        group_id = str(entry.get("id", f"group_{group_idx}"))
        label = str(entry.get("label", group_id))
        tags = dict(entry.get("tags", {}))
        trial_specs = entry.get("trials") or (
            [entry["path"]] if "path" in entry else []
        )
        if not trial_specs:
            raise ValueError(
                f"Manifest group '{label}' must specify 'path' or 'trials'"
            )
        has_many_trials = len(trial_specs) > 1
        bundles = tuple(
            discovery.resolve_bundle(
                _trial_path(base_dir, trial_spec),
                label=f"{label}#{trial_idx}" if has_many_trials else label,
                metadata=tags,
            )
            for trial_idx, trial_spec in enumerate(trial_specs)
        )
        groups.append(
            models.TrialGroup(
                group_id=group_id, label=label, trials=bundles, tags=tags
            )
        )

    config_section = manifest.get("config", {})
    aspects = (
        tuple(config_section["aspects"])
        if "aspects" in config_section
        else None
    )
    gates = {
        str(metric): float(threshold)
        for metric, threshold in config_section.get("gates", {}).items()
    }
    cache_dir = base_dir / config_section.get("cache_dir", ".cta_cache")

    config = models.AnalysisConfig(
        aspects=aspects,
        warmup_steps=int(config_section.get("warmup_steps", 0)),
        max_steps=_optional_int(config_section, "max_steps"),
        gates=gates,
        allow_missing_gates=bool(
            config_section.get("allow_missing_gates", False)
        ),
        collective_algorithm=str(
            config_section.get("collective_algorithm", "ring")
        ),
        alpha=float(config_section.get("alpha", 0.05)),
        plots=[dict(plot_spec) for plot_spec in manifest.get("plots", [])],
        ranks=_optional_int_tuple(config_section, "ranks"),
        devices=_optional_int_tuple(config_section, "devices"),
        scope_depth=_optional_int(config_section, "scope_depth"),
        predict=(
            str(config_section["predict"])
            if config_section.get("predict")
            else None
        ),
        interconnect=(
            str(config_section["interconnect"])
            if config_section.get("interconnect")
            else None
        ),
        baseline_interconnect=(
            str(config_section["baseline_interconnect"])
            if config_section.get("baseline_interconnect")
            else None
        ),
        cache_dir=cache_dir,
        **_streaming_options(config_section, base_dir),
    )
    return tuple(groups), config
