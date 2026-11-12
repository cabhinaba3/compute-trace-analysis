from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

from cta.core.bundle import resolve_bundle
from cta.core.models import CompareConfig, TrialGroup


def load_manifest(path: str | Path) -> tuple[tuple[TrialGroup, ...], CompareConfig]:
    manifest_path = Path(path).resolve()
    base_dir = manifest_path.parent
    raw_text = manifest_path.read_text()
    data: dict[str, Any] = (
        tomllib.loads(raw_text)
        if manifest_path.suffix.lower() == ".toml"
        else yaml.safe_load(raw_text)
    )
    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(f"Manifest {manifest_path} must define a top-level 'groups' list")

    groups: list[TrialGroup] = []
    for idx, entry in enumerate(data["groups"]):
        gid = str(entry.get("id", f"group_{idx}"))
        label = str(entry.get("label", gid))
        tags = dict(entry.get("tags", {}))
        trial_specs = entry.get("trials") or ([entry["path"]] if "path" in entry else [])
        if not trial_specs:
            raise ValueError(f"Manifest group '{label}' must specify 'path' or 'trials'")
        bundles = tuple(
            resolve_bundle(
                (base_dir / str(spec)) if not Path(str(spec)).is_absolute() else Path(str(spec)),
                label=f"{label}#{t_idx}" if len(trial_specs) > 1 else label,
                metadata=tags,
            )
            for t_idx, spec in enumerate(trial_specs)
        )
        groups.append(TrialGroup(group_id=gid, label=label, trials=bundles, tags=tags))

    cfg_raw = data.get("config", {})
    aspects = tuple(cfg_raw["aspects"]) if "aspects" in cfg_raw else None
    gates = {str(k): float(v) for k, v in cfg_raw.get("gates", {}).items()}
    cache_dir = Path(cfg_raw.get("cache_dir", ".cta_cache"))
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir

    cfg = CompareConfig(
        aspects=aspects,
        warmup_steps=int(cfg_raw.get("warmup_steps", 0)),
        max_steps=int(cfg_raw["max_steps"]) if cfg_raw.get("max_steps") is not None else None,
        gates=gates,
        allow_missing_gates=bool(cfg_raw.get("allow_missing_gates", False)),
        collective_algo=str(cfg_raw.get("collective_algo", "ring")),
        alpha=float(cfg_raw.get("alpha", 0.05)),
        plots=[dict(p) for p in data.get("plots", [])],
        ranks=tuple(int(r) for r in cfg_raw["ranks"]) if cfg_raw.get("ranks") is not None else None,
        devices=tuple(int(d) for d in cfg_raw["devices"]) if cfg_raw.get("devices") is not None else None,
        scope_depth=int(cfg_raw["scope_depth"]) if cfg_raw.get("scope_depth") is not None else None,
        cache_dir=cache_dir,
    )
    return tuple(groups), cfg
