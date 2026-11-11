from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl


@dataclass(slots=True, frozen=True)
class TraceArtifact:
    format: str
    paths: tuple[Path, ...]
    world_size: int = 1
    digest: str = ""


@dataclass(slots=True)
class RunBundle:
    run_id: str
    label: str
    artifacts: dict[str, TraceArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formats(self) -> set[str]:
        return set(self.artifacts)


@dataclass(slots=True)
class TrialGroup:
    group_id: str
    label: str
    trials: tuple[RunBundle, ...]
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def formats(self) -> set[str]:
        if not self.trials:
            return set()
        common = set(self.trials[0].formats)
        for t in self.trials[1:]:
            common &= t.formats
        return common


@dataclass(slots=True)
class CompareConfig:
    aspects: tuple[str, ...] | None = None
    warmup_steps: int = 0
    max_steps: int | None = None
    gates: dict[str, float] = field(default_factory=dict)
    collective_algo: str = "ring"
    alpha: float = 0.05
    cache_dir: Path = field(default_factory=lambda: Path(".cta_cache"))


@dataclass(slots=True, frozen=True)
class RegressionViolation:
    metric: str
    aspect: str
    observed_pct: float
    threshold_pct: float
    message: str


@dataclass(slots=True, frozen=True)
class BottleneckShift:
    baseline_label: str
    candidate_label: str
    baseline_regime: str
    candidate_regime: str
    shares_baseline: dict[str, float]
    shares_candidate: dict[str, float]
    summary: str


@dataclass(slots=True)
class AspectResult:
    name: str
    title: str
    summary: pl.DataFrame
    details: dict[str, pl.DataFrame] = field(default_factory=dict)
    scalars: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_polars(self, table: str = "summary") -> pl.DataFrame:
        if table == "summary":
            return self.summary
        if table in self.details:
            return self.details[table]
        raise KeyError(f"Unknown table '{table}' in aspect '{self.name}'. Available: {['summary', *self.details]}")

    def to_pandas(self, table: str = "summary") -> Any:
        return self.to_polars(table).to_pandas()


@dataclass(slots=True)
class ComparisonReport:
    mode: Literal["single", "pair", "multi"]
    groups: tuple[TrialGroup, ...]
    aspects: dict[str, AspectResult]
    diagnoses: list[BottleneckShift] = field(default_factory=list)
    violations: list[RegressionViolation] = field(default_factory=list)

    def aspect(self, name: str) -> AspectResult:
        if name not in self.aspects:
            raise KeyError(f"Aspect '{name}' not in report. Available: {list(self.aspects)}")
        return self.aspects[name]

    def to_polars(self, aspect: str, table: str = "summary") -> pl.DataFrame:
        return self.aspect(aspect).to_polars(table)

    def to_pandas(self, aspect: str, table: str = "summary") -> Any:
        return self.aspect(aspect).to_pandas(table)

    def plot(self, kind: str = "breakdown", path: str | Path | None = None, **kwargs: Any) -> Any:
        from cta.viz import render_plot

        return render_plot(self, kind=kind, path=Path(path) if path else None, **kwargs)

    def to_latex(self, path: str | Path | None = None, **kwargs: Any) -> str:
        from cta.exporters.latex import export_latex

        return export_latex(self, path=Path(path) if path else None, **kwargs)

    def to_markdown(self, path: str | Path | None = None) -> str:
        from cta.exporters.markdown import export_markdown

        return export_markdown(self, path=Path(path) if path else None)

    def export(self, out_dir: str | Path, formats: Iterable[str] = ("markdown", "csv", "json")) -> dict[str, Path]:
        from cta.exporters.tabular import export_bundle

        return export_bundle(self, Path(out_dir), formats=formats)
