"""Public data model: trace artifacts, trial groups, config and reports."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import pathlib
from typing import Any, Literal

import polars as pl


@dataclasses.dataclass(slots=True, frozen=True)
class TraceArtifact:
    """The trace files of one format that belong to one run.

    Attributes:
        format: Name of the adapter that reads these files (e.g. ``"nsys"``).
        paths: The trace files, one per rank for multi-rank traces.
        world_size: Number of ranks the files cover.
        digest: Precomputed cache key. When empty, the cache hashes the files.
    """

    format: str
    paths: tuple[pathlib.Path, ...]
    world_size: int = 1
    digest: str = ""


@dataclasses.dataclass(slots=True)
class RunBundle:
    """All trace artifacts recorded for one run, keyed by trace format.

    Attributes:
        run_id: Identifier of the run (usually the file stem or directory).
        label: Name shown for the run in reports.
        artifacts: Trace artifacts keyed by trace format.
        metadata: Free-form run metadata, e.g. read from ``bundle.yaml``.
    """

    run_id: str
    label: str
    artifacts: dict[str, TraceArtifact]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def formats(self) -> set[str]:
        """The trace formats present in this run."""
        return set(self.artifacts)


@dataclasses.dataclass(slots=True)
class TrialGroup:
    """Repeated runs (trials) of one configuration, compared as one unit.

    Attributes:
        group_id: Identifier of the group.
        label: Name shown for the group in reports.
        trials: The runs in the group.
        tags: Free-form tags describing the group.
    """

    group_id: str
    label: str
    trials: tuple[RunBundle, ...]
    tags: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def formats(self) -> set[str]:
        """The trace formats that every trial in the group provides."""
        if not self.trials:
            return set()
        return set.intersection(*(t.formats for t in self.trials))


@dataclasses.dataclass(slots=True)
class AnalysisConfig:
    """Options for one analysis or comparison run.

    Attributes:
        aspects: Names of the aspects to run. None runs every default-enabled
            aspect that applies to the formats all groups share.
        warmup_steps: Number of leading steps to drop as warm-up.
        max_steps: Maximum number of steps to keep after warm-up; None keeps
            all of them.
        gates: Regression gates: metric name -> allowed regression in percent.
        collective_algorithm: Collective algorithm assumed when estimating the
            expected traffic of collectives (``"ring"`` by default).
        alpha: Significance level of the hypothesis tests.
        allow_missing_gates: If True, gate metrics that no aspect reports are
            ignored instead of raising an error.
        plots: Declarative plot specs (the manifest's ``plots:`` list).
        ranks: Analyse only these ranks; None analyses all of them.
        devices: Analyse only kernels on these GPU devices; None analyses all.
        scope_depth: Scope level kernels are attributed to. None uses the
            innermost scope; N >= 1 uses the N-th enclosing level, counted
            from the outermost.
        predict: A what-if scenario (rules joined by ``;``, see
            ``cta.analysis.whatif``) that the ``whatif`` aspect applies to the
            baseline and compares with the measured candidates; None applies
            none.
        interconnect: An interconnect (``cta.analysis.interconnect.parse``,
            such as ``"gpus_per_node=8,inter=5us:50GB/s,algo=ring"``) on
            which the ``whatif`` aspect prices the baseline's communication
            before predicting the candidates; None leaves it as recorded.
        baseline_interconnect: The interconnect the baseline was measured
            on, so recorded communication times are scaled by how much
            ``interconnect`` costs against it; None keeps them as recorded.
        cache_dir: Directory of the Parquet ingest cache.
    """

    aspects: tuple[str, ...] | None = None
    warmup_steps: int = 0
    max_steps: int | None = None
    gates: dict[str, float] = dataclasses.field(default_factory=dict)
    collective_algorithm: str = "ring"
    alpha: float = 0.05
    allow_missing_gates: bool = False
    plots: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ranks: tuple[int, ...] | None = None
    devices: tuple[int, ...] | None = None
    scope_depth: int | None = None
    predict: str | None = None
    interconnect: str | None = None
    baseline_interconnect: str | None = None
    cache_dir: pathlib.Path = dataclasses.field(
        default_factory=lambda: pathlib.Path(".cta_cache")
    )


@dataclasses.dataclass(slots=True, frozen=True)
class GateViolation:
    """A regression gate that a comparison failed.

    Attributes:
        metric: Name of the gated metric.
        aspect: Name of the aspect that reported the metric.
        observed_pct: Observed regression in percent. For higher-is-better
            metrics the sign of the change is flipped, so a positive value
            is always a regression.
        threshold_pct: The gate's threshold in percent.
        message: Human-readable description of the failure.
    """

    metric: str
    aspect: str
    observed_pct: float
    threshold_pct: float
    message: str


@dataclasses.dataclass(slots=True, frozen=True)
class BottleneckShift:
    """The bottleneck regime of a baseline group and of one candidate group.

    In a single-group analysis the candidate fields repeat the baseline.

    Attributes:
        baseline_label: Label of the baseline group.
        candidate_label: Label of the candidate group.
        baseline_regime: Bottleneck regime of the baseline group.
        candidate_regime: Bottleneck regime of the candidate group.
        shares_baseline: Time share per bottleneck category in the baseline.
        shares_candidate: Time share per bottleneck category in the candidate.
        summary: One-line description of the regimes and their top shares.
    """

    baseline_label: str
    candidate_label: str
    baseline_regime: str
    candidate_regime: str
    shares_baseline: dict[str, float]
    shares_candidate: dict[str, float]
    summary: str


@dataclasses.dataclass(slots=True, frozen=True)
class Recommendation:
    """One optimization the evidence in a trace supports.

    The ceiling is the most the change could save, not a prediction: it is
    the time the trace shows being lost, taken out as a whole. How it was
    measured is in ``basis``.

    Attributes:
        group_label: Label of the group the advice is for.
        priority: Position among the group's recommendations, 1 being the
            one with the largest ceiling.
        rule: Which rule found it (``launch_bound``, ``exposed_comm``,
            ``exposed_copies``, or ``whatif:<op family>``).
        title: What is wrong, in one line.
        action: What to try.
        ceiling_ms: The most it could save, in ms per step (``per`` is
            ``"step"``) or for the whole recorded graph (``"run"``).
        ceiling_pct: ``ceiling_ms`` as a percentage of the step, or of the
            graph's critical path.
        per: ``"step"`` or ``"run"``.
        basis: ``"measured-removal"``: the loss was measured on the timeline
            (idle gaps, communication or copies that no compute covers) and
            removed on paper. ``"whatif-removal"``: the op family's time was
            set to zero in the execution graph and the critical path solved
            again (``cta whatif``).
        evidence: The numbers behind it, by name.
    """

    group_label: str
    priority: int
    rule: str
    title: str
    action: str
    ceiling_ms: float
    ceiling_pct: float
    per: str
    basis: str
    evidence: dict[str, float | str]


@dataclasses.dataclass(slots=True)
class AspectResult:
    """The output of one comparison aspect.

    Attributes:
        name: Registered name of the aspect.
        title: Human-readable title of the aspect.
        summary: Summary table (one row per metric or entity).
        details: Additional tables keyed by table name.
        scalars: Scalar values per group label, keyed by name.
    """

    name: str
    title: str
    summary: pl.DataFrame
    details: dict[str, pl.DataFrame] = dataclasses.field(default_factory=dict)
    scalars: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )

    def to_polars(self, table: str = "summary") -> pl.DataFrame:
        """Returns one of the aspect's tables as a polars DataFrame.

        Args:
            table: ``"summary"`` or the name of a details table.

        Returns:
            The requested table.

        Raises:
            KeyError: The aspect has no table named ``table``.
        """
        if table == "summary":
            return self.summary
        if table in self.details:
            return self.details[table]
        raise KeyError(
            f"Unknown table '{table}' in aspect '{self.name}'. "
            f"Available: {['summary', *self.details]}"
        )

    def to_pandas(self, table: str = "summary") -> Any:
        """Returns one of the aspect's tables as a pandas DataFrame.

        Args:
            table: ``"summary"`` or the name of a details table.

        Returns:
            The requested table, converted to pandas.

        Raises:
            KeyError: The aspect has no table named ``table``.
        """
        return self.to_polars(table).to_pandas()


@dataclasses.dataclass(slots=True)
class Report:
    """The result of analysing one group or comparing several groups.

    Attributes:
        mode: ``"single"`` for one group, ``"pair"`` for two, else
            ``"multi"``.
        groups: The analysed groups; the first one is the baseline.
        aspects: Aspect results keyed by aspect name.
        diagnoses: Bottleneck regime of each candidate versus the baseline.
        recommendations: Optimization advice of every group, ranked within
            each group by its ceiling (see ``Recommendation``).
        violations: Regression gates that failed.
        sources: Plot inputs, not part of the exported report. Per group id,
            one map of table name -> cached Parquet path per trial.
        config: The configuration the report was produced with.
    """

    mode: Literal["single", "pair", "multi"]
    groups: tuple[TrialGroup, ...]
    aspects: dict[str, AspectResult]
    diagnoses: list[BottleneckShift] = dataclasses.field(default_factory=list)
    violations: list[GateViolation] = dataclasses.field(default_factory=list)
    sources: dict[str, list[dict[str, pathlib.Path]]] = dataclasses.field(
        default_factory=dict, repr=False
    )
    config: AnalysisConfig | None = dataclasses.field(default=None, repr=False)
    recommendations: list[Recommendation] = dataclasses.field(
        default_factory=list
    )

    def aspect(self, name: str) -> AspectResult:
        """Returns the result of one aspect.

        Args:
            name: Registered name of the aspect.

        Returns:
            The aspect's result.

        Raises:
            KeyError: The report has no result for ``name``.
        """
        if name not in self.aspects:
            raise KeyError(
                f"Aspect '{name}' not in report. "
                f"Available: {list(self.aspects)}"
            )
        return self.aspects[name]

    def to_polars(self, aspect: str, table: str = "summary") -> pl.DataFrame:
        """Returns one table of one aspect as a polars DataFrame.

        Args:
            aspect: Registered name of the aspect.
            table: ``"summary"`` or the name of a details table.

        Returns:
            The requested table.

        Raises:
            KeyError: The aspect or the table does not exist.
        """
        return self.aspect(aspect).to_polars(table)

    def to_pandas(self, aspect: str, table: str = "summary") -> Any:
        """Returns one table of one aspect as a pandas DataFrame.

        Args:
            aspect: Registered name of the aspect.
            table: ``"summary"`` or the name of a details table.

        Returns:
            The requested table, converted to pandas.

        Raises:
            KeyError: The aspect or the table does not exist.
        """
        return self.aspect(aspect).to_pandas(table)

    def plot(
        self,
        kind: str = "breakdown",
        path: str | pathlib.Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """Draws one plot of the report.

        Args:
            kind: Plot kind (see ``cta plot --list-kinds``).
            path: File to save the figure to; a ``.html`` path gives an
                interactive Plotly file. None does not save the figure.
            **kwargs: Options passed to ``cta.viz.render_plot`` (style, dpi
                and the options the plot kind declares).

        Returns:
            The figure returned by ``cta.viz.render_plot``.
        """
        from cta import viz

        return viz.render_plot(
            self, kind=kind, path=pathlib.Path(path) if path else None, **kwargs
        )

    def dashboard(
        self,
        path: str | pathlib.Path,
        *,
        title: str = "CTA report",
        cdn: bool = False,
        kinds: Iterable[str] | None = None,
        **options: Any,
    ) -> pathlib.Path:
        """Writes one interactive HTML dashboard of the report.

        Needs the ``viz`` extra.

        Args:
            path: The HTML file to write.
            title: Title of the dashboard page.
            cdn: If True, the page loads plotly from a CDN instead of
                embedding it.
            kinds: Plot kinds to include. None uses the default plot set
                for the report's data.
            **options: Options passed to every plot kind in ``kinds``.
                Ignored when ``kinds`` is None.

        Returns:
            The path of the written dashboard.
        """
        from cta.viz import context
        from cta.viz import dashboard as viz_dashboard
        from cta.viz import spec

        specs = (
            [spec.PlotSpec(kind, dict(options)) for kind in kinds]
            if kinds
            else None
        )
        return viz_dashboard.build_dashboard(
            context.DataContext(self),
            specs,
            pathlib.Path(path),
            title=title,
            cdn=cdn,
        ).path

    def to_latex(
        self, path: str | pathlib.Path | None = None, **kwargs: Any
    ) -> str:
        """Renders the aspect summaries as LaTeX booktabs tables.

        Args:
            path: File to write the LaTeX to; None only returns it.
            **kwargs: Options passed to ``cta.export.latex.export_latex``.

        Returns:
            The LaTeX source.
        """
        from cta.export import latex

        return latex.export_latex(
            self, path=pathlib.Path(path) if path else None, **kwargs
        )

    def to_markdown(self, path: str | pathlib.Path | None = None) -> str:
        """Renders the report as Markdown.

        Args:
            path: File to write the Markdown to; None only returns it.

        Returns:
            The Markdown text.
        """
        from cta.export import markdown

        return markdown.export_markdown(
            self, path=pathlib.Path(path) if path else None
        )

    def export(
        self,
        out_dir: str | pathlib.Path,
        formats: Iterable[str] = ("markdown", "csv", "json"),
    ) -> dict[str, pathlib.Path]:
        """Writes the report in several formats into one directory.

        Args:
            out_dir: Directory to write the files to; created if missing.
            formats: Output format names (see ``cta.export.writers``).

        Returns:
            The written files, keyed by output (e.g. ``"csv:<aspect>"``).

        Raises:
            ValueError: A format is not registered.
        """
        from cta.export import writers

        return writers.write_report(self, pathlib.Path(out_dir), formats)
