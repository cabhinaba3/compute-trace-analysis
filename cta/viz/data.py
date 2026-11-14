"""Data access for plots: named raw tables, report tables and SQL.

Raw tables (kernels, steps, gpu_metrics, ...) come from the cache paths the
report keeps. They are sliced with the same function the aspects use
(``--warmup-steps``/``--max-steps`` per rank) so plots and tables always
agree, resolved through format-qualified keys with explicit precedence, and
carry ``label`` (group label), ``trial``, and times relative to each rank's
first event (ranks have independent clocks, so absolute times are never mixed
across ranks).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import pathlib
import re
from typing import Any, cast

import polars as pl

from cta.core import models
from cta.viz import spec

SOURCES = (
    "kernels",
    "steps",
    "gpu_metrics",
    "runtime",
    "ncu_kernels",
    "ncu_findings",
    "jtop",
    "osrt",
)
# Trace formats with step tables, in precedence order: a trial that has
# several of them is read from the first.
_TRACE_FORMAT_PRECEDENCE = ("nsys", "kineto", "jax", "trtexec_timing")
# Raw tables read as cached, without slicing: (source name, cache table key).
_UNSLICED_TABLES = (
    ("ncu_kernels", "ncu.ncu_kernels"),
    ("ncu_findings", "ncu.ncu_findings"),
    ("jtop", "jtop.samples"),
)


class DataContext:
    """The data a plot draws from: one report's tables, raw tables and SQL.

    Raw tables are loaded on first use and then kept.

    Attributes:
        report: The report being plotted; None for plots that need no report
            (such as ``history``).
        history: Glob of saved ``report.json`` files for the ``history`` plot.
    """

    def __init__(
        self,
        report: models.ComparisonReport | None = None,
        *,
        history: str | None = None,
    ) -> None:
        """Initialises the context.

        Args:
            report: The report to plot, if any.
            history: Glob of saved ``report.json`` files, if any.
        """
        self.report = report
        self.history = history
        self._raw_tables: dict[str, pl.DataFrame] = {}
        self._is_built = False

    # Report side.

    @property
    def labels(self) -> list[str]:
        """The group labels, in group order (empty without a report)."""
        if not self.report:
            return []
        return [group.label for group in self.report.groups]

    @property
    def config(self) -> models.CompareConfig:
        """The report's analysis config, or the default config."""
        report_config = self.report.config if self.report else None
        return report_config or models.CompareConfig()

    def require_report(self) -> models.ComparisonReport:
        """Returns the report.

        Returns:
            The report being plotted.

        Raises:
            spec.NotApplicableError: If the context has no report.
        """
        if self.report is None:
            raise spec.NotApplicableError(
                "this plot needs a trace/report "
                "(give trace paths or --manifest)"
            )
        return self.report

    def report_table(self, aspect: str, table: str = "summary") -> pl.DataFrame:
        """Returns one table of an aspect's results.

        Args:
            aspect: Aspect name.
            table: ``"summary"`` or the name of one of the aspect's detail
                tables.

        Returns:
            The table.

        Raises:
            spec.NotApplicableError: If there is no report, the report lacks the
                aspect, or the aspect has no such table.
        """
        report = self.require_report()
        if aspect not in report.aspects:
            available = ", ".join(report.aspects) or "none"
            raise spec.NotApplicableError(
                f"needs the '{aspect}' aspect (have: {available})"
            )
        try:
            return report.aspects[aspect].to_polars(table)
        except KeyError as exc:
            raise spec.NotApplicableError(str(exc.args[0])) from exc

    def has_aspect(self, aspect: str) -> bool:
        """Returns whether the report has results for an aspect.

        Args:
            aspect: Aspect name.

        Returns:
            True if there is a report and it has the aspect.
        """
        return self.report is not None and aspect in self.report.aspects

    def metric(self, aspect: str, metric: str) -> dict[str, float]:
        """Returns the per-label value of a summary metric.

        With exactly two groups and a summary in baseline/candidate form, the
        baseline and candidate columns map to the first and second label.

        Args:
            aspect: Aspect whose summary holds the metric.
            metric: Metric name.

        Returns:
            Group label -> value; labels without a value are omitted, and an
            unknown metric gives an empty dict.

        Raises:
            spec.NotApplicableError: If there is no report or it lacks the
                aspect.
        """
        summary = self.report_table(aspect)
        matching_rows = summary.filter(pl.col("metric") == metric)
        if matching_rows.is_empty():
            return {}
        metric_row = matching_rows.row(0, named=True)
        labels = self.labels
        if (
            len(labels) == 2
            and "baseline" in metric_row
            and "candidate" in metric_row
        ):
            return {
                labels[0]: float(metric_row["baseline"] or 0.0),
                labels[1]: float(metric_row["candidate"] or 0.0),
            }
        return {
            label: float(metric_row[label])
            for label in labels
            if metric_row.get(label) is not None
        }

    # Raw tables.

    def _trials(self) -> list[tuple[str, int, dict[str, pathlib.Path]]]:
        """Lists (group label, trial index, cache table paths) of every trial.

        Raises:
            spec.NotApplicableError: If there is no report or it has no cached
                trace tables.
        """
        report = self.require_report()
        trials = []
        for group in report.groups:
            group_sources = report.sources.get(group.group_id, [])
            for trial, table_paths in enumerate(group_sources):
                trials.append((group.label, trial, table_paths))
        if not trials:
            raise spec.NotApplicableError(
                "the report has no cached trace tables (re-run the analysis)"
            )
        return trials

    def _build_raw(self) -> None:
        """Loads, slices and tags the raw tables of every trial, once."""
        if self._is_built:
            return
        config = self.config
        frames_by_source: dict[str, list[pl.DataFrame]] = {
            name: [] for name in SOURCES
        }
        for label, trial, table_paths in self._trials():
            trial_tag = {"label": label, "trial": trial}
            for source, frame in _trial_tables(table_paths, config):
                if frame.height:
                    frames_by_source[source].append(
                        _with_constants(frame, trial_tag)
                    )
        self._raw_tables = {
            name: pl.concat(frames, how="diagonal_relaxed")
            for name, frames in frames_by_source.items()
            if frames
        }
        self._is_built = True

    def available(self) -> list[str]:
        """Returns the raw sources this report has.

        Returns:
            Source names, in ``SOURCES`` order.

        Raises:
            spec.NotApplicableError: If there is no report or it has no cached
                trace tables.
        """
        self._build_raw()
        return [name for name in SOURCES if name in self._raw_tables]

    def frame(
        self, name: str, opts: Mapping[str, Any] | None = None
    ) -> pl.DataFrame:
        """Returns a raw table with the common plot filters applied.

        Args:
            name: Source name, one of ``SOURCES``.
            opts: Resolved plot options; ``labels``, ``rank``, ``device``,
                ``steps`` and ``where`` filter the rows. Filters on a column
                the table lacks (rank, device, step index) are skipped.

        Returns:
            The filtered table.

        Raises:
            spec.NotApplicableError: If the source is unknown, missing from the
                report, or empty after filtering.
            ValueError: If the ``where`` predicate is invalid SQL.
        """
        if name not in SOURCES:
            raise spec.NotApplicableError(
                f"unknown source '{name}'. Sources: {', '.join(SOURCES)}"
            )
        self._build_raw()
        if name not in self._raw_tables:
            available = ", ".join(self.available()) or "none"
            raise spec.NotApplicableError(
                f"source '{name}' is not in this report "
                f"(available: {available})"
            )
        table = self._raw_tables[name]
        options = opts or {}
        if options.get("labels"):
            table = table.filter(pl.col("label").is_in(options["labels"]))
        if options.get("rank") and "rank" in table.columns:
            ranks = [int(rank) for rank in options["rank"]]
            table = table.filter(pl.col("rank").is_in(ranks))
        if options.get("device") and "device" in table.columns:
            devices = [int(device) for device in options["device"]]
            table = table.filter(pl.col("device").is_in(devices))
        if options.get("steps") and "idx" in table.columns:
            first_step, end_step = options["steps"]
            if first_step is not None:
                table = table.filter(pl.col("idx") >= first_step)
            if end_step is not None:
                table = table.filter(pl.col("idx") < end_step)
        if options.get("where"):
            table = _sql_filter(table, str(options["where"]))
        if table.is_empty():
            raise spec.NotApplicableError(
                f"source '{name}' is empty after filters"
            )
        return table

    # SQL.

    def sql(
        self, query: str, opts: Mapping[str, Any] | None = None
    ) -> pl.DataFrame:
        """Runs DuckDB SQL over the named tables the query mentions.

        Raw sources are registered under their name (see
        ``cta plot --list-sources``) with ``opts`` filters applied; report
        tables as ``report_<aspect>`` (summary) and
        ``report_<aspect>_<table>`` (details).

        Args:
            query: The SQL query.
            opts: Resolved plot options used to filter the raw tables.

        Returns:
            The query result.

        Raises:
            spec.NotApplicableError: If a raw table the query mentions is
                missing or empty after filtering.
            ValueError: If DuckDB rejects the query.
        """
        import duckdb

        connection = duckdb.connect()
        try:
            for name in SOURCES:
                if re.search(rf"\b{name}\b", query):
                    raw_table = self.frame(name, opts)
                    connection.register(name, raw_table.to_arrow())
            if self.report is not None:
                for aspect_name, result in self.report.aspects.items():
                    tables_by_suffix = {
                        "": result.summary,
                        **{
                            f"_{table_name}": table
                            for table_name, table in result.details.items()
                        },
                    }
                    for suffix, table in tables_by_suffix.items():
                        view_name = f"report_{aspect_name}{suffix}"
                        if (
                            re.search(rf"\b{view_name}\b", query)
                            and table.height
                        ):
                            connection.register(view_name, table.to_arrow())
            return _to_polars(connection.sql(query))
        except duckdb.Error as exc:
            raise ValueError(
                f"SQL error: {exc}\nTables: {', '.join(SOURCES)}, "
                "report_<aspect>[_<table>]"
            ) from exc
        finally:
            connection.close()

    def describe(self) -> dict[str, list[str]]:
        """Returns the columns of every table, for ``--list-sources``.

        Returns:
            Table name -> column names; raw tables first, then report tables
            under their SQL names.

        Raises:
            spec.NotApplicableError: If there is no report or it has no cached
                trace tables.
        """
        columns_by_table = {
            name: self._raw_tables[name].columns for name in self.available()
        }
        if self.report is not None:
            for aspect_name, result in self.report.aspects.items():
                columns_by_table[f"report_{aspect_name}"] = (
                    result.summary.columns
                )
                for table_name, table in result.details.items():
                    columns_by_table[f"report_{aspect_name}_{table_name}"] = (
                        table.columns
                    )
        return columns_by_table


def _trial_tables(
    table_paths: Mapping[str, pathlib.Path], config: models.CompareConfig
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yields (source name, frame) for every raw table of one trial.

    Args:
        table_paths: Cache table key -> Parquet path, for one trial.
        config: Analysis config with the step slicing options.

    Yields:
        The trace-format tables (see :func:`_trace_format_tables`), then the
        unsliced profiler tables. Frames may be empty.
    """
    trace_format = next(
        (
            candidate
            for candidate in _TRACE_FORMAT_PRECEDENCE
            if f"{candidate}.steps" in table_paths
        ),
        None,
    )
    if trace_format is not None:
        yield from _trace_format_tables(table_paths, trace_format, config)
    for source, table_key in _UNSLICED_TABLES:
        if table_key in table_paths:
            yield source, pl.read_parquet(table_paths[table_key])


def _trace_format_tables(
    table_paths: Mapping[str, pathlib.Path],
    trace_format: str,
    config: models.CompareConfig,
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yields the sliced and timed tables of one trace format of one trial.

    Steps and kernels are sliced like the aspects slice them. Steps get a
    per-rank step index ``idx``; kernels get the ``idx`` of their step. Steps,
    kernels and GPU metrics get times relative to their rank's origin.

    Args:
        table_paths: Cache table key -> Parquet path, for one trial.
        trace_format: Format whose ``<format>.*`` tables are read.
        config: Analysis config with the step slicing options.

    Yields:
        (source name, frame) for steps, kernels, runtime, gpu_metrics and
        osrt, in that order, for the tables the trial has. Frames may be
        empty.
    """
    from cta.aspects.device import step_slicing

    steps_raw = pl.read_parquet(table_paths[f"{trace_format}.steps"])
    kernels_key = f"{trace_format}.kernels"
    kernels_raw = (
        pl.read_parquet(table_paths[kernels_key])
        if kernels_key in table_paths
        else pl.DataFrame()
    )
    # Same slicing as the aspects, also when there are no kernels.
    steps, kernels = step_slicing.slice_steps_and_kernels(
        steps_raw, kernels_raw, config
    )
    steps = _with_step_index(steps)
    origins = _rank_origins(steps, kernels)
    if kernels.height:
        step_index = steps.select("rank", "step", "idx").unique(
            ["rank", "step"]
        )
        kernels = kernels.join(step_index, on=["rank", "step"], how="left")
        kernels = _with_times(kernels, origins)
    yield "steps", _with_times(steps, origins)
    yield "kernels", kernels
    runtime_key = f"{trace_format}.runtime"
    if runtime_key in table_paths:
        yield "runtime", pl.read_parquet(table_paths[runtime_key])
    gpu_metrics_key = f"{trace_format}.gpu_metrics"
    if gpu_metrics_key in table_paths:
        gpu_metrics = step_slicing.gpu_metrics_in_steps(
            pl.read_parquet(table_paths[gpu_metrics_key]), steps
        )
        if gpu_metrics.height:
            yield "gpu_metrics", _with_sample_times(gpu_metrics, origins)
    osrt_key = f"{trace_format}.osrt"
    if osrt_key in table_paths:
        yield "osrt", pl.read_parquet(table_paths[osrt_key])


def _with_step_index(steps: pl.DataFrame) -> pl.DataFrame:
    """Sorts steps by rank and start, numbering them per rank in ``idx``."""
    step_index = (
        pl.col("start_ns").cum_count().over("rank").cast(pl.Int32) - 1
    ).alias("idx")
    return steps.sort("rank", "start_ns").with_columns(step_index)


def _rank_origins(steps: pl.DataFrame, kernels: pl.DataFrame) -> pl.DataFrame:
    """Returns each rank's time origin: its earliest step or kernel start.

    Args:
        steps: Step table with ``rank`` and ``start_ns`` columns.
        kernels: Kernel table with the same columns; may be empty.

    Returns:
        One row per rank, with the origin in the temporary ``_t0`` column.
    """
    origins = steps.group_by("rank").agg(pl.col("start_ns").min().alias("_t0"))
    if not kernels.height:
        return origins
    kernel_origins = kernels.group_by("rank").agg(
        pl.col("start_ns").min().alias("_t0")
    )
    return (
        pl.concat([origins, kernel_origins])
        .group_by("rank")
        .agg(pl.col("_t0").min())
    )


def _with_constants(
    df: pl.DataFrame, values: Mapping[str, Any]
) -> pl.DataFrame:
    """Adds one constant column per entry of ``values`` (name -> value)."""
    return df.with_columns(
        [pl.lit(value).alias(column) for column, value in values.items()]
    )


def _to_polars(result: Any) -> pl.DataFrame:
    """Converts a DuckDB query result to a Polars frame."""
    relation = cast(Any, result)
    # duckdb >= 2 has to_arrow_table; older versions have fetch_arrow_table.
    table = (
        relation.to_arrow_table()
        if hasattr(relation, "to_arrow_table")
        else relation.fetch_arrow_table()
    )
    converted = pl.from_arrow(table)
    if isinstance(converted, pl.DataFrame):
        return converted
    return converted.to_frame()


def _with_times(df: pl.DataFrame, origins: pl.DataFrame) -> pl.DataFrame:
    """Adds ``t_ms``/``end_ms`` relative to the rank origin and ``duration_ms``.

    Args:
        df: Table with ``rank``, ``start_ns``, ``end_ns`` and ``duration_ns``.
        origins: Rank origins from :func:`_rank_origins`.

    Returns:
        ``df`` with the three millisecond columns added.
    """
    return (
        df.join(origins, on="rank", how="left")
        .with_columns(
            ((pl.col("start_ns") - pl.col("_t0")) / 1e6).alias("t_ms"),
            ((pl.col("end_ns") - pl.col("_t0")) / 1e6).alias("end_ms"),
            (pl.col("duration_ns") / 1e6).alias("duration_ms"),
        )
        .drop("_t0")
    )


def _with_sample_times(
    gpu_metrics: pl.DataFrame, origins: pl.DataFrame
) -> pl.DataFrame:
    """Adds ``t_ms``: each GPU-metric sample's time relative to its rank."""
    return (
        gpu_metrics.join(origins, on="rank", how="left")
        .with_columns(
            ((pl.col("timestamp_ns") - pl.col("_t0")) / 1e6).alias("t_ms"),
            pl.col("value").alias("value"),
        )
        .drop("_t0")
    )


def _sql_filter(df: pl.DataFrame, predicate: str) -> pl.DataFrame:
    """Keeps the rows of ``df`` that satisfy a SQL predicate (DuckDB).

    Args:
        df: Any table; the predicate sees it as table ``t``.
        predicate: SQL boolean expression.

    Returns:
        The matching rows.

    Raises:
        ValueError: If DuckDB rejects the predicate.
    """
    import duckdb

    connection = duckdb.connect()
    try:
        connection.register("t", df.to_arrow())
        return _to_polars(connection.sql(f"SELECT * FROM t WHERE {predicate}"))
    except duckdb.Error as exc:
        raise ValueError(
            f"invalid 'where' predicate {predicate!r}: {exc}"
        ) from exc
    finally:
        connection.close()
