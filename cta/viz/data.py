"""Data access for plots: named raw tables (kernels, steps, gpu_metrics, ...) and report tables, plus SQL.

Raw tables come from the cache paths the report keeps. They are sliced with the same function the aspects use
(``--warmup-steps/--max-steps`` per rank) so plots and tables always agree, resolved through format-qualified keys
with explicit precedence, and carry ``label`` (group label), ``trial``, and times relative to each rank's first step
(ranks have independent clocks, so absolute times are never mixed across ranks).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from cta.core.models import CompareConfig
from cta.viz.spec import NotApplicable

if TYPE_CHECKING:
    from cta.core.models import ComparisonReport

SOURCES = ("kernels", "steps", "gpu_metrics", "runtime", "ncu_kernels", "ncu_findings", "jtop", "osrt")
_FMT_PRECEDENCE = ("nsys", "kineto", "jax", "trtexec_timing")


class DataContext:
    def __init__(self, report: ComparisonReport | None = None, *, history: str | None = None) -> None:
        self.report = report
        self.history = history
        self._raw: dict[str, pl.DataFrame] = {}
        self._built = False

    # ---- report side -------------------------------------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        return [g.label for g in self.report.groups] if self.report else []

    @property
    def config(self) -> CompareConfig:
        return (self.report.config if self.report else None) or CompareConfig()

    def require_report(self) -> ComparisonReport:
        if self.report is None:
            raise NotApplicable("this plot needs a trace/report (give trace paths or --manifest)")
        return self.report

    def report_table(self, aspect: str, table: str = "summary") -> pl.DataFrame:
        rep = self.require_report()
        if aspect not in rep.aspects:
            raise NotApplicable(f"needs the '{aspect}' aspect (have: {', '.join(rep.aspects) or 'none'})")
        try:
            return rep.aspects[aspect].to_polars(table)
        except KeyError as exc:
            raise NotApplicable(str(exc.args[0])) from exc

    def has_aspect(self, aspect: str) -> bool:
        return self.report is not None and aspect in self.report.aspects

    def metric(self, aspect: str, metric: str) -> dict[str, float]:
        """Per-label value of a summary metric (missing labels are omitted)."""
        df = self.report_table(aspect)
        row = df.filter(pl.col("metric") == metric)
        if row.is_empty():
            return {}
        r = row.row(0, named=True)
        labels = self.labels
        if len(labels) == 2 and "baseline" in r and "candidate" in r:
            return {labels[0]: float(r["baseline"] or 0.0), labels[1]: float(r["candidate"] or 0.0)}
        return {lb: float(r[lb]) for lb in labels if r.get(lb) is not None}

    # ---- raw tables --------------------------------------------------------------------------------------------
    def _trials(self) -> list[tuple[str, int, dict[str, Path]]]:
        rep = self.require_report()
        out = []
        for g in rep.groups:
            for i, tmap in enumerate(rep.sources.get(g.group_id, [])):
                out.append((g.label, i, tmap))
        if not out:
            raise NotApplicable("the report has no cached trace tables (re-run the analysis)")
        return out

    def _build_raw(self) -> None:
        if self._built:
            return
        from cta.aspects.device import _gpu_metrics_in_steps, _slice_steps_and_kernels

        cfg = self.config
        acc: dict[str, list[pl.DataFrame]] = {n: [] for n in SOURCES}
        for label, trial, tmap in self._trials():
            tag = {"label": label, "trial": trial}

            def add(name: str, df: pl.DataFrame, _tag: dict[str, Any] = tag) -> None:
                if df.height:
                    acc[name].append(df.with_columns([pl.lit(v).alias(k) for k, v in _tag.items()]))

            fmt = next((f for f in _FMT_PRECEDENCE if f"{f}.steps" in tmap), None)
            if fmt is not None:
                steps_raw = pl.read_parquet(tmap[f"{fmt}.steps"])
                kern_raw = pl.read_parquet(tmap[f"{fmt}.kernels"]) if f"{fmt}.kernels" in tmap else pl.DataFrame()
                steps, kern = _slice_steps_and_kernels(steps_raw, kern_raw, cfg)  # same slicing as the aspects, also when there are no kernels
                steps = steps.sort("rank", "start_ns").with_columns((pl.col("start_ns").cum_count().over("rank").cast(pl.Int32) - 1).alias("idx"))
                t0 = steps.group_by("rank").agg(pl.col("start_ns").min().alias("_t0"))
                if kern.height:
                    kern = kern.join(steps.select("rank", "step", "idx").unique(["rank", "step"]), on=["rank", "step"], how="left")
                    t0 = pl.concat([t0, kern.group_by("rank").agg(pl.col("start_ns").min().alias("_t0"))]).group_by("rank").agg(pl.col("_t0").min())
                    kern = _with_times(kern, t0)
                add("steps", _with_times(steps, t0))
                add("kernels", kern)
                if f"{fmt}.runtime" in tmap:
                    add("runtime", pl.read_parquet(tmap[f"{fmt}.runtime"]))
                if f"{fmt}.gpu_metrics" in tmap:
                    gm = _gpu_metrics_in_steps(pl.read_parquet(tmap[f"{fmt}.gpu_metrics"]), steps)
                    if gm.height:
                        gm = gm.join(t0, on="rank", how="left").with_columns(
                            ((pl.col("timestamp_ns") - pl.col("_t0")) / 1e6).alias("t_ms"), pl.col("value").alias("value")
                        ).drop("_t0")
                        add("gpu_metrics", gm)
                if f"{fmt}.osrt" in tmap:
                    add("osrt", pl.read_parquet(tmap[f"{fmt}.osrt"]))
            for src, key in (("ncu_kernels", "ncu.ncu_kernels"), ("ncu_findings", "ncu.ncu_findings"), ("jtop", "jtop.samples")):
                if key in tmap:
                    add(src, pl.read_parquet(tmap[key]))
        self._raw = {n: pl.concat(frames, how="diagonal_relaxed") for n, frames in acc.items() if frames}
        self._built = True

    def available(self) -> list[str]:
        self._build_raw()
        return [n for n in SOURCES if n in self._raw]

    def frame(self, name: str, opts: Mapping[str, Any] | None = None) -> pl.DataFrame:
        """A raw table with the common filters (labels, rank, device, steps, where) applied."""
        if name not in SOURCES:
            raise NotApplicable(f"unknown source '{name}'. Sources: {', '.join(SOURCES)}")
        self._build_raw()
        if name not in self._raw:
            raise NotApplicable(f"source '{name}' is not in this report (available: {', '.join(self.available()) or 'none'})")
        df = self._raw[name]
        o = opts or {}
        if o.get("labels"):
            df = df.filter(pl.col("label").is_in(o["labels"]))
        if o.get("rank") and "rank" in df.columns:
            df = df.filter(pl.col("rank").is_in([int(r) for r in o["rank"]]))
        if o.get("device") and "device" in df.columns:
            df = df.filter(pl.col("device").is_in([int(d) for d in o["device"]]))
        if o.get("steps") and "idx" in df.columns:
            lo, hi = o["steps"]
            if lo is not None:
                df = df.filter(pl.col("idx") >= lo)
            if hi is not None:
                df = df.filter(pl.col("idx") < hi)
        if o.get("where"):
            df = _sql_filter(df, str(o["where"]))
        if df.is_empty():
            raise NotApplicable(f"source '{name}' is empty after filters")
        return df

    # ---- SQL ---------------------------------------------------------------------------------------------------
    def sql(self, query: str, opts: Mapping[str, Any] | None = None) -> pl.DataFrame:
        """Run DuckDB SQL over the named tables used in the query (see ``cta plot --list-sources``)."""
        import duckdb

        con = duckdb.connect()
        try:
            for name in SOURCES:
                if re.search(rf"\b{name}\b", query):
                    con.register(name, self.frame(name, opts).to_arrow())
            if self.report is not None:
                for asp_name, asp in self.report.aspects.items():
                    for tname, df in {"": asp.summary, **{f"_{t}": d for t, d in asp.details.items()}}.items():
                        view = f"report_{asp_name}{tname}"
                        if re.search(rf"\b{view}\b", query) and df.height:
                            con.register(view, df.to_arrow())
            return _to_polars(con.sql(query))
        except duckdb.Error as exc:
            raise ValueError(f"SQL error: {exc}\nTables: {', '.join(SOURCES)}, report_<aspect>[_<table>]") from exc
        finally:
            con.close()

    def describe(self) -> dict[str, list[str]]:
        """Source name -> column names, for ``--list-sources`` (raw tables first, then report tables)."""
        out = {n: self._raw[n].columns for n in self.available()}
        if self.report is not None:
            for asp_name, asp in self.report.aspects.items():
                out[f"report_{asp_name}"] = asp.summary.columns
                for t, d in asp.details.items():
                    out[f"report_{asp_name}_{t}"] = d.columns
        return out


def _to_polars(result: Any) -> pl.DataFrame:
    res = cast(Any, result)
    table = res.to_arrow_table() if hasattr(res, "to_arrow_table") else res.fetch_arrow_table()  # duckdb >= 2 / older
    out = pl.from_arrow(table)
    return out if isinstance(out, pl.DataFrame) else out.to_frame()


def _with_times(df: pl.DataFrame, t0: pl.DataFrame) -> pl.DataFrame:
    return (
        df.join(t0, on="rank", how="left")
        .with_columns(
            ((pl.col("start_ns") - pl.col("_t0")) / 1e6).alias("t_ms"),
            ((pl.col("end_ns") - pl.col("_t0")) / 1e6).alias("end_ms"),
            (pl.col("duration_ns") / 1e6).alias("duration_ms"),
        )
        .drop("_t0")
    )


def _sql_filter(df: pl.DataFrame, predicate: str) -> pl.DataFrame:
    import duckdb

    con = duckdb.connect()
    try:
        con.register("t", df.to_arrow())
        return _to_polars(con.sql(f"SELECT * FROM t WHERE {predicate}"))
    except duckdb.Error as exc:
        raise ValueError(f"invalid 'where' predicate {predicate!r}: {exc}") from exc
    finally:
        con.close()
