"""NVIDIA Nsight Systems (.nsys-rep and .sqlite) trace adapter.

A ``.nsys-rep`` report is read through its SQLite export. GPU activities are
attributed to the NVTX step and scope ranges of the thread that launched them,
matched through the CUDA runtime call's correlation id. This module also
defines the kernels/steps/runtime table schemas shared by the other trace
adapters.
"""

from __future__ import annotations

import ast
import contextlib
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
from typing import Any
import warnings

import numpy as np
import polars as pl

from cta import _native as native
from cta.adapters.nsys import network_ops
from cta.core import alignment
from cta.core import bundle
from cta.core import intervals
from cta.core import models
from cta.core import network
from cta.core import registry

KERNELS_SCHEMA = pl.Schema(
    {
        "step": pl.Int32(),
        "scope": pl.String(),
        # Enclosing ranges, "outermost > ... > innermost" (equal to scope for
        # the last level).
        "scope_path": pl.String(),
        "name": pl.String(),
        "demangled_name": pl.String(),
        "canonical_name": pl.String(),
        "op_family": pl.String(),
        "category": pl.String(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
        "bytes": pl.Int64(),
        "stream": pl.Int32(),
        # Per-report/process index; 0 for single-report sources.
        "rank": pl.Int32(),
        # CUDA device id within the report; 0 for single-GPU sources.
        "device": pl.Int32(),
    }
)

STEPS_SCHEMA = pl.Schema(
    {
        "step": pl.Int32(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
        "rank": pl.Int32(),
    }
)

_SCHEMA_DEFAULTS: dict[str, object] = {"rank": 0, "device": 0, "scope_path": ""}


def conform(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Selects and casts a frame to a schema, filling defaultable columns.

    Missing ``rank`` and ``device`` columns are filled with 0 and a missing
    ``scope_path`` column with "".

    Args:
        df: The frame to conform.
        schema: The target schema.

    Returns:
        The frame with exactly the columns of ``schema``, in its order and
        with its types.

    Raises:
        KeyError: ``df`` lacks a column of ``schema`` that has no default.
    """
    missing = [column for column in schema if column not in df.columns]
    if missing:
        df = df.with_columns(
            [
                pl.lit(_SCHEMA_DEFAULTS[column]).alias(column)
                for column in missing
            ]
        )
    return df.select(list(schema.keys())).cast(schema)


RUNTIME_SCHEMA = pl.Schema(
    {"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()}
)
OSRT_SCHEMA = pl.Schema(
    {"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()}
)
GPU_METRICS_SCHEMA = pl.Schema(
    {
        "metric_name": pl.String(),
        "timestamp_ns": pl.Int64(),
        "value": pl.Float64(),
        "rank": pl.Int32(),
    }
)

# The category codes the native readers give GPU activities.
GPU_CATEGORY_NAMES = {0: "compute", 1: "comm", 2: "memcpy"}
# Step ranges of one report, before duration and rank are added.
_STEP_RANGES_SCHEMA = {
    "step": pl.Int32,
    "start_ns": pl.Int64,
    "end_ns": pl.Int64,
}

_STEP_RE = re.compile(
    r"(?:profilerstep|step|iter|iteration|inference)[#_\s:-]*(\d+)",
    re.IGNORECASE,
)
# Tables of which any one marks an SQLite file as an Nsight Systems export.
_NSYS_MARKER_TABLES = frozenset(
    {
        "StringIds",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "NVTX_EVENTS",
        "GPU_METRICS",
        "GENERIC_EVENT",
    }
)


_CONVERTER_TIMEOUT_S = float(os.environ.get("CTA_CONVERTER_TIMEOUT_S", "3600"))


def _resolve_sqlite(path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Returns the SQLite export to read for a report or SQLite file.

    A ``.sqlite`` path is returned as is. For a report, a sibling ``.sqlite``
    at least as new as the report is reused; otherwise the report is exported
    into ``out_dir`` with ``nsys export`` (``nsys`` on PATH, or ``NSYS_PATH``
    or ``NSYS``). Without ``nsys``, an older sibling is used.

    Args:
        path: A ``.nsys-rep`` report or a ``.sqlite`` export.
        out_dir: The directory a fresh export is written to.

    Returns:
        The path of the SQLite export.

    Raises:
        RuntimeError: ``nsys`` is unavailable and no sibling export exists.
    """
    if path.suffix.lower() == ".sqlite":
        return path
    sibling_path = path.with_suffix(".sqlite")
    if (
        sibling_path.exists()
        and sibling_path.stat().st_mtime_ns >= path.stat().st_mtime_ns
    ):
        return sibling_path
    nsys_executable = (
        shutil.which("nsys")
        or os.environ.get("NSYS_PATH")
        or os.environ.get("NSYS")
    )
    export_path = out_dir / f"{path.stem}.sqlite"
    if nsys_executable:
        subprocess.run(
            [
                nsys_executable,
                "export",
                "-t",
                "sqlite",
                "--force-overwrite=true",
                "-o",
                str(export_path),
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_CONVERTER_TIMEOUT_S,
        )
        return export_path
    if sibling_path.exists():
        return sibling_path
    raise RuntimeError(
        f"Cannot export '{path.name}': 'nsys' CLI not found on PATH and no "
        f"exported '{sibling_path.name}' exists."
    )


def _list_tables(conn: sqlite3.Connection) -> set[str]:
    """Returns the names of all tables in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Returns the column names of a table."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


_STEP_PATTERN = (
    r"(?i)(?:profilerstep|step|iter|iteration|inference)[#_\s:-]*(\d+)"
)


def _step_pattern_of_texts(
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns, per distinct NVTX text, whether it names a step and its number.

    The pattern is matched with Polars, so the cost does not grow with a
    Python call per text.

    Args:
        texts: The distinct texts.

    Returns:
        A boolean array (the text matches ``_STEP_RE``) and an int64 array of
        the matched step number (-1 where there is no match).
    """
    digits = pl.Series(texts, dtype=pl.String).str.extract(_STEP_PATTERN, 1)
    is_step = digits.is_not_null().to_numpy()
    number = digits.cast(pl.Int64, strict=False).fill_null(-1).to_numpy().copy()
    # Digits the cast rejects (non-ASCII ones) are read by Python.
    for code in np.flatnonzero(is_step & (number < 0)).tolist():
        match = _STEP_RE.search(texts[code])
        number[code] = int(match.group(1)) if match else -1
    return is_step, number


def _ranges_by_thread(
    starts: np.ndarray, ends: np.ndarray, threads: np.ndarray
) -> dict[int, intervals.ThreadRanges]:
    """Builds the nested ranges of every thread; a range's id is its index."""
    if len(starts) == 0:
        return {}
    by_thread = np.argsort(threads, kind="stable")
    thread_ids, first = np.unique(threads[by_thread], return_index=True)
    return {
        int(thread_id): intervals.build_thread_ranges_arrays(
            starts[members], ends[members], members
        )
        for thread_id, members in zip(
            thread_ids.tolist(), np.split(by_thread, first[1:]), strict=True
        )
    }


def _label_events(
    columns: dict[str, Any], order: np.ndarray
) -> tuple[np.ndarray, pl.Series | None, np.ndarray | None, pl.DataFrame]:
    """Finds the step and NVTX scope of every GPU activity of one report.

    Named NVTX ranges matching a step pattern (``ProfilerStep#N``, ``step``,
    ``iter``, ...) are steps, and all other named ranges are scopes. Without
    any step marker, ExecutionContext/enqueue ranges (TensorRT) are the
    steps, numbered in order. Each activity gets the innermost range on its
    launching thread at launch time (see intervals.assign_ranges).

    Args:
        columns: The report as read by ``native.read_nsys_sqlite``.
        order: The order of the activities, as indices into the columns.

    Returns:
        The step of each activity (-1 if none), its scope as a String series
        and its scope path (both None when the report has no NVTX ranges at
        all), and the step ranges of the report sorted by start. With no
        activities or no NVTX ranges there are no step ranges.
    """
    empty_steps = pl.DataFrame(schema=_STEP_RANGES_SCHEMA)
    count = len(order)
    if count == 0 or len(columns["nvtx_start"]) == 0:
        return np.full(count, -1, dtype=np.int32), None, None, empty_steps

    texts: list[str] = columns["nvtx_texts"]
    code = columns["nvtx_text"]
    named = np.array([bool(text) for text in texts], dtype=bool)[code]
    code = code[named]
    start = columns["nvtx_start"][named]
    end = columns["nvtx_end"][named]
    thread = columns["nvtx_thread"][named]

    text_is_step, text_number = _step_pattern_of_texts(texts)
    is_step = text_is_step[code]
    number = text_number[code]
    # Without explicit step markers, repeated ExecutionContext/enqueue ranges
    # (TensorRT) delimit the steps.
    if not is_step.any():
        text_is_step = np.array(
            [
                (
                    "executioncontext" in text.lower()
                    or "enqueue" in text.lower()
                )
                for text in texts
            ],
            dtype=bool,
        )
        is_step = text_is_step[code]

    step_start, step_end, step_thread = (
        start[is_step],
        end[is_step],
        thread[is_step],
    )
    step_count = len(step_start)
    # A step without a number in its text takes its place among the steps.
    step_ids = np.where(
        number[is_step] >= 0, number[is_step], np.arange(step_count)
    )
    scope_start, scope_end, scope_thread = (
        start[~is_step],
        end[~is_step],
        thread[~is_step],
    )
    scope_code = code[~is_step]

    launch_ns = columns["gpu_launch_ns"][order]
    launch_thread_ids = columns["gpu_launch_thread"][order]
    gpu_start_ns = columns["gpu_start"][order]
    # Launch time, or the GPU start time where the launch is unknown.
    issue_ns = np.where(np.isnan(launch_ns), gpu_start_ns, launch_ns).astype(
        np.int64
    )

    step_ranges = _ranges_by_thread(step_start, step_end, step_thread)
    scope_ranges = _ranges_by_thread(scope_start, scope_end, scope_thread)
    step_range_ids = intervals.assign_ranges(
        step_ranges,
        step_start,
        issue_ns.astype(np.float64),
        launch_thread_ids,
        gpu_start_ns,
    )
    scope_range_ids = intervals.assign_ranges(
        scope_ranges,
        scope_start,
        issue_ns.astype(np.float64),
        launch_thread_ids,
        gpu_start_ns,
    )

    # Range id -1 (no enclosing range) indexes the trailing -1 / "".
    event_steps = np.append(step_ids, -1).astype(np.int32)[step_range_ids]
    scope_texts = pl.Series([*texts, ""], dtype=pl.String)
    event_scopes = scope_texts.gather(
        np.append(scope_code, len(texts))[scope_range_ids]
    )
    event_scope_paths = intervals.range_paths(
        scope_ranges,
        scope_range_ids,
        intervals.CodedNames(texts, scope_code),
    )
    steps = (
        pl.DataFrame(
            {
                "step": step_ids.astype(np.int32),
                "start_ns": step_start,
                "end_ns": step_end,
            },
            schema=empty_steps.schema,
        )
        # One row per range; ids may repeat across devices/loops.
        .unique(subset=["step", "start_ns", "end_ns"], maintain_order=True)
        .sort("start_ns", maintain_order=True)
    )
    return event_steps, event_scopes, event_scope_paths, steps


def _text_gpu_metrics(
    conn: sqlite3.Connection, tables: set[str]
) -> pl.DataFrame:
    """Returns the GPU metric samples an export keeps as dict-strings.

    Reads the legacy ``GPU_METRICS`` table of dict-strings, or
    ``GENERIC_EVENTS``/``GENERIC_EVENT`` dict-strings. The modern table form
    is read by the native reader.
    Dict-strings that do not parse are skipped; a legacy sample without a
    timestamp gets its row index.

    Args:
        conn: The open export.
        tables: The names of the export's tables.

    Returns:
        A frame with GPU_METRICS_SCHEMA and ``rank`` 0.
    """
    samples: list[dict[str, object]] = []

    def add_metric_dict(raw: object, timestamp_ns: int) -> None:
        """Adds one sample per entry of a dict-string, if it parses."""
        try:
            parsed = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return
        if isinstance(parsed, dict):
            for metric_name, value in parsed.items():
                samples.append(
                    {
                        "metric_name": str(metric_name),
                        "timestamp_ns": timestamp_ns,
                        "value": float(value),
                        "rank": 0,
                    }
                )

    if "GPU_METRICS" in tables:
        metric_columns = _table_columns(conn, "GPU_METRICS")
        if "value" in metric_columns or "data" in metric_columns:
            value_column = "value" if "value" in metric_columns else "data"
            timestamp_expr = (
                "rawTimestamp" if "rawTimestamp" in metric_columns else "0"
            )
            rows = conn.execute(
                f"SELECT {value_column}, {timestamp_expr} FROM GPU_METRICS"
            ).fetchall()
            for row_index, (raw_value, timestamp) in enumerate(rows):
                add_metric_dict(raw_value, int(timestamp or row_index))
    elif "GENERIC_EVENTS" in tables or "GENERIC_EVENT" in tables:
        generic_table = (
            "GENERIC_EVENTS" if "GENERIC_EVENTS" in tables else "GENERIC_EVENT"
        )
        value_column = (
            "data" if "data" in _table_columns(conn, generic_table) else "value"
        )
        rows = conn.execute(
            f"SELECT {value_column} FROM {generic_table}"
        ).fetchall()
        for row_index, (raw_value,) in enumerate(rows):
            add_metric_dict(raw_value, row_index)
    if samples:
        return pl.DataFrame(samples, schema=GPU_METRICS_SCHEMA)
    return pl.DataFrame(schema=GPU_METRICS_SCHEMA)


def _summary_frame(summary: dict[str, Any], schema: pl.Schema) -> pl.DataFrame:
    """Builds a total-duration and call-count table per API name."""
    return pl.DataFrame(
        {
            "name": pl.Series(summary["names"], dtype=pl.String),
            "duration_ns": summary["duration_ns"],
            "calls": summary["calls"],
        },
        schema=schema,
    )


def _gpu_events_frame(
    columns: dict[str, Any], order: np.ndarray
) -> pl.DataFrame:
    """Builds the GPU activities, in the given order, with demangled names."""
    codes = columns["gpu_name"][order]
    # Demangling and classifying are per name, so they run on the distinct
    # names and are spread over the activities afterwards.
    used_codes, name_rows = np.unique(codes, return_inverse=True)
    names: list[str] = columns["names"]
    distinct = alignment.enrich_ops_df(
        pl.DataFrame(
            {
                "name": pl.Series(
                    [names[code] for code in used_codes.tolist()],
                    dtype=pl.String,
                )
            }
        ),
        "name",
    )
    return pl.DataFrame(
        {
            "name": distinct["name"].gather(name_rows),
            "category": pl.Series(
                columns["gpu_category"][order]
            ).replace_strict(GPU_CATEGORY_NAMES, return_dtype=pl.String),
            "start_ns": columns["gpu_start"][order],
            "end_ns": columns["gpu_end"][order],
            "bytes": columns["gpu_bytes"][order],
            "stream": columns["gpu_stream"][order],
            "device": columns["gpu_device"][order],
        }
    ).with_columns(
        distinct["demangled_name"].gather(name_rows),
        distinct["canonical_name"].gather(name_rows),
        distinct["op_family"].gather(name_rows),
    )


def _table_gpu_metrics(columns: dict[str, Any]) -> pl.DataFrame:
    """Builds the GPU metric samples the native reader found in the table."""
    names: list[str] = columns["metric_names"]
    return pl.DataFrame(
        {
            "metric_name": pl.Series(names, dtype=pl.String).gather(
                columns["metric_name"]
            ),
            "timestamp_ns": columns["metric_timestamp"],
            "value": columns["metric_value"],
            "rank": np.zeros(len(columns["metric_name"]), dtype=np.int32),
        },
        schema=GPU_METRICS_SCHEMA,
    )


def _ingest_report(db_path: pathlib.Path, rank: int) -> dict[str, pl.DataFrame]:
    """Reads the tables of one SQLite export.

    A RuntimeWarning is issued when a report with GPU activities has no NVTX
    step markers; its whole trace then becomes step 0.

    Args:
        db_path: The SQLite export, opened read-only.
        rank: The rank written to the kernels, steps and gpu_metrics rows.

    Returns:
        The ``kernels``, ``steps``, ``runtime``, ``osrt``, ``gpu_metrics``
        and ``network_ops`` frames, by table name.
    """
    columns = native.read_nsys_sqlite(str(db_path))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _list_tables(conn)
        runtime = _summary_frame(columns["runtime"], RUNTIME_SCHEMA)
        osrt = _summary_frame(columns["osrt"], OSRT_SCHEMA)
        gpu_metrics = (
            _table_gpu_metrics(columns)
            if columns["has_metrics"]
            else _text_gpu_metrics(conn, tables)
        ).with_columns(pl.lit(rank, dtype=pl.Int32).alias("rank"))
    finally:
        conn.close()

    # Activities in start order (ties keep their table order).
    order = np.argsort(columns["gpu_start"], kind="stable")
    event_steps, event_scopes, event_scope_paths, steps = _label_events(
        columns, order
    )
    if len(order):
        events = _gpu_events_frame(columns, order).with_columns(
            pl.Series("step", event_steps, dtype=pl.Int32),
            (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"),
            pl.lit(rank).alias("rank"),
        )
        if event_scopes is None:
            events = events.with_columns(pl.lit("").alias("scope"))
        else:
            events = events.with_columns(
                event_scopes.alias("scope"),
                pl.Series("scope_path", event_scope_paths, dtype=pl.String),
            )
        kernels = conform(events, KERNELS_SCHEMA)
    else:
        kernels = pl.DataFrame(schema=KERNELS_SCHEMA)

    if steps.is_empty() and kernels.height:
        warnings.warn(
            f"{db_path.name}: no NVTX step markers (e.g. 'ProfilerStep#N'); "
            "the whole trace is treated as one step, "
            "so --warmup-steps/--max-steps have no effect.",
            RuntimeWarning,
            stacklevel=2,
        )
        steps = pl.DataFrame(
            {
                "step": [0],
                "start_ns": [int(kernels["start_ns"].to_numpy().min())],
                "end_ns": [int(kernels["end_ns"].to_numpy().max())],
            },
            schema=_STEP_RANGES_SCHEMA,
        )
    steps = conform(
        steps.with_columns(
            (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"),
            pl.lit(rank).alias("rank"),
        ),
        STEPS_SCHEMA,
    )
    with contextlib.closing(
        sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ) as host_conn:
        host_ops = network_ops.host_network_ops(host_conn, tables, rank, steps)
    return {
        "kernels": kernels,
        "steps": steps,
        "runtime": runtime,
        "osrt": osrt,
        "gpu_metrics": gpu_metrics,
        "network_ops": pl.concat(
            [network.ops_from_kernels(kernels), host_ops]
        ),
    }


@registry.register_adapter("nsys", version="2.3")
class NsysAdapter:
    """Reads Nsight Systems reports and SQLite exports, one per rank.

    Attributes:
        format_name: The registered format name, ``"nsys"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "nsys"
    version = "2.3"

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file is an Nsight Systems report or export.

        Args:
            path: The candidate file.

        Returns:
            True for a ``.nsys-rep`` report, or a ``.sqlite`` database with
            at least one table typical of an Nsight Systems export.
        """
        suffix = path.suffix.lower()
        if suffix == ".nsys-rep":
            return True
        if suffix == ".sqlite":
            try:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                    tables = _list_tables(conn)
                return bool(tables & _NSYS_MARKER_TABLES)
            except sqlite3.Error:
                return False
        return False

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Ingests every report of the artifact (one per rank).

        Rows carry ``rank`` and ``device``. Kernels and steps are sorted by
        rank and start time; runtime and OS runtime summaries are summed by
        name over all reports.

        Args:
            artifact: The artifact whose reports or exports are read.
            out_dir: The directory exports and Parquet tables are written to.

        Returns:
            The paths of the ``kernels``, ``steps``, ``runtime``, ``osrt``,
            ``gpu_metrics`` and ``network_ops`` Parquet tables, by table name.

        Raises:
            RuntimeError: A report has neither ``nsys`` nor an SQLite export.
        """
        per_report: list[dict[str, pl.DataFrame]] = []
        for report_index, path in enumerate(sorted(artifact.paths)):
            db_path = _resolve_sqlite(path, out_dir)
            rank = bundle.infer_rank(path, report_index)
            per_report.append(_ingest_report(db_path, rank))

        def concat_reports(key: str, schema: pl.Schema) -> pl.DataFrame:
            """Concatenates one table over all reports (empty ones skipped)."""
            frames = [
                report[key] for report in per_report if report[key].height
            ]
            return pl.concat(frames) if frames else pl.DataFrame(schema=schema)

        def merge_summaries(key: str, schema: pl.Schema) -> pl.DataFrame:
            """Sums one runtime summary table by name over all reports."""
            summaries = concat_reports(key, schema)
            if summaries.is_empty():
                return summaries
            return (
                summaries.group_by("name")
                .agg(pl.col("duration_ns").sum(), pl.col("calls").sum())
                .cast(schema)
            )

        merged_tables = {
            "kernels": concat_reports("kernels", KERNELS_SCHEMA).sort(
                ["rank", "start_ns"]
            ),
            "network_ops": concat_reports(
                "network_ops", network.NETWORK_OPS_SCHEMA
            ).sort(["rank", "start_ns"]),
            "steps": concat_reports("steps", STEPS_SCHEMA).sort(
                ["rank", "start_ns"]
            ),
            "runtime": merge_summaries("runtime", RUNTIME_SCHEMA),
            "osrt": merge_summaries("osrt", OSRT_SCHEMA),
            "gpu_metrics": concat_reports("gpu_metrics", GPU_METRICS_SCHEMA),
        }
        return bundle.write_tables(out_dir, merged_tables)
