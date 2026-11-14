"""NVIDIA Nsight Systems (.nsys-rep and .sqlite) trace adapter.

A ``.nsys-rep`` report is read through its SQLite export. GPU activities are
attributed to the NVTX step and scope ranges of the thread that launched them,
matched through the CUDA runtime call's correlation id. This module also
defines the kernels/steps/runtime table schemas shared by the other trace
adapters.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import warnings

import numpy as np
import polars as pl

from cta.core import alignment
from cta.core import bundle
from cta.core import intervals
from cta.core import models
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

# GPU activities of one report, before labelling.
_GPU_EVENTS_SCHEMA = {
    "name": pl.String,
    "category": pl.String,
    "start_ns": pl.Int64,
    "end_ns": pl.Int64,
    "bytes": pl.Int64,
    "stream": pl.Int32,
    "device": pl.Int32,
    "correlation_id": pl.Int64,
}
_LAUNCHES_SCHEMA = {
    "correlation_id": pl.Int64,
    "launch_thread_id": pl.Int64,
    "launch_ns": pl.Int64,
}
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
# CUPTI memcpy kinds: 1 HtoD, 2 DtoH, 8 DtoD, 9 HtoH, 10 PtoP (others are rare
# array/unified-memory copies).
_MEMCPY_NAME_BY_KIND = {
    1: "Memcpy_H2D",
    2: "Memcpy_D2H",
    8: "Memcpy_D2D",
    9: "Memcpy_H2H",
    10: "Memcpy_P2P",
}
_NVTX_RANGE_EVENT_TYPES = (59, 60)  # NvtxPushPopRange, NvtxStartEndRange
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


def _column_or_default(columns: set[str], name: str, default: str) -> str:
    """Returns the SQL column ``t.<name>`` if it exists, else ``default``."""
    return f"t.{name}" if name in columns else default


def _nvtx_ranges(
    conn: sqlite3.Connection, tables: set[str]
) -> list[tuple[str, int, int, int]]:
    """Returns (text, start ns, end ns, globalTid) of every closed NVTX range.

    Only push/pop and start/end ranges are read when the event type is known.
    The text comes from the range itself or from ``StringIds``; globalTid is
    0 when unknown.

    Args:
        conn: The open export.
        tables: The names of the export's tables.

    Returns:
        The ranges in database order; empty without NVTX text.
    """
    if "NVTX_EVENTS" not in tables:
        return []
    nvtx_columns = _table_columns(conn, "NVTX_EVENTS")
    thread_expr = "n.globalTid" if "globalTid" in nvtx_columns else "0"
    if "eventType" in nvtx_columns:
        event_type_filter = f" AND n.eventType IN {_NVTX_RANGE_EVENT_TYPES}"
    else:
        event_type_filter = ""
    where_clause = (
        f"WHERE n.end IS NOT NULL AND n.end > n.start{event_type_filter}"
    )
    if "textId" in nvtx_columns and "StringIds" in tables:
        if "text" in nvtx_columns:
            text_expr = "COALESCE(n.text, s.value, '')"
        else:
            text_expr = "COALESCE(s.value, '')"
        sql = (
            f"SELECT {text_expr}, n.start, n.end, {thread_expr} "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id "
            f"{where_clause}"
        )
    elif "text" in nvtx_columns:
        sql = (
            f"SELECT COALESCE(n.text, ''), n.start, n.end, {thread_expr} "
            f"FROM NVTX_EVENTS n {where_clause}"
        )
    else:
        return []
    return [
        (str(row[0]), int(row[1]), int(row[2]), int(row[3] or 0))
        for row in conn.execute(sql).fetchall()
    ]


def _gpu_event_row(
    name: str,
    category: str,
    start_ns: int,
    end_ns: int,
    nbytes: int,
    stream: int,
    device: int,
    correlation_id: int,
) -> dict[str, object]:
    """Builds one row of the GPU activities frame."""
    return {
        "name": name,
        "category": category,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "bytes": nbytes,
        "stream": stream,
        "device": device,
        "correlation_id": correlation_id,
    }


def _gpu_events(conn: sqlite3.Connection, tables: set[str]) -> pl.DataFrame:
    """Returns the kernels, memcpys and memsets of an export.

    Each activity carries its device, stream and the correlation id of its
    launching API call. Activities that do not end after they start are
    skipped; kernels with "nccl" in their name are ``comm``, other kernels
    ``compute`` and copies and sets ``memcpy``.

    Args:
        conn: The open export.
        tables: The names of the export's tables.

    Returns:
        One row per activity: kernels, then memcpys, then memsets.
    """
    has_string_ids = "StringIds" in tables
    rows: list[dict[str, object]] = []

    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        kernel_columns = _table_columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_column = (
            "demangledName"
            if "demangledName" in kernel_columns
            else "shortName"
        )
        if has_string_ids:
            name_expr = f"COALESCE(s.value, CAST(t.{name_column} AS TEXT))"
            join_clause = f"LEFT JOIN StringIds s ON t.{name_column} = s.id"
        else:
            name_expr = f"CAST(t.{name_column} AS TEXT)"
            join_clause = ""
        device_expr = _column_or_default(kernel_columns, "deviceId", "0")
        stream_expr = _column_or_default(kernel_columns, "streamId", "0")
        correlation_expr = _column_or_default(
            kernel_columns, "correlationId", "-1"
        )
        sql = (
            f"SELECT {name_expr}, t.start, t.end, {device_expr}, "
            f"{stream_expr}, {correlation_expr} "
            f"FROM CUPTI_ACTIVITY_KIND_KERNEL t {join_clause} "
            "WHERE t.end > t.start"
        )
        for row in conn.execute(sql):
            name, start_ns, end_ns, device, stream, correlation_id = row
            category = "comm" if "nccl" in str(name).lower() else "compute"
            rows.append(
                _gpu_event_row(
                    name=str(name),
                    category=category,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    nbytes=0,
                    stream=stream,
                    device=device,
                    correlation_id=correlation_id,
                )
            )

    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        memcpy_columns = _table_columns(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
        copy_kind_expr = _column_or_default(memcpy_columns, "copyKind", "0")
        bytes_expr = _column_or_default(memcpy_columns, "bytes", "0")
        device_expr = _column_or_default(memcpy_columns, "deviceId", "0")
        stream_expr = _column_or_default(memcpy_columns, "streamId", "0")
        correlation_expr = _column_or_default(
            memcpy_columns, "correlationId", "-1"
        )
        sql = (
            f"SELECT {copy_kind_expr}, t.start, t.end, {bytes_expr}, "
            f"{device_expr}, {stream_expr}, {correlation_expr} "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY t WHERE t.end > t.start"
        )
        for row in conn.execute(sql):
            (
                copy_kind,
                start_ns,
                end_ns,
                nbytes,
                device,
                stream,
                correlation_id,
            ) = row
            rows.append(
                _gpu_event_row(
                    name=_MEMCPY_NAME_BY_KIND.get(
                        int(copy_kind or 0), "Memcpy_Other"
                    ),
                    category="memcpy",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    nbytes=nbytes,
                    stream=stream,
                    device=device,
                    correlation_id=correlation_id,
                )
            )

    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        memset_columns = _table_columns(conn, "CUPTI_ACTIVITY_KIND_MEMSET")
        bytes_expr = _column_or_default(memset_columns, "bytes", "0")
        device_expr = _column_or_default(memset_columns, "deviceId", "0")
        stream_expr = _column_or_default(memset_columns, "streamId", "0")
        correlation_expr = _column_or_default(
            memset_columns, "correlationId", "-1"
        )
        sql = (
            f"SELECT t.start, t.end, {bytes_expr}, {device_expr}, "
            f"{stream_expr}, {correlation_expr} "
            "FROM CUPTI_ACTIVITY_KIND_MEMSET t WHERE t.end > t.start"
        )
        for row in conn.execute(sql):
            start_ns, end_ns, nbytes, device, stream, correlation_id = row
            rows.append(
                _gpu_event_row(
                    name="Memset",
                    category="memcpy",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    nbytes=nbytes,
                    stream=stream,
                    device=device,
                    correlation_id=correlation_id,
                )
            )

    return pl.DataFrame(rows, schema=_GPU_EVENTS_SCHEMA)


def _launch_info(conn: sqlite3.Connection, tables: set[str]) -> pl.DataFrame:
    """Returns the launching thread and time of each correlation id.

    The data comes from the CUDA runtime table; the first row of a repeated
    correlation id wins.

    Args:
        conn: The open export.
        tables: The names of the export's tables.

    Returns:
        ``correlation_id``, ``launch_thread_id`` and ``launch_ns`` columns;
        empty if the runtime table or its needed columns are missing.
    """
    empty = pl.DataFrame(schema=_LAUNCHES_SCHEMA)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
        return empty
    runtime_columns = _table_columns(conn, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if (
        "correlationId" not in runtime_columns
        or "globalTid" not in runtime_columns
    ):
        return empty
    rows = conn.execute(
        "SELECT correlationId, globalTid, start "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchall()
    launches = pl.DataFrame(rows, schema=empty.schema, orient="row")
    return launches.unique(subset="correlation_id", keep="first")


def _range_starts(
    ranges_by_thread: dict[int, list[tuple[int, int, int]]], count: int
) -> np.ndarray:
    """Returns the start ns of every range, indexed by its global range id."""
    starts = np.zeros(count, dtype=np.int64)
    for ranges in ranges_by_thread.values():
        for start_ns, _, range_id in ranges:
            starts[range_id] = start_ns
    return starts


def _label_events(
    events: pl.DataFrame,
    launches: pl.DataFrame,
    nvtx: list[tuple[str, int, int, int]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attaches ``step``/``scope`` to every GPU activity of one report.

    Named NVTX ranges matching a step pattern (``ProfilerStep#N``, ``step``,
    ``iter``, ...) are steps, and all other named ranges are scopes. Without
    any step marker, ExecutionContext/enqueue ranges (TensorRT) are the
    steps, numbered in order. Each activity gets the innermost range on its
    launching thread at launch time (see intervals.assign_ranges).

    Args:
        events: The report's GPU activities (see _gpu_events).
        launches: The launch thread and time per correlation id.
        nvtx: (text, start ns, end ns, globalTid) of the NVTX ranges.

    Returns:
        The activities with ``step`` (-1 if none), ``scope`` and
        ``scope_path`` columns, and the step ranges of the report sorted by
        start. With no activities or no NVTX ranges, ``step`` is -1 and
        ``scope`` "" everywhere (no ``scope_path``) and there are no step
        ranges.
    """
    empty_steps = pl.DataFrame(schema=_STEP_RANGES_SCHEMA)
    if events.height == 0 or not nvtx:
        unlabelled = events.with_columns(
            pl.lit(-1, dtype=pl.Int32).alias("step"), pl.lit("").alias("scope")
        )
        return unlabelled, empty_steps

    named_ranges = [nvtx_range for nvtx_range in nvtx if nvtx_range[0]]
    is_step_range = [bool(_STEP_RE.search(text)) for text, *_ in named_ranges]
    # Without explicit step markers, repeated ExecutionContext/enqueue ranges
    # (TensorRT) delimit the steps.
    if not any(is_step_range):
        is_step_range = [
            ("executioncontext" in text.lower() or "enqueue" in text.lower())
            for text, *_ in named_ranges
        ]

    # (step id, start ns, end ns) of every step range.
    step_rows: list[tuple[int, int, int]] = []
    # Thread -> (start ns, end ns, index into step_ids / scope_names).
    step_ranges_by_thread: dict[int, list[tuple[int, int, int]]] = {}
    scope_ranges_by_thread: dict[int, list[tuple[int, int, int]]] = {}
    step_ids: list[int] = []
    scope_names: list[str] = []
    for (text, start_ns, end_ns, thread_id), is_step in zip(
        named_ranges, is_step_range, strict=True
    ):
        if is_step:
            step_match = _STEP_RE.search(text)
            step_id = int(step_match.group(1)) if step_match else len(step_ids)
            step_ranges_by_thread.setdefault(thread_id, []).append(
                (start_ns, end_ns, len(step_ids))
            )
            step_ids.append(step_id)
            step_rows.append((step_id, start_ns, end_ns))
        else:
            scope_ranges_by_thread.setdefault(thread_id, []).append(
                (start_ns, end_ns, len(scope_names))
            )
            scope_names.append(text)

    events_with_launch = events.join(launches, on="correlation_id", how="left")
    launch_ns = (
        events_with_launch["launch_ns"]
        .cast(pl.Float64)
        .fill_null(float("nan"))
        .to_numpy()
    )
    launch_thread_ids = (
        events_with_launch["launch_thread_id"].fill_null(-1).to_numpy()
    )
    gpu_start_ns = events_with_launch["start_ns"].to_numpy()
    # Launch time, or the GPU start time where the launch is unknown.
    issue_ns = np.where(np.isnan(launch_ns), gpu_start_ns, launch_ns).astype(
        np.int64
    )

    step_ranges = {
        thread_id: intervals.build_thread_ranges(list(ranges))
        for thread_id, ranges in step_ranges_by_thread.items()
    }
    scope_ranges = {
        thread_id: intervals.build_thread_ranges(list(ranges))
        for thread_id, ranges in scope_ranges_by_thread.items()
    }
    step_range_ids = intervals.assign_ranges(
        step_ranges,
        _range_starts(step_ranges_by_thread, len(step_ids)),
        issue_ns.astype(np.float64),
        launch_thread_ids,
        gpu_start_ns,
    )
    scope_range_ids = intervals.assign_ranges(
        scope_ranges,
        _range_starts(scope_ranges_by_thread, len(scope_names)),
        issue_ns.astype(np.float64),
        launch_thread_ids,
        gpu_start_ns,
    )

    # Range id -1 (no enclosing range) indexes the trailing -1 / "".
    event_steps = np.array(step_ids + [-1], dtype=np.int32)[step_range_ids]
    event_scopes = np.array(scope_names + [""], dtype=object)[scope_range_ids]
    event_scope_paths = intervals.range_paths(
        scope_ranges, scope_range_ids, scope_names
    )
    labelled = events_with_launch.drop(
        "launch_thread_id", "launch_ns"
    ).with_columns(
        pl.Series("step", event_steps, dtype=pl.Int32),
        pl.Series("scope", event_scopes, dtype=pl.String),
        pl.Series("scope_path", event_scope_paths, dtype=pl.String),
    )
    steps = (
        pl.DataFrame(
            {
                "step": [row[0] for row in step_rows],
                "start_ns": [row[1] for row in step_rows],
                "end_ns": [row[2] for row in step_rows],
            },
            schema=empty_steps.schema,
        )
        # One row per range; ids may repeat across devices/loops.
        .unique(subset=["step", "start_ns", "end_ns"])
        .sort("start_ns")
    )
    return labelled, steps


def _gpu_metrics(conn: sqlite3.Connection, tables: set[str]) -> pl.DataFrame:
    """Returns the GPU metric samples of an export; zero values are kept.

    Reads the modern ``GPU_METRICS`` table joined with
    ``TARGET_INFO_GPU_METRICS``, the legacy ``GPU_METRICS`` table of
    dict-strings, or ``GENERIC_EVENTS``/``GENERIC_EVENT`` dict-strings.
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
        if "metricId" in metric_columns and "TARGET_INFO_GPU_METRICS" in tables:
            timestamp_column = (
                "timestamp" if "timestamp" in metric_columns else "rawTimestamp"
            )
            rows = conn.execute(
                f"SELECT t.metricName, g.{timestamp_column}, g.value "
                "FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS t "
                "ON g.metricId = t.metricId"
            ).fetchall()
            for metric_name, timestamp, value in rows:
                samples.append(
                    {
                        "metric_name": str(metric_name),
                        "timestamp_ns": int(timestamp or 0),
                        "value": float(value),
                        "rank": 0,
                    }
                )
        elif "value" in metric_columns or "data" in metric_columns:
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


def _summary_by_name(
    conn: sqlite3.Connection, tables: set[str], table: str, schema: pl.Schema
) -> pl.DataFrame:
    """Returns the total duration and call count per name of an API table.

    Args:
        conn: The open export.
        tables: The names of the export's tables.
        table: The API table to summarise, with ``nameId``, ``start`` and
            ``end`` columns.
        schema: The schema of the result.

    Returns:
        One row per API name (``"unknown"`` if not in ``StringIds``); empty
        if the table or ``StringIds`` is missing.
    """
    if table not in tables or "StringIds" not in tables:
        return pl.DataFrame(schema=schema)
    rows = conn.execute(
        "SELECT COALESCE(s.value, 'unknown'), SUM(r.end - r.start), COUNT(*) "
        f"FROM {table} r LEFT JOIN StringIds s ON r.nameId = s.id "
        "WHERE r.end > r.start GROUP BY 1"
    ).fetchall()
    return pl.DataFrame(
        [
            {
                "name": str(row[0]),
                "duration_ns": int(row[1]),
                "calls": int(row[2]),
            }
            for row in rows
        ],
        schema=schema,
    )


def _ingest_report(db_path: pathlib.Path, rank: int) -> dict[str, pl.DataFrame]:
    """Reads the tables of one SQLite export.

    A RuntimeWarning is issued when a report with GPU activities has no NVTX
    step markers; its whole trace then becomes step 0.

    Args:
        db_path: The SQLite export, opened read-only.
        rank: The rank written to the kernels, steps and gpu_metrics rows.

    Returns:
        The ``kernels``, ``steps``, ``runtime``, ``osrt`` and
        ``gpu_metrics`` frames, by table name.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _list_tables(conn)
        events = _gpu_events(conn, tables)
        if events.height:
            events = alignment.enrich_ops_df(events.sort("start_ns"), "name")
        launches = _launch_info(conn, tables)
        nvtx = _nvtx_ranges(conn, tables)
        labelled, steps = _label_events(events, launches, nvtx)
        runtime = _summary_by_name(
            conn, tables, "CUPTI_ACTIVITY_KIND_RUNTIME", RUNTIME_SCHEMA
        )
        osrt = _summary_by_name(conn, tables, "OSRT_API", OSRT_SCHEMA)
        gpu_metrics = _gpu_metrics(conn, tables).with_columns(
            pl.lit(rank, dtype=pl.Int32).alias("rank")
        )
    finally:
        conn.close()

    if labelled.height:
        kernels = conform(
            labelled.with_columns(
                (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"),
                pl.lit(rank).alias("rank"),
            ),
            KERNELS_SCHEMA,
        )
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
    return {
        "kernels": kernels,
        "steps": steps,
        "runtime": runtime,
        "osrt": osrt,
        "gpu_metrics": gpu_metrics,
    }


@registry.register_adapter("nsys", version="2.1")
class NsysAdapter:
    """Reads Nsight Systems reports and SQLite exports, one per rank.

    Attributes:
        format_name: The registered format name, ``"nsys"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "nsys"
    version = "2.1"

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
            The paths of the ``kernels``, ``steps``, ``runtime``, ``osrt``
            and ``gpu_metrics`` Parquet tables, by table name.

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
            "steps": concat_reports("steps", STEPS_SCHEMA).sort(
                ["rank", "start_ns"]
            ),
            "runtime": merge_summaries("runtime", RUNTIME_SCHEMA),
            "osrt": merge_summaries("osrt", OSRT_SCHEMA),
            "gpu_metrics": concat_reports("gpu_metrics", GPU_METRICS_SCHEMA),
        }
        table_paths: dict[str, pathlib.Path] = {}
        for table_name, table in merged_tables.items():
            table_path = out_dir / f"{table_name}.parquet"
            table.write_parquet(table_path, compression="zstd")
            table_paths[table_name] = table_path
        return table_paths
