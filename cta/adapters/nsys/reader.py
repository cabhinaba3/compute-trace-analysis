"""NVIDIA Nsight Systems (.nsys-rep and .sqlite) trace adapter."""

from __future__ import annotations

import ast
import os
import re
import shutil
import sqlite3
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from cta.core.alignment import enrich_ops_df
from cta.core.bundle import infer_rank
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

KERNELS_SCHEMA = pl.Schema(
    {
        "step": pl.Int32(),
        "scope": pl.String(),
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
        "rank": pl.Int32(),  # per-report/process index; 0 for single-report sources
        "device": pl.Int32(),  # CUDA device id within the report; 0 for single-GPU sources
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

_SCHEMA_DEFAULTS: dict[str, object] = {"rank": 0, "device": 0}


def conform(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Select/cast ``df`` to ``schema``, filling columns a source does not provide (rank, device) with 0."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(_SCHEMA_DEFAULTS[c]).alias(c) for c in missing])
    return df.select(list(schema.keys())).cast(schema)


RUNTIME_SCHEMA = pl.Schema({"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()})
OSRT_SCHEMA = pl.Schema({"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()})
GPU_METRICS_SCHEMA = pl.Schema(
    {"metric_name": pl.String(), "timestamp_ns": pl.Int64(), "value": pl.Float64(), "rank": pl.Int32()}
)

_STEP_RE = re.compile(r"(?:profilerstep|step|iter|iteration|inference)[#_\s:-]*(\d+)", re.IGNORECASE)
# CUPTI memcpy kinds: 1 HtoD, 2 DtoH, 8 DtoD, 9 HtoH, 10 PtoP (others are rare array/unified-memory copies)
_MEMCPY_KIND_NAMES = {1: "Memcpy_H2D", 2: "Memcpy_D2H", 8: "Memcpy_D2D", 9: "Memcpy_H2H", 10: "Memcpy_P2P"}
_NVTX_RANGE_EVENT_TYPES = (59, 60)  # NvtxPushPopRange, NvtxStartEndRange


def _resolve_sqlite(path: Path, out_dir: Path) -> Path:
    if path.suffix.lower() == ".sqlite":
        return path
    sibling = path.with_suffix(".sqlite")
    if sibling.exists() and sibling.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        return sibling
    nsys_bin = shutil.which("nsys") or os.environ.get("NSYS_PATH") or os.environ.get("NSYS")
    dst_sqlite = out_dir / f"{path.stem}.sqlite"
    if nsys_bin:
        subprocess.run(
            [nsys_bin, "export", "-t", "sqlite", "--force-overwrite=true", "-o", str(dst_sqlite), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return dst_sqlite
    if sibling.exists():
        return sibling
    raise RuntimeError(
        f"Cannot export '{path.name}': 'nsys' CLI not found on PATH and no exported '{sibling.name}' exists."
    )


def _list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(r[0]) for r in rows}


def _table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


@dataclass(slots=True)
class _ThreadRanges:
    """NVTX ranges of one thread, sorted by start, with a parent link to the enclosing range."""

    starts: np.ndarray
    ends: np.ndarray
    gids: np.ndarray
    parent: np.ndarray


def _build_thread(rows: list[tuple[int, int, int]]) -> _ThreadRanges:
    rows.sort(key=lambda r: (r[0], -r[1]))
    starts = np.array([r[0] for r in rows], dtype=np.int64)
    ends = np.array([r[1] for r in rows], dtype=np.int64)
    gids = np.array([r[2] for r in rows], dtype=np.int64)
    parent = np.full(len(rows), -1, dtype=np.int64)
    stack: list[int] = []
    for i in range(len(rows)):
        while stack and ends[stack[-1]] <= starts[i]:
            stack.pop()
        parent[i] = stack[-1] if stack else -1
        stack.append(i)
    return _ThreadRanges(starts, ends, gids, parent)


def _innermost(th: _ThreadRanges, q: np.ndarray) -> np.ndarray:
    """Global id of the innermost range of this thread containing each query time (-1 if none).

    ``searchsorted`` finds the latest-starting range; if it has already ended, walk up to the enclosing range
    (this is what a plain "latest start" lookup gets wrong for sibling-then-outer cases).
    """
    out = np.full(len(q), -1, dtype=np.int64)
    idx = np.searchsorted(th.starts, q, side="right") - 1
    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    inside = ok & (q <= th.ends[safe])
    out[inside] = th.gids[idx[inside]]
    for i in np.flatnonzero(ok & ~inside):
        k = int(th.parent[idx[i]])
        while k >= 0 and q[i] > th.ends[k]:
            k = int(th.parent[k])
        if k >= 0:
            out[i] = th.gids[k]
    return out


def _assign_ranges(
    threads: dict[int, _ThreadRanges],
    gstart: np.ndarray,
    launch_t: np.ndarray,
    launch_tid: np.ndarray,
    gpu_t: np.ndarray,
) -> np.ndarray:
    """Range id per GPU activity: the innermost range on the *launching* thread at launch time.

    Launches are asynchronous, so the GPU start time says nothing about which CPU range issued the work. When
    the launch is unknown (no runtime row) or its thread has no enclosing range, fall back to the latest-starting
    containing range on any thread (queried at launch time if known, else at GPU start time).
    """
    n = len(gpu_t)
    out = np.full(n, -1, dtype=np.int64)
    for tid, th in threads.items():
        sel = launch_tid == tid
        if sel.any():
            out[sel] = _innermost(th, launch_t[sel])
    todo = out < 0
    if todo.any() and threads:
        q = np.where(np.isnan(launch_t[todo]), gpu_t[todo], launch_t[todo]).astype(np.int64)
        best = np.full(len(q), -1, dtype=np.int64)
        best_start = np.full(len(q), -1, dtype=np.int64)
        for th in threads.values():
            cand = _innermost(th, q)
            hit = cand >= 0
            starts = np.where(hit, gstart[np.where(hit, cand, 0)], -1)
            better = hit & (starts > best_start)
            best[better] = cand[better]
            best_start[better] = starts[better]
        out[np.flatnonzero(todo)] = best
    return out


def _nvtx_ranges(conn: sqlite3.Connection, tbls: set[str]) -> list[tuple[str, int, int, int]]:
    """(text, start, end, globalTid) for every closed NVTX range."""
    if "NVTX_EVENTS" not in tbls:
        return []
    ncols = _table_cols(conn, "NVTX_EVENTS")
    tid = "n.globalTid" if "globalTid" in ncols else "0"
    types = f" AND n.eventType IN {_NVTX_RANGE_EVENT_TYPES}" if "eventType" in ncols else ""
    where = f"WHERE n.end IS NOT NULL AND n.end > n.start{types}"
    if "textId" in ncols and "StringIds" in tbls:
        txt = "COALESCE(n.text, s.value, '')" if "text" in ncols else "COALESCE(s.value, '')"
        sql = f"SELECT {txt}, n.start, n.end, {tid} FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id {where}"
    elif "text" in ncols:
        sql = f"SELECT COALESCE(n.text, ''), n.start, n.end, {tid} FROM NVTX_EVENTS n {where}"
    else:
        return []
    return [(str(r[0]), int(r[1]), int(r[2]), int(r[3] or 0)) for r in conn.execute(sql).fetchall()]


def _gpu_events(conn: sqlite3.Connection, tbls: set[str]) -> pl.DataFrame:
    """Kernels, memcpys and memsets with device, stream and the correlation id of their launching API call."""
    has_str = "StringIds" in tbls
    rows: list[dict[str, object]] = []

    def col(cols: set[str], name: str, default: str, alias: str = "t") -> str:
        return f"{alias}.{name}" if name in cols else default

    if "CUPTI_ACTIVITY_KIND_KERNEL" in tbls:
        kc = _table_cols(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_col = "demangledName" if "demangledName" in kc else "shortName"
        nm = f"COALESCE(s.value, CAST(t.{name_col} AS TEXT))" if has_str else f"CAST(t.{name_col} AS TEXT)"
        join = f"LEFT JOIN StringIds s ON t.{name_col} = s.id" if has_str else ""
        sql = (
            f"SELECT {nm}, t.start, t.end, {col(kc, 'deviceId', '0')}, {col(kc, 'streamId', '0')}, "
            f"{col(kc, 'correlationId', '-1')} FROM CUPTI_ACTIVITY_KIND_KERNEL t {join} WHERE t.end > t.start"
        )
        for nm_, st, en, dev, strm, corr in conn.execute(sql).fetchall():
            cat = "comm" if "nccl" in str(nm_).lower() else "compute"
            rows.append(
                {"name": str(nm_), "category": cat, "start_ns": st, "end_ns": en, "bytes": 0,
                 "stream": strm, "device": dev, "correlation_id": corr}
            )

    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tbls:
        mc = _table_cols(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
        sql = (
            f"SELECT {col(mc, 'copyKind', '0')}, t.start, t.end, {col(mc, 'bytes', '0')}, {col(mc, 'deviceId', '0')}, "
            f"{col(mc, 'streamId', '0')}, {col(mc, 'correlationId', '-1')} "
            f"FROM CUPTI_ACTIVITY_KIND_MEMCPY t WHERE t.end > t.start"
        )
        for ck, st, en, nbytes, dev, strm, corr in conn.execute(sql).fetchall():
            rows.append(
                {"name": _MEMCPY_KIND_NAMES.get(int(ck or 0), "Memcpy_Other"), "category": "memcpy", "start_ns": st,
                 "end_ns": en, "bytes": nbytes, "stream": strm, "device": dev, "correlation_id": corr}
            )

    if "CUPTI_ACTIVITY_KIND_MEMSET" in tbls:
        sc = _table_cols(conn, "CUPTI_ACTIVITY_KIND_MEMSET")
        sql = (
            f"SELECT t.start, t.end, {col(sc, 'bytes', '0')}, {col(sc, 'deviceId', '0')}, {col(sc, 'streamId', '0')}, "
            f"{col(sc, 'correlationId', '-1')} FROM CUPTI_ACTIVITY_KIND_MEMSET t WHERE t.end > t.start"
        )
        for st, en, nbytes, dev, strm, corr in conn.execute(sql).fetchall():
            rows.append(
                {"name": "Memset", "category": "memcpy", "start_ns": st, "end_ns": en, "bytes": nbytes,
                 "stream": strm, "device": dev, "correlation_id": corr}
            )

    schema = {
        "name": pl.String, "category": pl.String, "start_ns": pl.Int64, "end_ns": pl.Int64, "bytes": pl.Int64,
        "stream": pl.Int32, "device": pl.Int32, "correlation_id": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def _launch_info(conn: sqlite3.Connection, tbls: set[str]) -> pl.DataFrame:
    """correlation id -> (launching thread, launch time) from the CUDA runtime table."""
    empty = pl.DataFrame(schema={"correlation_id": pl.Int64, "launch_tid": pl.Int64, "launch_ns": pl.Int64})
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tbls:
        return empty
    rc = _table_cols(conn, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if "correlationId" not in rc or "globalTid" not in rc:
        return empty
    rows = conn.execute("SELECT correlationId, globalTid, start FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    df = pl.DataFrame(rows, schema=empty.schema, orient="row")
    return df.unique(subset="correlation_id", keep="first")


def _label_events(
    events: pl.DataFrame, launches: pl.DataFrame, nvtx: list[tuple[str, int, int, int]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach ``step``/``scope`` to every GPU activity and return the per-step table for this report."""
    n = events.height
    no_steps = pl.DataFrame(schema={"step": pl.Int32, "start_ns": pl.Int64, "end_ns": pl.Int64})
    if n == 0 or not nvtx:
        return events.with_columns(pl.lit(-1, dtype=pl.Int32).alias("step"), pl.lit("").alias("scope")), no_steps

    named = [(t, s, e, tid) for t, s, e, tid in nvtx if t]
    is_step = [bool(_STEP_RE.search(t)) for t, *_ in named]
    # Without explicit step markers, repeated ExecutionContext/enqueue ranges (TensorRT) delimit the steps.
    exec_fallback = not any(is_step)
    if exec_fallback:
        is_step = [("executioncontext" in t.lower() or "enqueue" in t.lower()) for t, *_ in named]

    step_rows: list[tuple[int, int, int, int, int]] = []  # (tid, start, end, gid, step id)
    scope_by_tid: dict[int, list[tuple[int, int, int]]] = {}
    step_by_tid: dict[int, list[tuple[int, int, int]]] = {}
    scope_names: list[str] = []
    step_ids: list[int] = []
    for (t, s, e, tid), st in zip(named, is_step, strict=True):
        if st:
            m = _STEP_RE.search(t)
            sid = int(m.group(1)) if m else len(step_ids)
            step_by_tid.setdefault(tid, []).append((s, e, len(step_ids)))
            step_ids.append(sid)
            step_rows.append((tid, s, e, len(step_ids) - 1, sid))
        else:
            scope_by_tid.setdefault(tid, []).append((s, e, len(scope_names)))
            scope_names.append(t)

    ev = events.join(launches, on="correlation_id", how="left")
    launch_t = ev["launch_ns"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    launch_tid = ev["launch_tid"].fill_null(-1).to_numpy()
    gpu_t = ev["start_ns"].to_numpy()
    launch_q = np.where(np.isnan(launch_t), gpu_t, launch_t).astype(np.int64)

    def gstarts(by_tid: dict[int, list[tuple[int, int, int]]], count: int) -> np.ndarray:
        arr = np.zeros(count, dtype=np.int64)
        for rows in by_tid.values():
            for s, _, g in rows:
                arr[g] = s
        return arr

    step_threads = {tid: _build_thread(list(r)) for tid, r in step_by_tid.items()}
    scope_threads = {tid: _build_thread(list(r)) for tid, r in scope_by_tid.items()}
    step_gid = _assign_ranges(step_threads, gstarts(step_by_tid, len(step_ids)), launch_q.astype(np.float64), launch_tid, gpu_t)
    scope_gid = _assign_ranges(scope_threads, gstarts(scope_by_tid, len(scope_names)), launch_q.astype(np.float64), launch_tid, gpu_t)

    step_arr = np.array(step_ids + [-1], dtype=np.int32)[step_gid]  # gid -1 indexes the trailing -1
    scope_arr = np.array(scope_names + [""], dtype=object)[scope_gid]
    labelled = ev.drop("launch_tid", "launch_ns").with_columns(
        pl.Series("step", step_arr, dtype=pl.Int32), pl.Series("scope", scope_arr, dtype=pl.String)
    )
    steps = (
        pl.DataFrame(
            {"step": [r[4] for r in step_rows], "start_ns": [r[1] for r in step_rows], "end_ns": [r[2] for r in step_rows]},
            schema=no_steps.schema,
        )
        .unique(subset=["step", "start_ns", "end_ns"])  # one row per range; ids may repeat across devices/loops
        .sort("start_ns")
    )
    return labelled, steps


def _gpu_metrics(conn: sqlite3.Connection, tbls: set[str]) -> pl.DataFrame:
    """GPU_METRICS samples (modern TARGET_INFO_GPU_METRICS join and legacy dict-string schemas). Zeros are kept."""
    out: list[dict[str, object]] = []

    def parse(raw: object, ts: int) -> None:
        try:
            parsed = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return
        if isinstance(parsed, dict):
            for mk, mv in parsed.items():
                out.append({"metric_name": str(mk), "timestamp_ns": ts, "value": float(mv), "rank": 0})

    if "GPU_METRICS" in tbls:
        gm_cols = _table_cols(conn, "GPU_METRICS")
        if "metricId" in gm_cols and "TARGET_INFO_GPU_METRICS" in tbls:
            ts_col = "timestamp" if "timestamp" in gm_cols else "rawTimestamp"
            for mname, ts, val in conn.execute(
                f"SELECT t.metricName, g.{ts_col}, g.value "
                f"FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS t ON g.metricId = t.metricId"
            ).fetchall():
                out.append({"metric_name": str(mname), "timestamp_ns": int(ts or 0), "value": float(val), "rank": 0})
        elif "value" in gm_cols or "data" in gm_cols:
            vcol = "value" if "value" in gm_cols else "data"
            ts_expr = "rawTimestamp" if "rawTimestamp" in gm_cols else "0"
            for idx_r, (raw_v, ts) in enumerate(conn.execute(f"SELECT {vcol}, {ts_expr} FROM GPU_METRICS").fetchall()):
                parse(raw_v, int(ts or idx_r))
    elif "GENERIC_EVENTS" in tbls or "GENERIC_EVENT" in tbls:
        gtbl = "GENERIC_EVENTS" if "GENERIC_EVENTS" in tbls else "GENERIC_EVENT"
        vcol = "data" if "data" in _table_cols(conn, gtbl) else "value"
        for idx_r, (raw_v,) in enumerate(conn.execute(f"SELECT {vcol} FROM {gtbl}").fetchall()):
            parse(raw_v, idx_r)
    return pl.DataFrame(out, schema=GPU_METRICS_SCHEMA) if out else pl.DataFrame(schema=GPU_METRICS_SCHEMA)


def _summary_by_name(conn: sqlite3.Connection, tbls: set[str], table: str, schema: pl.Schema) -> pl.DataFrame:
    if table not in tbls or "StringIds" not in tbls:
        return pl.DataFrame(schema=schema)
    rows = conn.execute(
        f"SELECT COALESCE(s.value, 'unknown'), SUM(r.end - r.start), COUNT(*) "
        f"FROM {table} r LEFT JOIN StringIds s ON r.nameId = s.id WHERE r.end > r.start GROUP BY 1"
    ).fetchall()
    return pl.DataFrame(
        [{"name": str(r[0]), "duration_ns": int(r[1]), "calls": int(r[2])} for r in rows], schema=schema
    )


def _ingest_report(db_path: Path, rank: int) -> dict[str, pl.DataFrame]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tbls = _list_tables(conn)
        events = _gpu_events(conn, tbls)
        if events.height:
            events = enrich_ops_df(events.sort("start_ns"), "name")
        labelled, steps = _label_events(events, _launch_info(conn, tbls), _nvtx_ranges(conn, tbls))
        runtime = _summary_by_name(conn, tbls, "CUPTI_ACTIVITY_KIND_RUNTIME", RUNTIME_SCHEMA)
        osrt = _summary_by_name(conn, tbls, "OSRT_API", OSRT_SCHEMA)
        metrics = _gpu_metrics(conn, tbls).with_columns(pl.lit(rank, dtype=pl.Int32).alias("rank"))
    finally:
        conn.close()

    if labelled.height:
        kernels = conform(
            labelled.with_columns(
                (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"), pl.lit(rank).alias("rank")
            ),
            KERNELS_SCHEMA,
        )
    else:
        kernels = pl.DataFrame(schema=KERNELS_SCHEMA)

    if steps.is_empty() and kernels.height:
        warnings.warn(
            f"{db_path.name}: no NVTX step markers (e.g. 'ProfilerStep#N'); the whole trace is treated as one step, "
            "so --warmup-steps/--max-steps have no effect.",
            RuntimeWarning,
            stacklevel=2,
        )
        steps = pl.DataFrame(
            {"step": [0], "start_ns": [int(kernels["start_ns"].to_numpy().min())], "end_ns": [int(kernels["end_ns"].to_numpy().max())]},
            schema={"step": pl.Int32, "start_ns": pl.Int64, "end_ns": pl.Int64},
        )
    steps = conform(steps.with_columns((pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"), pl.lit(rank).alias("rank")), STEPS_SCHEMA)
    return {"kernels": kernels, "steps": steps, "runtime": runtime, "osrt": osrt, "gpu_metrics": metrics}


@register_adapter("nsys", version="2.0")
class NsysAdapter:
    format_name = "nsys"
    version = "2.0"

    def detect(self, path: Path) -> bool:
        suf = path.suffix.lower()
        if suf == ".nsys-rep":
            return True
        if suf == ".sqlite":
            try:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                    tbls = _list_tables(conn)
                return bool(
                    tbls
                    & {
                        "StringIds",
                        "CUPTI_ACTIVITY_KIND_KERNEL",
                        "CUPTI_ACTIVITY_KIND_RUNTIME",
                        "NVTX_EVENTS",
                        "GPU_METRICS",
                        "GENERIC_EVENT",
                    }
                )
            except sqlite3.Error:
                return False
        return False

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        """Ingest every report of the artifact (one per rank); rows carry ``rank`` and ``device``."""
        per_report: list[dict[str, pl.DataFrame]] = []
        for idx, path in enumerate(sorted(artifact.paths)):
            per_report.append(_ingest_report(_resolve_sqlite(path, out_dir), infer_rank(path, idx)))

        def cat(key: str, schema: pl.Schema) -> pl.DataFrame:
            frames = [r[key] for r in per_report if r[key].height]
            return pl.concat(frames) if frames else pl.DataFrame(schema=schema)

        def merge_summary(key: str, schema: pl.Schema) -> pl.DataFrame:
            df = cat(key, schema)
            if df.is_empty():
                return df
            return df.group_by("name").agg(pl.col("duration_ns").sum(), pl.col("calls").sum()).cast(schema)

        merged = {
            "kernels": cat("kernels", KERNELS_SCHEMA).sort(["rank", "start_ns"]),
            "steps": cat("steps", STEPS_SCHEMA).sort(["rank", "start_ns"]),
            "runtime": merge_summary("runtime", RUNTIME_SCHEMA),
            "osrt": merge_summary("osrt", OSRT_SCHEMA),
            "gpu_metrics": cat("gpu_metrics", GPU_METRICS_SCHEMA),
        }
        out: dict[str, Path] = {}
        for name, df in merged.items():
            p = out_dir / f"{name}.parquet"
            df.write_parquet(p, compression="zstd")
            out[name] = p
        return out
