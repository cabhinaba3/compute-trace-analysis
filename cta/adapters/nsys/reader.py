"""NVIDIA Nsight Systems (.nsys-rep and .sqlite) trace adapter."""

from __future__ import annotations

import ast
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from cta.core.alignment import enrich_ops_df
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
    }
)

STEPS_SCHEMA = pl.Schema(
    {"step": pl.Int32(), "start_ns": pl.Int64(), "end_ns": pl.Int64(), "duration_ns": pl.Int64()}
)

RUNTIME_SCHEMA = pl.Schema({"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()})
OSRT_SCHEMA = pl.Schema({"name": pl.String(), "duration_ns": pl.Int64(), "calls": pl.Int64()})
GPU_METRICS_SCHEMA = pl.Schema({"metric_name": pl.String(), "timestamp_ns": pl.Int64(), "value": pl.Float64()})

_STEP_RE = re.compile(r"(?:profilerstep|step|iter|iteration|inference)[#_\s:-]*(\d+)", re.IGNORECASE)
_MEMCPY_KIND_NAMES = {1: "Memcpy_H2D", 2: "Memcpy_D2H", 8: "Memcpy_D2D", 10: "Memcpy_P2P"}


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


def _assign_scopes_and_steps(
    k_starts: np.ndarray,
    nvtx_rows: list[tuple[str, int, int]],
) -> tuple[list[int], list[str], pl.DataFrame]:
    n = len(k_starts)
    if not nvtx_rows:
        return ([-1] * n, [""] * n, pl.DataFrame(schema=STEPS_SCHEMA))

    step_intervals: list[tuple[int, int, int]] = []
    scope_intervals: list[tuple[str, int, int]] = []
    for txt, s, e in nvtx_rows:
        if e <= s or not txt:
            continue
        m = _STEP_RE.search(txt)
        if m:
            step_intervals.append((int(m.group(1)), s, e))
        else:
            scope_intervals.append((txt, s, e))

    # If no explicit step# markers exist, treat ExecutionContext / top-level repeated ranges as steps
    if not step_intervals and scope_intervals:
        exec_ranges = [(i, s, e) for i, (t, s, e) in enumerate(scope_intervals) if "executioncontext" in t.lower() or "enqueue" in t.lower()]
        if exec_ranges:
            step_intervals = exec_ranges

    steps_out = [-1] * n
    if step_intervals:
        step_intervals.sort(key=lambda x: x[1])
        s_ids = np.array([x[0] for x in step_intervals], dtype=np.int32)
        s_beg = np.array([x[1] for x in step_intervals], dtype=np.int64)
        s_end = np.array([x[2] for x in step_intervals], dtype=np.int64)
        idx = np.searchsorted(s_beg, k_starts, side="right") - 1
        valid = (idx >= 0) & (k_starts <= s_end[np.maximum(0, idx)])
        for i in np.flatnonzero(valid):
            steps_out[int(i)] = int(s_ids[idx[i]])
        steps_df = pl.DataFrame(
            {
                "step": s_ids,
                "start_ns": s_beg,
                "end_ns": s_end,
                "duration_ns": s_end - s_beg,
            },
            schema=STEPS_SCHEMA,
        )
    else:
        steps_df = pl.DataFrame(schema=STEPS_SCHEMA)

    scopes_out = [""] * n
    if scope_intervals:
        scope_intervals.sort(key=lambda x: x[1])
        sc_names = [x[0] for x in scope_intervals]
        sc_beg = np.array([x[1] for x in scope_intervals], dtype=np.int64)
        sc_end = np.array([x[2] for x in scope_intervals], dtype=np.int64)
        idx = np.searchsorted(sc_beg, k_starts, side="right") - 1
        valid = (idx >= 0) & (k_starts <= sc_end[np.maximum(0, idx)])
        for i in np.flatnonzero(valid):
            scopes_out[int(i)] = sc_names[int(idx[i])]

    return steps_out, scopes_out, steps_df


@register_adapter("nsys", version="1.0")
class NsysAdapter:
    format_name = "nsys"
    version = "1.0"

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
        db_path = _resolve_sqlite(artifact.paths[0], out_dir)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            tbls = _list_tables(conn)
            has_str = "StringIds" in tbls

            # 1. NVTX ranges
            nvtx_rows: list[tuple[str, int, int]] = []
            if "NVTX_EVENTS" in tbls:
                ncols = _table_cols(conn, "NVTX_EVENTS")
                if "textId" in ncols and has_str:
                    sql = (
                        "SELECT COALESCE(n.text, s.value, ''), n.start, n.end "
                        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id "
                        "WHERE n.end IS NOT NULL AND n.end > n.start"
                        if "text" in ncols
                        else "SELECT COALESCE(s.value, ''), n.start, n.end "
                        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id "
                        "WHERE n.end IS NOT NULL AND n.end > n.start"
                    )
                elif "text" in ncols:
                    sql = "SELECT COALESCE(text, ''), start, end FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"
                else:
                    sql = ""
                if sql:
                    nvtx_rows = [(str(r[0]), int(r[1]), int(r[2])) for r in conn.execute(sql).fetchall()]

            # 2. Kernels + Memcpy + Memset
            raw_events: list[dict[str, object]] = []
            if "CUPTI_ACTIVITY_KIND_KERNEL" in tbls:
                kcols = _table_cols(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
                name_col = "demangledName" if "demangledName" in kcols else "shortName"
                stream_expr = "k.streamId" if "streamId" in kcols else "0"
                sql = (
                    f"SELECT COALESCE(s.value, CAST(k.{name_col} AS TEXT)), k.start, k.end, {stream_expr} "
                    f"FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    f"LEFT JOIN StringIds s ON k.{name_col} = s.id WHERE k.end > k.start"
                    if has_str
                    else f"SELECT CAST({name_col} AS TEXT), start, end, {stream_expr} FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end > start"
                )
                for nm, st, en, strm in conn.execute(sql).fetchall():
                    s_nm = str(nm)
                    cat = "comm" if "nccl" in s_nm.lower() else "compute"
                    raw_events.append(
                        {"name": s_nm, "category": cat, "start_ns": int(st), "end_ns": int(en), "bytes": 0, "stream": int(strm or 0)}
                    )

            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tbls:
                mcols = _table_cols(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
                stream_expr = "streamId" if "streamId" in mcols else "0"
                bytes_expr = "bytes" if "bytes" in mcols else "0"
                for ck, st, en, b, strm in conn.execute(
                    f"SELECT copyKind, start, end, {bytes_expr}, {stream_expr} FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end > start"
                ).fetchall():
                    raw_events.append(
                        {
                            "name": _MEMCPY_KIND_NAMES.get(int(ck or 0), "Memcpy_D2D"),
                            "category": "memcpy",
                            "start_ns": int(st),
                            "end_ns": int(en),
                            "bytes": int(b or 0),
                            "stream": int(strm or 0),
                        }
                    )

            if raw_events:
                ev_df = pl.DataFrame(raw_events).sort("start_ns")
                ev_df = enrich_ops_df(ev_df, "name")
                starts_np = ev_df["start_ns"].to_numpy()
                steps_col, scopes_col, steps_df = _assign_scopes_and_steps(starts_np, nvtx_rows)
                kernels_df = ev_df.with_columns(
                    pl.Series("step", steps_col, dtype=pl.Int32),
                    pl.Series("scope", scopes_col, dtype=pl.String),
                    (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"),
                ).select(list(KERNELS_SCHEMA.keys())).cast(KERNELS_SCHEMA)
            else:
                kernels_df = pl.DataFrame(schema=KERNELS_SCHEMA)
                _, _, steps_df = _assign_scopes_and_steps(np.empty(0, dtype=np.int64), nvtx_rows)

            if steps_df.is_empty() and kernels_df.height > 0:
                s0 = int(kernels_df["start_ns"].to_numpy().min())
                e0 = int(kernels_df["end_ns"].to_numpy().max())
                steps_df = pl.DataFrame(
                    [{"step": 0, "start_ns": s0, "end_ns": e0, "duration_ns": e0 - s0}],
                    schema=STEPS_SCHEMA,
                )

            # 3. CUDA Runtime summary
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in tbls and has_str:
                rt_rows = conn.execute(
                    "SELECT COALESCE(s.value, 'unknown'), SUM(r.end - r.start), COUNT(*) "
                    "FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON r.nameId = s.id "
                    "WHERE r.end > r.start GROUP BY 1"
                ).fetchall()
                runtime_df = pl.DataFrame(
                    [{"name": str(r[0]), "duration_ns": int(r[1]), "calls": int(r[2])} for r in rt_rows],
                    schema=RUNTIME_SCHEMA,
                )
            else:
                runtime_df = pl.DataFrame(schema=RUNTIME_SCHEMA)

            # 4. OSRT API summary
            if "OSRT_API" in tbls and has_str:
                os_rows = conn.execute(
                    "SELECT COALESCE(s.value, 'unknown'), SUM(o.end - o.start), COUNT(*) "
                    "FROM OSRT_API o LEFT JOIN StringIds s ON o.nameId = s.id "
                    "WHERE o.end > o.start GROUP BY 1"
                ).fetchall()
                osrt_df = pl.DataFrame(
                    [{"name": str(r[0]), "duration_ns": int(r[1]), "calls": int(r[2])} for r in os_rows],
                    schema=OSRT_SCHEMA,
                )
            else:
                osrt_df = pl.DataFrame(schema=OSRT_SCHEMA)

            # 5. GPU_METRICS / GENERIC_EVENT (supports both modern TARGET_INFO_GPU_METRICS join & legacy dict strings)
            gm_list: list[dict[str, object]] = []
            if "GPU_METRICS" in tbls:
                gm_cols = _table_cols(conn, "GPU_METRICS")
                if "metricId" in gm_cols and "TARGET_INFO_GPU_METRICS" in tbls:
                    ts_col = "timestamp" if "timestamp" in gm_cols else "rawTimestamp"
                    for mname, ts, val in conn.execute(
                        f"SELECT t.metricName, g.{ts_col}, g.value "
                        f"FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS t ON g.metricId = t.metricId"
                    ).fetchall():
                        gm_list.append({"metric_name": str(mname), "timestamp_ns": int(ts or 0), "value": float(val)})
                elif "value" in gm_cols or "data" in gm_cols:
                    vcol = "value" if "value" in gm_cols else "data"
                    ts_expr = "rawTimestamp" if "rawTimestamp" in gm_cols else "0"
                    for idx_r, (raw_v, ts) in enumerate(conn.execute(f"SELECT {vcol}, {ts_expr} FROM GPU_METRICS").fetchall()):
                        try:
                            parsed = ast.literal_eval(str(raw_v))
                            if isinstance(parsed, dict):
                                for mk, mv in parsed.items():
                                    fv = float(mv)
                                    if fv != 0.0:
                                        gm_list.append({"metric_name": str(mk), "timestamp_ns": int(ts or idx_r), "value": fv})
                        except (ValueError, SyntaxError):
                            continue
            elif "GENERIC_EVENTS" in tbls or "GENERIC_EVENT" in tbls:
                gtbl = "GENERIC_EVENTS" if "GENERIC_EVENTS" in tbls else "GENERIC_EVENT"
                gcols = _table_cols(conn, gtbl)
                vcol = "data" if "data" in gcols else "value"
                for idx_r, (raw_v,) in enumerate(conn.execute(f"SELECT {vcol} FROM {gtbl}").fetchall()):
                    try:
                        parsed = ast.literal_eval(str(raw_v))
                        if isinstance(parsed, dict):
                            for mk, mv in parsed.items():
                                fv = float(mv)
                                if fv != 0.0:
                                    gm_list.append({"metric_name": str(mk), "timestamp_ns": idx_r, "value": fv})
                    except (ValueError, SyntaxError):
                        continue

            gpu_metrics_df = pl.DataFrame(gm_list, schema=GPU_METRICS_SCHEMA) if gm_list else pl.DataFrame(schema=GPU_METRICS_SCHEMA)
        finally:
            conn.close()

        out: dict[str, Path] = {}
        for name, df in (
            ("kernels", kernels_df),
            ("steps", steps_df),
            ("runtime", runtime_df),
            ("osrt", osrt_df),
            ("gpu_metrics", gpu_metrics_df),
        ):
            p = out_dir / f"{name}.parquet"
            df.write_parquet(p, compression="zstd")
            out[name] = p
        return out
