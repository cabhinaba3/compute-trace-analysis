"""PyTorch Profiler / Kineto Chrome Trace (.json and .json.gz) adapter."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cta.adapters.nsys.reader import KERNELS_SCHEMA, RUNTIME_SCHEMA, STEPS_SCHEMA, conform
from cta.core.alignment import enrich_ops_df
from cta.core.bundle import infer_rank
from cta.core.intervals import assign_ranges, build_thread_ranges, range_paths
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

_STEP_RE = re.compile(r"profilerstep#\s*(\d+)", re.IGNORECASE)


def _open_text_head(path: Path, nbytes: int = 8192) -> str:
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as gz:
                return gz.read(nbytes).decode("utf-8", errors="ignore")
        with path.open("rb") as fh:
            return fh.read(nbytes).decode("utf-8", errors="ignore")
    except OSError:
        return ""


@register_adapter("kineto", version="2.2")
class KinetoAdapter:
    format_name = "kineto"
    version = "2.2"

    def detect(self, path: Path) -> bool:
        name_low = path.name.lower()
        if not (name_low.endswith(".json") or name_low.endswith(".json.gz")):
            return False
        head = _open_text_head(path)
        return '"traceEvents"' in head
    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        """Ingest every shard of the artifact (one trace per rank). Each shard keeps its own clock and rank."""
        kernel_frames: list[pl.DataFrame] = []
        step_frames: list[pl.DataFrame] = []
        rt_agg: dict[str, list[int]] = {}

        for idx, path in enumerate(sorted(artifact.paths)):
            doc = _load(path)
            rank = _trace_rank(doc, path, idx)
            kdf, sdf = _ingest_shard(doc, rank, rt_agg)
            if kdf.height:
                kernel_frames.append(kdf)
            if sdf.height:
                step_frames.append(sdf)

        kernels_df = pl.concat(kernel_frames).sort("rank", "start_ns") if kernel_frames else pl.DataFrame(schema=KERNELS_SCHEMA)
        steps_df = pl.concat(step_frames).sort("rank", "start_ns") if step_frames else pl.DataFrame(schema=STEPS_SCHEMA)
        runtime_df = (
            pl.DataFrame([{"name": k, "duration_ns": v[0], "calls": v[1]} for k, v in rt_agg.items()], schema=RUNTIME_SCHEMA)
            if rt_agg
            else pl.DataFrame(schema=RUNTIME_SCHEMA)
        )

        out: dict[str, Path] = {}
        for tname, df in (("kernels", kernels_df), ("steps", steps_df), ("runtime", runtime_df)):
            p = out_dir / f"{tname}.parquet"
            df.write_parquet(p, compression="zstd")
            out[tname] = p
        return out


def _load(path: Path) -> Any:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as gz:
            return json.load(gz)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _trace_rank(doc: Any, path: Path, index: int) -> int:
    """Rank of a shard: the trace's own ``distributedInfo.rank``, else a rank in the file name, else its index."""
    info = doc.get("distributedInfo") if isinstance(doc, dict) else None
    if isinstance(info, dict) and isinstance(info.get("rank"), int):
        return int(info["rank"])
    return infer_rank(path, index)


def _ingest_shard(doc: Any, rank: int, rt_agg: dict[str, list[int]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Kernels and steps of one rank's trace (single clock), plus its CUDA runtime totals accumulated into rt_agg."""
    events = doc.get("traceEvents", []) if isinstance(doc, dict) else doc
    gpu_rows: list[dict[str, object]] = []
    step_rows: list[tuple[int, int, int]] = []
    op_names: list[str] = []
    ops_by_thread: dict[int, list[tuple[int, int, int]]] = {}  # thread id -> (start, end, index into op_names)
    thread_ids: dict[tuple[object, object], int] = {}
    launches: dict[int, tuple[int, int]] = {}  # correlation id -> (launch time ns, thread id)

    def tid_of(ev: dict[str, Any]) -> int:
        return thread_ids.setdefault((ev.get("pid"), ev.get("tid")), len(thread_ids))

    # Pass 1: collect ProfilerStep intervals, CPU ops per thread, and the thread/time of every CUDA launch
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        cat = str(ev.get("cat", "")).lower()
        name = str(ev.get("name", ""))
        # Kineto timestamps & durations are in microseconds (can be float with fractional ns)
        start_ns = int(round(float(ev.get("ts", 0.0)) * 1000.0))
        dur_ns = int(round(float(ev.get("dur", 0.0)) * 1000.0))
        if dur_ns <= 0:
            continue
        end_ns = start_ns + dur_ns

        m_step = _STEP_RE.search(name)
        if m_step:
            # PyTorch also emits each step as a `gpu_user_annotation` (the GPU-side projection of the same
            # step); counting it would add a second, differently-timed step with the same id.
            if cat != "gpu_user_annotation":
                step_rows.append((int(m_step.group(1)), start_ns, end_ns))
            continue

        if cat in ("cpu_op", "operator", "user_annotation"):
            ops_by_thread.setdefault(tid_of(ev), []).append((start_ns, end_ns, len(op_names)))
            op_names.append(name)
        elif cat in ("cuda_runtime", "cuda_driver", "runtime"):
            prev = rt_agg.setdefault(name, [0, 0])
            prev[0] += dur_ns
            prev[1] += 1
            args = ev.get("args")
            if isinstance(args, dict) and "correlation" in args:
                launches[int(args["correlation"])] = (start_ns, tid_of(ev))

    # Pass 2: collect GPU kernels & memcpy/memset; remember when and from which thread each was launched
    launch_t: list[float] = []
    launch_tid: list[int] = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        cat = str(ev.get("cat", "")).lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset", "memcpy", "memset"):
            continue
        name = str(ev.get("name", ""))
        start_ns = int(round(float(ev.get("ts", 0.0)) * 1000.0))
        dur_ns = int(round(float(ev.get("dur", 0.0)) * 1000.0))
        if dur_ns <= 0:
            continue
        end_ns = start_ns + dur_ns
        raw_args = ev.get("args")
        k_args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        corr = int(k_args["correlation"]) if "correlation" in k_args else -1
        stream = int(k_args.get("stream", ev.get("tid", 0)) or 0)
        nbytes = int(k_args.get("bytes", 0) or 0)
        device = int(k_args["device"]) if isinstance(k_args.get("device"), int) else int(ev.get("pid", 0) or 0)

        if "memcpy" in cat or "memset" in cat:
            ev_cat = "memcpy"
        elif "nccl" in name.lower():
            ev_cat = "comm"
        else:
            ev_cat = "compute"

        launched = launches.get(corr)
        launch_t.append(float(launched[0]) if launched else float("nan"))
        launch_tid.append(launched[1] if launched else -1)
        gpu_rows.append(
            {
                "name": name, "scope": "", "category": ev_cat, "start_ns": start_ns, "end_ns": end_ns,
                "duration_ns": dur_ns, "bytes": nbytes, "stream": stream, "device": device, "rank": rank,
            }
        )  # fmt: skip

    if not gpu_rows:
        return pl.DataFrame(schema=KERNELS_SCHEMA), pl.DataFrame(schema=STEPS_SCHEMA)

    order = np.argsort([r["start_ns"] for r in gpu_rows], kind="stable")  # keep launch info aligned with sorted rows
    kdf = enrich_ops_df(pl.DataFrame(gpu_rows)[order.tolist()], "name")
    lt = np.asarray(launch_t, dtype=np.float64)[order]
    ltid = np.asarray(launch_tid, dtype=np.int64)[order]
    gpu_t = kdf["start_ns"].to_numpy()

    # Scope = the innermost CPU op that encloses the launch *on the launching thread* (a finished sibling that
    # started later does not hide the enclosing op). Kernels without a known launch get no scope.
    if op_names:
        threads = {t: build_thread_ranges(list(rows)) for t, rows in ops_by_thread.items()}
        gstart = np.zeros(len(op_names), dtype=np.int64)
        for rows in ops_by_thread.values():
            for st, _, g in rows:
                gstart[g] = st
        gid = assign_ranges(threads, gstart, lt, ltid, gpu_t)
        names = np.array([*op_names, ""], dtype=object)
        scope = np.where(ltid >= 0, names[gid], "")
        paths = np.where(ltid >= 0, range_paths(threads, gid, op_names), "")
        kdf = kdf.with_columns(pl.Series("scope", scope, dtype=pl.String), pl.Series("scope_path", paths, dtype=pl.String))

    # Launches are asynchronous: a kernel belongs to the step whose CPU range issued it, not the one it runs in.
    launch = np.where(np.isnan(lt), gpu_t, lt).astype(np.int64)
    if step_rows:
        step_rows.sort(key=lambda x: x[1])
        s_ids = np.array([x[0] for x in step_rows], dtype=np.int32)
        s_beg = np.array([x[1] for x in step_rows], dtype=np.int64)
        s_end = np.array([x[2] for x in step_rows], dtype=np.int64)
        s_idx = np.searchsorted(s_beg, launch, side="right") - 1
        valid = (s_idx >= 0) & (launch <= s_end[np.maximum(0, s_idx)])
        step_col = np.where(valid, s_ids[np.maximum(0, s_idx)], -1).astype(np.int32)
        steps = {"step": s_ids, "start_ns": s_beg, "end_ns": s_end, "duration_ns": s_end - s_beg}
    else:
        step_col = np.zeros(kdf.height, dtype=np.int32)
        s0, e0 = int(kdf["start_ns"].to_numpy().min()), int(kdf["end_ns"].to_numpy().max())
        steps = {"step": [0], "start_ns": [s0], "end_ns": [e0], "duration_ns": [e0 - s0]}
    steps_df = conform(pl.DataFrame(steps).with_columns(pl.lit(rank).alias("rank")), STEPS_SCHEMA)
    return conform(kdf.with_columns(pl.Series("step", step_col, dtype=pl.Int32)), KERNELS_SCHEMA), steps_df
