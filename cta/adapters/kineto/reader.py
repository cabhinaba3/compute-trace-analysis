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


@register_adapter("kineto", version="1.0")
class KinetoAdapter:
    format_name = "kineto"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        name_low = path.name.lower()
        if not (name_low.endswith(".json") or name_low.endswith(".json.gz")):
            return False
        head = _open_text_head(path)
        return '"traceEvents"' in head

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        # Aggregate across paths (supports single or multi-rank Kineto trace shards)
        gpu_rows: list[dict[str, object]] = []
        rt_agg: dict[str, list[int]] = {}
        step_rows: list[tuple[int, int, int]] = []

        for path in sorted(artifact.paths):
            if path.suffix.lower() == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as gz:
                    doc = json.load(gz)
            else:
                with path.open("r", encoding="utf-8") as fh:
                    doc = json.load(fh)

            events = doc.get("traceEvents", []) if isinstance(doc, dict) else doc
            cpu_ops: list[tuple[int, int, str]] = []
            corr_to_cpu_ts: dict[int, int] = {}

            # Pass 1: collect ProfilerStep intervals, CPU ops, and CUDA runtime launch correlations
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
                    cpu_ops.append((start_ns, end_ns, name))
                elif cat in ("cuda_runtime", "cuda_driver", "runtime"):
                    prev = rt_agg.setdefault(name, [0, 0])
                    prev[0] += dur_ns
                    prev[1] += 1
                    args = ev.get("args")
                    if isinstance(args, dict) and "correlation" in args:
                        corr_to_cpu_ts[int(args["correlation"])] = start_ns

            cpu_ops.sort(key=lambda x: x[0])
            cpu_starts = np.array([x[0] for x in cpu_ops], dtype=np.int64) if cpu_ops else np.empty(0, dtype=np.int64)
            cpu_ends = np.array([x[1] for x in cpu_ops], dtype=np.int64) if cpu_ops else np.empty(0, dtype=np.int64)

            # Pass 2: collect GPU kernels & memcpy/memset and link back to originating CPU op via correlation ID
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

                scope = ""
                if corr in corr_to_cpu_ts and len(cpu_ops) > 0:
                    launch_ts = corr_to_cpu_ts[corr]
                    idx = int(np.searchsorted(cpu_starts, launch_ts, side="right")) - 1
                    if idx >= 0 and launch_ts <= cpu_ends[idx]:
                        scope = cpu_ops[idx][2]

                if "memcpy" in cat or "memset" in cat:
                    ev_cat = "memcpy"
                elif "nccl" in name.lower():
                    ev_cat = "comm"
                else:
                    ev_cat = "compute"

                gpu_rows.append(
                    {
                        "name": name,
                        "scope": scope,
                        "category": ev_cat,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": dur_ns,
                        "bytes": nbytes,
                        "stream": stream,
                    }
                )

        if gpu_rows:
            kdf = pl.DataFrame(gpu_rows).sort("start_ns")
            kdf = enrich_ops_df(kdf, "name")
            if step_rows:
                step_rows.sort(key=lambda x: x[1])
                s_ids = np.array([x[0] for x in step_rows], dtype=np.int32)
                s_beg = np.array([x[1] for x in step_rows], dtype=np.int64)
                s_end = np.array([x[2] for x in step_rows], dtype=np.int64)
                k_starts = kdf["start_ns"].to_numpy()
                s_idx = np.searchsorted(s_beg, k_starts, side="right") - 1
                valid = (s_idx >= 0) & (k_starts <= s_end[np.maximum(0, s_idx)])
                step_col = np.where(valid, s_ids[np.maximum(0, s_idx)], -1).astype(np.int32)
                steps_df = conform(
                    pl.DataFrame({"step": s_ids, "start_ns": s_beg, "end_ns": s_end, "duration_ns": s_end - s_beg}),
                    STEPS_SCHEMA,
                )
            else:
                step_col = np.zeros(kdf.height, dtype=np.int32)
                s0 = int(kdf["start_ns"].to_numpy().min())
                e0 = int(kdf["end_ns"].to_numpy().max())
                steps_df = conform(
                    pl.DataFrame([{"step": 0, "start_ns": s0, "end_ns": e0, "duration_ns": e0 - s0}]),
                    STEPS_SCHEMA,
                )
            kernels_df = conform(kdf.with_columns(pl.Series("step", step_col, dtype=pl.Int32)), KERNELS_SCHEMA)
        else:
            kernels_df = pl.DataFrame(schema=KERNELS_SCHEMA)
            steps_df = pl.DataFrame(schema=STEPS_SCHEMA)

        runtime_df = (
            pl.DataFrame(
                [{"name": k, "duration_ns": v[0], "calls": v[1]} for k, v in rt_agg.items()],
                schema=RUNTIME_SCHEMA,
            )
            if rt_agg
            else pl.DataFrame(schema=RUNTIME_SCHEMA)
        )

        out: dict[str, Path] = {}
        for tname, df in (("kernels", kernels_df), ("steps", steps_df), ("runtime", runtime_df)):
            p = out_dir / f"{tname}.parquet"
            df.write_parquet(p, compression="zstd")
            out[tname] = p
        return out
