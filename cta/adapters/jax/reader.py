"""JAX profiler (XProf) adapter: ``jax.profiler.trace`` log directories.

A trace session (``<logdir>/plugins/profile/<run>/``) holds ``<host>.xplane.pb`` and, with
``create_perfetto_trace=True``, ``<host>.trace.json.gz`` and ``perfetto_trace.json.gz``. The three describe the same
profile, so exactly one file per host is read: the XProf Chrome trace if present, else the Perfetto one, else the
``xplane.pb`` (converted with the optional ``xprof`` package: ``pip install cta[jax]``).

Device ops are every event on a ``/device:*`` stream (GPU kernels, memcpys, memsets) plus XLA:CPU thunks (events with an
``hlo_op`` argument on the host). Steps come from ``jax.profiler.StepTraceAnnotation`` (``step_num``); without them each
execution of the busiest jitted module is a step. Work outside every step (compilation, autotuning) is dropped. Work is assigned to the step whose host code *launched* it (the
``correlation_id`` shared by a host launch event and its device op), like the Kineto reader does.
"""

from __future__ import annotations

import gzip
import json
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cta.adapters.nsys.reader import KERNELS_SCHEMA, RUNTIME_SCHEMA, STEPS_SCHEMA, conform
from cta.core.alignment import enrich_ops_df
from cta.core.models import TraceArtifact
from cta.core.registry import register_adapter

_XPLANE = ".xplane.pb"
_HOST_JSON = ".trace.json.gz"
_PERFETTO = "perfetto_trace.json.gz"
_COMM_RE = re.compile(r"all[-_]?reduce|all[-_]?gather|reduce[-_]?scatter|all[-_]?to[-_]?all|collective[-_]?permute|nccl|^(send|recv)[-.\d]", re.IGNORECASE)
_COPY_RE = re.compile(r"memcpy|memset|copy[-_]?(start|done)|infeed|outfeed", re.IGNORECASE)
_EXECUTE = "CommonPjRtLoadedExecutable::Execute ("  # host event per execution of a compiled module (PJRT)
_BYTES_RE = re.compile(r"(?:size|num_bytes):(\d+)")
_DEVICE_RE = re.compile(r"/device:(?:[A-Za-z]+):(\d+)")


def _head(path: Path, nbytes: int = 512) -> str:
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            return fh.read(nbytes).decode("utf-8", errors="ignore")  # type: ignore[no-any-return]
    except OSError:
        return ""


def _host_stem(name: str) -> str | None:
    for suffix in (_XPLANE, _HOST_JSON):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def select_sources(paths: Sequence[Path]) -> list[Path]:
    """One file per host and session directory: XProf trace, else the Perfetto export, else the xplane.pb."""
    chosen: list[Path] = []
    for folder in sorted({p.parent for p in paths}):
        here = [p for p in paths if p.parent == folder]
        hosts = [p for p in here if p.name.endswith(_HOST_JSON)]
        perfetto = [p for p in here if p.name == _PERFETTO]
        have = {_host_stem(p.name) for p in hosts}
        pb = [p for p in here if p.name.endswith(_XPLANE) and _host_stem(p.name) not in have]
        chosen += hosts or perfetto or pb
        if hosts:
            chosen += [p for p in pb]  # hosts without a converted trace
    return sorted(set(chosen)) or sorted(paths)


def _xplane_to_events(path: Path) -> list[dict[str, Any]]:
    try:
        from xprof.convert import raw_to_tool_data  # optional dependency
    except ImportError as exc:
        raise ImportError(f"{path.name} is an XPlane protobuf; converting it needs xprof: pip install 'cta[jax]'") from exc
    data, _ = raw_to_tool_data.xspace_to_tool_data([str(path)], "trace_viewer", {})
    doc = json.loads(data if isinstance(data, (str, bytes)) else str(data))
    return list(doc.get("traceEvents", []))  # type: ignore[no-any-return]


def _load_events(path: Path) -> list[dict[str, Any]]:
    if path.name.endswith(_XPLANE):
        return _xplane_to_events(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    return list(doc.get("traceEvents", [])) if isinstance(doc, dict) else list(doc)


@register_adapter("jax", version="1.0")
class JaxAdapter:
    format_name = "jax"
    version = "1.0"

    def detect(self, path: Path) -> bool:
        name = path.name
        if name.endswith(_XPLANE):
            return True
        if not (name.endswith(_HOST_JSON) or name == _PERFETTO or name.endswith(".json") or name.endswith(".json.gz")):
            return False
        head = _head(path).lstrip()
        # XProf and jax's Perfetto export both start with displayTimeUnit "ns"; PyTorch/Kineto traces start with schemaVersion.
        return re.match(r'\{\s*"displayTimeUnit"\s*:\s*"ns"', head) is not None

    def ingest(self, artifact: TraceArtifact, out_dir: Path) -> dict[str, Path]:
        kernel_frames: list[pl.DataFrame] = []
        step_frames: list[pl.DataFrame] = []
        rt_agg: dict[str, list[int]] = {}
        for rank, path in enumerate(select_sources(artifact.paths)):
            kdf, sdf = _ingest_shard(_load_events(path), rank, path.name, rt_agg)
            if kdf.height:
                kernel_frames.append(kdf)
            if sdf.height:
                step_frames.append(sdf)
        kernels_df = pl.concat(kernel_frames).sort("rank", "start_ns") if kernel_frames else pl.DataFrame(schema=KERNELS_SCHEMA)
        steps_df = pl.concat(step_frames).sort("rank", "start_ns") if step_frames else pl.DataFrame(schema=STEPS_SCHEMA)
        runtime_df = pl.DataFrame([{"name": k, "duration_ns": v[0], "calls": v[1]} for k, v in rt_agg.items()], schema=RUNTIME_SCHEMA)
        out: dict[str, Path] = {}
        for tname, df in (("kernels", kernels_df), ("steps", steps_df), ("runtime", runtime_df)):
            p = out_dir / f"{tname}.parquet"
            df.write_parquet(p, compression="zstd")
            out[tname] = p
        return out


def _ns(us: Any) -> int:
    return int(round(float(us) * 1000.0))


def _scope_path(args: dict[str, Any], module: str) -> list[str]:
    """HLO name stack, e.g. ``jit(train_step)/jvp()/dot_general``: GPU ops carry it in ``name``, others in ``tf_op``."""
    for key in ("name", "tf_op"):
        val = str(args.get(key) or "")
        if "/" in val or val.startswith("jit("):
            return [seg for seg in val.split("/") if seg]
    if module.startswith(("jit_", "pjit_", "pmap_")):  # same spelling as the name stack: jit_f -> jit(f)
        kind, _, fn = module.partition("_")
        return [f"{kind}({fn})"]
    return [module] if module else []


def _ingest_shard(
    events: list[dict[str, Any]], rank: int, source: str, rt_agg: dict[str, list[int]] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Kernels and steps of one host's trace; host-side CUDA API calls (events with a correlation id) go into rt_agg."""
    rt_agg = {} if rt_agg is None else rt_agg
    proc_names: dict[Any, str] = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            proc_names[ev.get("pid")] = str((ev.get("args") or {}).get("name", ""))
    device_of_pid = {pid: int(m.group(1)) for pid, name in proc_names.items() if (m := _DEVICE_RE.search(name))}
    thread_names: dict[tuple[Any, Any], str] = {
        (ev.get("pid"), ev.get("tid")): str((ev.get("args") or {}).get("name", ""))
        for ev in events
        if ev.get("ph") == "M" and ev.get("name") == "thread_name"
    }
    # A device process may also carry XProf's derived lines (XLA Ops, XLA Modules, Steps, ...) that repeat the raw
    # stream events: read the raw GPU streams, or on TPU the XLA Ops line.
    line_filter: dict[Any, str] = {}
    for pid in device_of_pid:
        names = [n for (p, _), n in thread_names.items() if p == pid]
        line_filter[pid] = "Stream #" if any(n.startswith("Stream #") for n in names) else "XLA Ops" if "XLA Ops" in names else ""

    rows: list[dict[str, object]] = []
    extra: list[tuple[str, str, int]] = []  # (hlo module, hlo op, correlation id) per row
    annotated: list[tuple[int, int, int]] = []  # (step_num, start_ns, end_ns)
    launches: dict[int, int] = {}  # correlation id -> host launch time (ns)
    api_calls: list[tuple[str, int, int]] = []  # host CUDA API calls: (name, start, duration)
    executes: list[tuple[str, int, int]] = []  # host-side executions of a jitted module: (module, start, end)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = _ns(ev.get("dur", 0.0))
        if dur <= 0:
            continue
        start = _ns(ev.get("ts", 0.0))
        raw = ev.get("args")
        args: dict[str, Any] = raw if isinstance(raw, dict) else {}
        pid = ev.get("pid")
        corr = int(args["correlation_id"]) if str(args.get("correlation_id", "")).isdigit() else -1
        on_device = pid in device_of_pid
        if "step_num" in args and str(args["step_num"]).lstrip("-").isdigit() and not on_device:
            annotated.append((int(args["step_num"]), start, start + dur))
            continue
        if not on_device and str(ev.get("name", "")).startswith(_EXECUTE) and args.get("name"):
            executes.append((str(args["name"]), start, start + dur))
        if not on_device and "hlo_op" not in args:
            if corr >= 0:
                launches.setdefault(corr, start)
                api_calls.append((str(ev.get("name", "")), start, dur))
            continue
        if on_device and not thread_names.get((pid, ev.get("tid")), "").startswith(line_filter.get(pid, "")):
            continue
        name = str(ev.get("name", ""))
        hlo_op = str(args.get("hlo_op") or name)
        module = str(args.get("hlo_module") or args.get("module") or "")
        path = _scope_path(args, module)
        category = "comm" if _COMM_RE.search(name) or _COMM_RE.search(hlo_op) else "memcpy" if _COPY_RE.search(name) else "compute"
        m_bytes = _BYTES_RE.search(str(args.get("memcpy_details") or args.get("Memset_details") or ""))
        device = device_of_pid.get(pid, int(args["device_ordinal"]) if str(args.get("device_ordinal", "")).isdigit() else 0)
        rows.append(
            {
                "name": name, "scope": path[-1] if path else "", "scope_path": " > ".join(path), "category": category,
                "start_ns": start, "end_ns": start + dur, "duration_ns": dur, "bytes": int(m_bytes.group(1)) if m_bytes else 0,
                "stream": int(ev.get("tid", 0) or 0) % 2**31, "device": device, "rank": rank,
            }
        )  # fmt: skip
        extra.append((module, str(args.get("hlo_op") or ""), corr))

    if not rows:
        return pl.DataFrame(schema=KERNELS_SCHEMA), pl.DataFrame(schema=STEPS_SCHEMA)

    order = np.argsort([r["start_ns"] for r in rows], kind="stable")
    kdf = enrich_ops_df(pl.DataFrame(rows)[order.tolist()], "name")
    extra = [extra[i] for i in order]
    starts = kdf["start_ns"].to_numpy()
    ends = kdf["end_ns"].to_numpy()
    # An op belongs to the step whose host code launched it; ops without a known launch use their own start.
    launch = np.array([launches.get(c, int(starts[i])) if c >= 0 else int(starts[i]) for i, (_, _, c) in enumerate(extra)], dtype=np.int64)

    if annotated:
        spans = sorted(set(annotated), key=lambda x: x[1])
        by_launch = True
    else:
        warnings.warn(
            f"{source}: no jax.profiler.StepTraceAnnotation step markers; each host-side execution of the busiest jitted module "
            "is treated as a step (mark steps with StepTraceAnnotation(name, step_num=i) for exact boundaries).",
            RuntimeWarning,
            stacklevel=3,
        )
        spans = _infer_module_steps(kdf, extra, executes)
        by_launch = True

    if spans:
        s_ids = np.array([s[0] for s in spans], dtype=np.int32)
        s_beg = np.array([s[1] for s in spans], dtype=np.int64)
        s_end = np.array([s[2] for s in spans], dtype=np.int64)
        q = launch if by_launch else starts  # inferred spans are device-time windows
        idx = np.searchsorted(s_beg, q, side="right") - 1
        safe = np.maximum(0, idx)
        valid = (idx >= 0) & (q <= s_end[safe])
        step_col = np.where(valid, s_ids[safe], -1).astype(np.int32)
        steps = {"step": s_ids, "start_ns": s_beg, "end_ns": s_end, "duration_ns": s_end - s_beg}
    else:
        step_col = np.zeros(kdf.height, dtype=np.int32)
        s0, e0 = int(starts.min()), int(ends.max())
        steps = {"step": [0], "start_ns": [s0], "end_ns": [e0], "duration_ns": [e0 - s0]}
    steps_df = conform(pl.DataFrame(steps).with_columns(pl.lit(rank).alias("rank")), STEPS_SCHEMA)
    kdf = kdf.with_columns(pl.Series("step", step_col, dtype=pl.Int32))
    for api, a_start, a_dur in api_calls:  # like the kernels, only calls made inside a step count
        if not spans or any(s <= a_start <= e for _, s, e in spans):
            agg = rt_agg.setdefault(api, [0, 0])
            agg[0] += a_dur
            agg[1] += 1
    if spans:  # compile/autotune work and transfers outside every step are not step work
        kdf = kdf.filter(pl.col("step") >= 0)
    return conform(kdf, KERNELS_SCHEMA), steps_df


def _infer_module_steps(
    kdf: pl.DataFrame, extra: list[tuple[str, str, int]], executes: list[tuple[str, int, int]]
) -> list[tuple[int, int, int]]:
    """One step per host-side execution of the jitted module with the most device time (ops map to it by launch time)."""
    dur = kdf["duration_ns"].to_numpy()
    totals: dict[str, int] = {}
    for (module, op, _), d in zip(extra, dur, strict=True):
        if module and op:
            totals[module] = totals.get(module, 0) + int(d)
    runs = [e for e in executes if e[0] in totals]
    if not runs:
        return []
    top = max({e[0] for e in runs}, key=lambda m: totals[m])
    return [(n, s, e) for n, (_, s, e) in enumerate(sorted(x for x in runs if x[0] == top))]
