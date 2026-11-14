"""PyTorch Profiler / Kineto Chrome Trace (.json and .json.gz) adapter.

Every file of an artifact is one rank's trace with its own clock. GPU kernels,
memcpys and memsets are attributed to the CPU op and ``ProfilerStep#N`` range
that launched them, matched through the CUDA runtime call's correlation id.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re
from typing import Any

import numpy as np
import polars as pl

from cta.adapters.nsys import reader as nsys_reader
from cta.core import alignment
from cta.core import bundle
from cta.core import intervals
from cta.core import models
from cta.core import registry

_XPROF_HEAD_RE = re.compile(r'\{\s*"displayTimeUnit"\s*:\s*"ns"')
_STEP_RE = re.compile(r"profilerstep#\s*(\d+)", re.IGNORECASE)

# Event categories (lower-cased ``cat``) of CPU ops, CUDA API calls and GPU
# activities.
_CPU_OP_CATEGORIES = ("cpu_op", "operator", "user_annotation")
_RUNTIME_CATEGORIES = ("cuda_runtime", "cuda_driver", "runtime")
_GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset", "memcpy", "memset")


def _open_text_head(path: pathlib.Path, max_bytes: int = 8192) -> str:
    """Returns the first bytes of a (possibly gzipped) file as text.

    Args:
        path: The file to read; a ``.gz`` suffix means gzip-compressed.
        max_bytes: The number of (decompressed) bytes to read at most.

    Returns:
        The decoded head, or "" if the file cannot be read.
    """
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as gz:
                return gz.read(max_bytes).decode("utf-8", errors="ignore")
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="ignore")
    except OSError:
        return ""


@registry.register_adapter("kineto", version="2.2")
class KinetoAdapter:
    """Reads PyTorch Profiler (Kineto) Chrome traces, one file per rank.

    Attributes:
        format_name: The registered format name, ``"kineto"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "kineto"
    version = "2.2"

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file looks like a Kineto Chrome trace.

        Args:
            path: The candidate file.

        Returns:
            True for a ``.json`` or ``.json.gz`` file whose head contains
            ``"traceEvents"``. JAX/XProf traces (``displayTimeUnit`` "ns")
            are rejected: they belong to the jax adapter.
        """
        lower_name = path.name.lower()
        if not (
            lower_name.endswith(".json") or lower_name.endswith(".json.gz")
        ):
            return False
        head = _open_text_head(path)
        if _XPROF_HEAD_RE.match(head.lstrip()):
            return False
        return '"traceEvents"' in head

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Ingests every shard of the artifact (one trace per rank).

        Each shard keeps its own clock and rank. CUDA runtime totals are
        summed over all shards.

        Args:
            artifact: The artifact whose trace files are read.
            out_dir: The directory the Parquet tables are written to.

        Returns:
            The paths of the ``kernels``, ``steps`` and ``runtime`` Parquet
            tables, by table name.
        """
        kernel_frames: list[pl.DataFrame] = []
        step_frames: list[pl.DataFrame] = []
        # Runtime API name -> [total duration in ns, number of calls].
        runtime_totals: dict[str, list[int]] = {}

        for shard_index, path in enumerate(sorted(artifact.paths)):
            trace = _load_trace(path)
            rank = _trace_rank(trace, path, shard_index)
            shard_kernels, shard_steps = _ingest_shard(
                trace, rank, runtime_totals
            )
            if shard_kernels.height:
                kernel_frames.append(shard_kernels)
            if shard_steps.height:
                step_frames.append(shard_steps)

        if kernel_frames:
            kernels = pl.concat(kernel_frames).sort("rank", "start_ns")
        else:
            kernels = pl.DataFrame(schema=nsys_reader.KERNELS_SCHEMA)
        if step_frames:
            steps = pl.concat(step_frames).sort("rank", "start_ns")
        else:
            steps = pl.DataFrame(schema=nsys_reader.STEPS_SCHEMA)
        if runtime_totals:
            runtime_rows = [
                {"name": name, "duration_ns": duration_ns, "calls": calls}
                for name, (duration_ns, calls) in runtime_totals.items()
            ]
            runtime = pl.DataFrame(
                runtime_rows, schema=nsys_reader.RUNTIME_SCHEMA
            )
        else:
            runtime = pl.DataFrame(schema=nsys_reader.RUNTIME_SCHEMA)

        table_paths: dict[str, pathlib.Path] = {}
        for table_name, table in (
            ("kernels", kernels),
            ("steps", steps),
            ("runtime", runtime),
        ):
            table_path = out_dir / f"{table_name}.parquet"
            table.write_parquet(table_path, compression="zstd")
            table_paths[table_name] = table_path
        return table_paths


def _load_trace(path: pathlib.Path) -> Any:
    """Parses a (possibly gzipped) JSON trace file."""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as gz:
            return json.load(gz)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _trace_rank(trace: Any, path: pathlib.Path, index: int) -> int:
    """Returns the rank of a shard.

    The rank is the trace's own ``distributedInfo.rank``, else a rank in the
    file name, else the shard's index.
    """
    info = trace.get("distributedInfo") if isinstance(trace, dict) else None
    if isinstance(info, dict) and isinstance(info.get("rank"), int):
        return int(info["rank"])
    return bundle.infer_rank(path, index)


def _event_span_ns(event: dict[str, Any]) -> tuple[int, int]:
    """Returns an event's (start, duration) in ns.

    Kineto timestamps and durations are in microseconds and can be floats
    with a fractional nanosecond part.
    """
    start_ns = int(round(float(event.get("ts", 0.0)) * 1000.0))
    duration_ns = int(round(float(event.get("dur", 0.0)) * 1000.0))
    return start_ns, duration_ns


def _ingest_shard(
    trace: Any, rank: int, runtime_totals: dict[str, list[int]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds the kernels and steps tables of one rank's trace.

    All timestamps of a shard share one clock. CUDA runtime call durations
    and counts are added to ``runtime_totals`` as a side effect.

    Args:
        trace: The parsed trace: a dict with ``traceEvents`` or a bare list
            of events.
        rank: The rank written to every row.
        runtime_totals: Runtime API name -> [total duration in ns, calls];
            updated in place.

    Returns:
        The kernels and steps tables, conformed to the nsys schemas.
    """
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    gpu_rows: list[dict[str, object]] = []
    # (step id, start ns, end ns) of every ProfilerStep range.
    step_rows: list[tuple[int, int, int]] = []
    op_names: list[str] = []
    # Thread id -> (start ns, end ns, index into op_names) of its CPU ops.
    ops_by_thread: dict[int, list[tuple[int, int, int]]] = {}
    thread_id_by_pid_tid: dict[tuple[object, object], int] = {}
    # Correlation id -> (launch time ns, thread id) of a CUDA runtime call.
    launch_by_correlation: dict[int, tuple[int, int]] = {}

    def thread_id_of(event: dict[str, Any]) -> int:
        """Returns a dense id for the event's (pid, tid) pair."""
        return thread_id_by_pid_tid.setdefault(
            (event.get("pid"), event.get("tid")), len(thread_id_by_pid_tid)
        )

    # Pass 1: collect ProfilerStep intervals, CPU ops per thread, and the
    # thread and time of every CUDA launch.
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        start_ns, duration_ns = _event_span_ns(event)
        if duration_ns <= 0:
            continue
        end_ns = start_ns + duration_ns

        step_match = _STEP_RE.search(name)
        if step_match:
            # PyTorch also emits each step as a `gpu_user_annotation` (the
            # GPU-side projection of the same step); counting it would add a
            # second, differently-timed step with the same id.
            if category != "gpu_user_annotation":
                step_rows.append((int(step_match.group(1)), start_ns, end_ns))
            continue

        if category in _CPU_OP_CATEGORIES:
            ops_by_thread.setdefault(thread_id_of(event), []).append(
                (start_ns, end_ns, len(op_names))
            )
            op_names.append(name)
        elif category in _RUNTIME_CATEGORIES:
            totals = runtime_totals.setdefault(name, [0, 0])
            totals[0] += duration_ns
            totals[1] += 1
            args = event.get("args")
            if isinstance(args, dict) and "correlation" in args:
                launch_by_correlation[int(args["correlation"])] = (
                    start_ns,
                    thread_id_of(event),
                )

    # Pass 2: collect GPU kernels and memcpy/memset; remember when and from
    # which thread each was launched.
    launch_times: list[float] = []
    launch_thread_ids: list[int] = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        if category not in _GPU_CATEGORIES:
            continue
        name = str(event.get("name", ""))
        start_ns, duration_ns = _event_span_ns(event)
        if duration_ns <= 0:
            continue
        end_ns = start_ns + duration_ns
        raw_args = event.get("args")
        gpu_args: dict[str, Any] = (
            raw_args if isinstance(raw_args, dict) else {}
        )
        correlation_id = (
            int(gpu_args["correlation"]) if "correlation" in gpu_args else -1
        )
        stream = int(gpu_args.get("stream", event.get("tid", 0)) or 0)
        nbytes = int(gpu_args.get("bytes", 0) or 0)
        if isinstance(gpu_args.get("device"), int):
            device = int(gpu_args["device"])
        else:
            device = int(event.get("pid", 0) or 0)

        if "memcpy" in category or "memset" in category:
            row_category = "memcpy"
        elif "nccl" in name.lower():
            row_category = "comm"
        else:
            row_category = "compute"

        launch = launch_by_correlation.get(correlation_id)
        launch_times.append(float(launch[0]) if launch else float("nan"))
        launch_thread_ids.append(launch[1] if launch else -1)
        gpu_rows.append(
            {
                "name": name,
                "scope": "",
                "category": row_category,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
                "bytes": nbytes,
                "stream": stream,
                "device": device,
                "rank": rank,
            }
        )

    if not gpu_rows:
        return (
            pl.DataFrame(schema=nsys_reader.KERNELS_SCHEMA),
            pl.DataFrame(schema=nsys_reader.STEPS_SCHEMA),
        )

    # Sort by GPU start time; the launch arrays are permuted alongside so
    # they stay aligned with the sorted rows.
    order = np.argsort([row["start_ns"] for row in gpu_rows], kind="stable")
    kernels = alignment.enrich_ops_df(
        pl.DataFrame(gpu_rows)[order.tolist()], "name"
    )
    launch_ns = np.asarray(launch_times, dtype=np.float64)[order]
    launch_thread = np.asarray(launch_thread_ids, dtype=np.int64)[order]
    gpu_start_ns = kernels["start_ns"].to_numpy()

    # Scope = the innermost CPU op that encloses the launch *on the launching
    # thread* (a finished sibling that started later does not hide the
    # enclosing op). Kernels without a known launch get no scope.
    if op_names:
        ranges_by_thread = {
            thread_id: intervals.build_thread_ranges(list(op_rows))
            for thread_id, op_rows in ops_by_thread.items()
        }
        op_start_ns = np.zeros(len(op_names), dtype=np.int64)
        for op_rows in ops_by_thread.values():
            for range_start_ns, _, op_index in op_rows:
                op_start_ns[op_index] = range_start_ns
        op_ids = intervals.assign_ranges(
            ranges_by_thread,
            op_start_ns,
            launch_ns,
            launch_thread,
            gpu_start_ns,
        )
        # Range id -1 (no enclosing op) indexes the trailing "".
        names_with_blank = np.array([*op_names, ""], dtype=object)
        has_launch = launch_thread >= 0
        scopes = np.where(has_launch, names_with_blank[op_ids], "")
        scope_paths = np.where(
            has_launch,
            intervals.range_paths(ranges_by_thread, op_ids, op_names),
            "",
        )
        kernels = kernels.with_columns(
            pl.Series("scope", scopes, dtype=pl.String),
            pl.Series("scope_path", scope_paths, dtype=pl.String),
        )

    # Launches are asynchronous: a kernel belongs to the step whose CPU range
    # issued it, not the one it runs in.
    issue_ns = np.where(np.isnan(launch_ns), gpu_start_ns, launch_ns).astype(
        np.int64
    )
    if step_rows:
        step_rows.sort(key=lambda step_row: step_row[1])
        step_ids = np.array([row[0] for row in step_rows], dtype=np.int32)
        step_start_ns = np.array([row[1] for row in step_rows], dtype=np.int64)
        step_end_ns = np.array([row[2] for row in step_rows], dtype=np.int64)
        # Latest step starting at or before each issue time (-1 if none).
        step_pos = np.searchsorted(step_start_ns, issue_ns, side="right") - 1
        safe_pos = np.maximum(0, step_pos)
        is_in_step = (step_pos >= 0) & (issue_ns <= step_end_ns[safe_pos])
        kernel_steps = np.where(is_in_step, step_ids[safe_pos], -1).astype(
            np.int32
        )
        step_columns = {
            "step": step_ids,
            "start_ns": step_start_ns,
            "end_ns": step_end_ns,
            "duration_ns": step_end_ns - step_start_ns,
        }
    else:
        # No ProfilerStep markers: the whole trace is step 0.
        kernel_steps = np.zeros(kernels.height, dtype=np.int32)
        trace_start_ns = int(kernels["start_ns"].to_numpy().min())
        trace_end_ns = int(kernels["end_ns"].to_numpy().max())
        step_columns = {
            "step": [0],
            "start_ns": [trace_start_ns],
            "end_ns": [trace_end_ns],
            "duration_ns": [trace_end_ns - trace_start_ns],
        }
    steps = nsys_reader.conform(
        pl.DataFrame(step_columns).with_columns(pl.lit(rank).alias("rank")),
        nsys_reader.STEPS_SCHEMA,
    )
    kernels = nsys_reader.conform(
        kernels.with_columns(pl.Series("step", kernel_steps, dtype=pl.Int32)),
        nsys_reader.KERNELS_SCHEMA,
    )
    return kernels, steps
