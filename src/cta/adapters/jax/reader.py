"""JAX profiler (XProf) adapter: ``jax.profiler.trace`` log directories.

A trace session (``<logdir>/plugins/profile/<run>/``) holds
``<host>.xplane.pb`` and, with ``create_perfetto_trace=True``,
``<host>.trace.json.gz`` and ``perfetto_trace.json.gz`` (or binary Perfetto
``.pftrace`` exports). These describe the same profile, so one source per host
is read: the XProf Chrome trace if present, else ``perfetto_trace.json.gz``,
else binary ``.pftrace`` exports, else ``<host>.xplane.pb`` (all parsed
natively in C++20 via ``cta._native.read_jax``).

Device ops are every event on a ``/device:*`` stream (GPU kernels, memcpys,
memsets) plus XLA:CPU thunks (events with an ``hlo_op`` argument on the host).
Steps come from ``jax.profiler.StepTraceAnnotation`` (``step_num``); without
them each execution of the busiest jitted module is a step. Work outside every
step (compilation, autotuning) is dropped. Work is assigned to the step whose
host code *launched* it (the ``correlation_id`` shared by a host launch event
and its device op), like the Kineto reader does.
"""

from __future__ import annotations

from collections.abc import Sequence
import gzip
import pathlib
import re
from typing import Any
import warnings
import zlib

import numpy as np
import polars as pl

from cta import _native as native
from cta import models
from cta import plugins
from cta import table_schemas
from cta.adapters import common
from cta.adapters import runtime_calls
from cta.analysis import alignment
from cta.analysis import network

_XPLANE_SUFFIXES = (".xplane.pb.gz", ".xplane.pb")
_HOST_TRACE_SUFFIX = ".trace.json.gz"
_PERFETTO_NAME = "perfetto_trace.json.gz"
# XProf and jax's Perfetto export both start with displayTimeUnit "ns";
# PyTorch/Kineto traces start with schemaVersion.
_XPROF_HEAD_RE = re.compile(r'\{\s*"displayTimeUnit"\s*:\s*"ns"')
_PFTRACE_SUFFIXES = (
    ".pftrace.gz",
    ".pftrace",
    ".perfetto-trace.gz",
    ".perfetto-trace",
)
_XLA_COLL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"all[-_]?reduce", re.IGNORECASE), "allreduce"),
    (re.compile(r"all[-_]?gather", re.IGNORECASE), "allgather"),
    (re.compile(r"reduce[-_]?scatter", re.IGNORECASE), "reducescatter"),
    (re.compile(r"all[-_]?to[-_]?all", re.IGNORECASE), "alltoall"),
    (re.compile(r"collective[-_]?permute", re.IGNORECASE), "collectivepermute"),
    (re.compile(r"^send[-.\d]", re.IGNORECASE), "send"),
    (re.compile(r"^recv[-.\d]", re.IGNORECASE), "recv"),
)


def _is_xplane_name(name: str) -> bool:
    """Returns whether ``name`` ends with an XPlane protobuf suffix."""
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in _XPLANE_SUFFIXES)


def _is_pftrace_name(name: str) -> bool:
    """Returns whether ``name`` ends with a binary Perfetto trace suffix."""
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in _PFTRACE_SUFFIXES)


def _host_stem(name: str) -> str | None:
    """Returns the host part of an XProf xplane/trace file name, else None."""
    for suffix in (*_XPLANE_SUFFIXES, _HOST_TRACE_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _has_pftrace_header(path: pathlib.Path) -> bool:
    """Checks that ``path`` begins with a Perfetto ``Trace.packet`` varint."""
    try:
        with path.open("rb") as fh:
            prefix = fh.read(2)
        if len(prefix) < 2:
            return False
        if prefix == b"\x1f\x8b" or path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as gz:
                head = gz.read(16)
        else:
            with path.open("rb") as fh:
                head = fh.read(16)
    except (OSError, EOFError, zlib.error):
        return False
    # Perfetto Trace has a single repeated field: TracePacket packet = 1
    # (tag byte 0x0a: field 1, wire type 2), followed by a base-128 length.
    if len(head) < 2 or head[0] != 0x0A:
        return False
    return any((byte & 0x80) == 0 for byte in head[1:11])


def select_sources(paths: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    """Selects one file per host and trace session directory.

    In each directory the XProf Chrome traces are read if there are any, else
    ``perfetto_trace.json.gz``, else binary ``.pftrace`` exports, else the
    ``xplane.pb`` files. Next to XProf traces, the ``xplane.pb`` files of
    hosts without a converted trace are read too.

    Args:
        paths: The files of a JAX artifact.

    Returns:
        The selected files, sorted; all of ``paths``, sorted, if none is
        selected.
    """
    selected: list[pathlib.Path] = []
    for session_dir in sorted({path.parent for path in paths}):
        session_paths = [path for path in paths if path.parent == session_dir]
        host_traces = [
            path
            for path in session_paths
            if path.name.endswith(_HOST_TRACE_SUFFIX)
        ]
        perfetto_traces = [
            path for path in session_paths if path.name == _PERFETTO_NAME
        ]
        pftrace_traces = [
            path for path in session_paths if _is_pftrace_name(path.name)
        ]
        converted_hosts = {_host_stem(path.name) for path in host_traces}
        xplane_by_host: dict[str | None, pathlib.Path] = {}
        for path in sorted(session_paths):
            if not _is_xplane_name(path.name):
                continue
            stem = _host_stem(path.name)
            if stem in converted_hosts:
                continue
            # Prefer plain .xplane.pb over .xplane.pb.gz for the same host.
            existing = xplane_by_host.get(stem)
            if existing is None or (
                existing.name.endswith(".gz") and not path.name.endswith(".gz")
            ):
                xplane_by_host[stem] = path
        unconverted_xplanes = list(xplane_by_host.values())
        selected += (
            host_traces
            or perfetto_traces
            or pftrace_traces
            or unconverted_xplanes
        )
        if host_traces:
            selected += unconverted_xplanes
    return sorted(set(selected)) or sorted(paths)


@plugins.register_adapter("jax", version="2.0")
class JaxAdapter:
    """Reads JAX profiler traces into kernels, steps and runtime tables.

    Attributes:
        format_name: The registered trace format name, ``"jax"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "jax"
    version = "2.0"

    def detect(self, path: pathlib.Path) -> bool:
        """Checks whether a file is a JAX/XProf or Perfetto profiler trace.

        Args:
            path: The candidate file.

        Returns:
            True for an ``.xplane.pb[.gz]`` file, a valid binary Perfetto
            ``.pftrace[.gz]`` / ``.perfetto-trace[.gz]`` file, or a JSON
            (optionally gzipped) trace starting with ``"displayTimeUnit":
            "ns"``.
        """
        name = path.name
        if _is_xplane_name(name):
            return True
        if _is_pftrace_name(name):
            return _has_pftrace_header(path)
        is_json = (
            name.endswith(_HOST_TRACE_SUFFIX)
            or name == _PERFETTO_NAME
            or name.endswith(".json")
            or name.endswith(".json.gz")
        )
        if not is_json:
            return False
        head = common.peek_text(path).lstrip()
        return _XPROF_HEAD_RE.match(head) is not None

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the artifact's traces to CTA's standard Parquet tables.

        One file per host is read (see ``select_sources``); each becomes a
        rank, numbered in sorted path order. The ``runtime`` table sums the
        host-side API calls made inside a step per call name, ``runtime_calls``
        records every host correlation call (``step = -1`` outside steps), and
        ``network_ops`` captures NCCL/XLA collectives and peer-to-peer copies.

        Args:
            artifact: The JAX artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"kernels"``, ``"steps"``, ``"runtime"``,
            ``"network_ops"`` and ``"runtime_calls"`` mapped to their Parquet
            paths.
        """
        kernel_frames: list[pl.DataFrame] = []
        step_frames: list[pl.DataFrame] = []
        net_frames: list[pl.DataFrame] = []
        call_frames: list[pl.DataFrame] = []
        runtime_by_name: dict[str, list[int]] = {}
        for rank, path in enumerate(select_sources(artifact.paths)):
            columns = native.read_jax(str(path))
            shard_kernels, shard_steps, shard_net, shard_calls = _ingest_shard(
                columns, rank, path.name, runtime_by_name
            )
            if shard_kernels.height:
                kernel_frames.append(shard_kernels)
            if shard_steps.height:
                step_frames.append(shard_steps)
            if shard_net.height:
                net_frames.append(shard_net)
            if shard_calls.height:
                call_frames.append(shard_calls)
        kernels = (
            pl.concat(kernel_frames).sort("rank", "start_ns")
            if kernel_frames
            else pl.DataFrame(schema=table_schemas.KERNELS)
        )
        steps = (
            pl.concat(step_frames).sort("rank", "start_ns")
            if step_frames
            else pl.DataFrame(schema=table_schemas.STEPS)
        )
        network_ops = (
            pl.concat(net_frames).sort("rank", "start_ns")
            if net_frames
            else pl.DataFrame(schema=table_schemas.NETWORK_OPS)
        )
        calls = (
            pl.concat(call_frames).sort("rank", "start_ns")
            if call_frames
            else pl.DataFrame(schema=table_schemas.RUNTIME_CALLS)
        )
        runtime = pl.DataFrame(
            [
                {"name": name, "duration_ns": totals[0], "calls": totals[1]}
                for name, totals in runtime_by_name.items()
            ],
            schema=table_schemas.RUNTIME,
        )
        return common.write_tables(
            out_dir,
            {
                "kernels": kernels,
                "steps": steps,
                "runtime": runtime,
                "network_ops": network_ops,
                "runtime_calls": calls,
            },
        )


def _assign_steps(
    launch_ns: np.ndarray, step_spans: list[tuple[int, int, int]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Assigns each op to the step span that contains its launch time.

    Args:
        launch_ns: The launch time of each op.
        step_spans: The (step, start_ns, end_ns) spans, sorted by start.

    Returns:
        The step of each op (-1 if it is outside every span) and the columns
        of the steps table.
    """
    step_ids = np.array([span[0] for span in step_spans], dtype=np.int32)
    step_starts_ns = np.array([span[1] for span in step_spans], dtype=np.int64)
    step_ends_ns = np.array([span[2] for span in step_spans], dtype=np.int64)
    # The candidate span is the last one starting at or before the launch.
    span_index = np.searchsorted(step_starts_ns, launch_ns, side="right") - 1
    clamped_index = np.maximum(0, span_index)
    is_in_span = (span_index >= 0) & (launch_ns <= step_ends_ns[clamped_index])
    step_of_op = np.where(is_in_span, step_ids[clamped_index], -1)
    step_columns: dict[str, np.ndarray] = {
        "step": step_ids,
        "start_ns": step_starts_ns,
        "end_ns": step_ends_ns,
        "duration_ns": step_ends_ns - step_starts_ns,
    }
    return step_of_op.astype(np.int32), step_columns


def _is_in_any_span(
    times_ns: np.ndarray, step_spans: list[tuple[int, int, int]]
) -> np.ndarray:
    """Returns whether each time lies inside at least one step span.

    Spans can nest, so a time past the end of the last span that starts
    before it can still be inside an earlier, longer one. Each time is
    compared with the latest end among the spans starting at or before it.

    Args:
        times_ns: The times to test.
        step_spans: The (step, start_ns, end_ns) spans, sorted by start.

    Returns:
        One boolean per time; all True when there are no spans, since the
        whole trace is then one step.
    """
    if not step_spans:
        return np.ones(len(times_ns), dtype=bool)
    starts_ns = np.array([span[1] for span in step_spans], dtype=np.int64)
    latest_ends_ns = np.maximum.accumulate(
        np.array([span[2] for span in step_spans], dtype=np.int64)
    )
    span_index = np.searchsorted(starts_ns, times_ns, side="right") - 1
    return (span_index >= 0) & (
        times_ns <= latest_ends_ns[np.maximum(0, span_index)]
    )


def _xla_collective_op(name: str, hlo_op: str) -> str | None:
    """Extracts a logical collective op name from an XLA HLO op or name."""
    for text in (hlo_op, name):
        if not text:
            continue
        for pattern, logical in _XLA_COLL_PATTERNS:
            if pattern.search(text):
                return logical
    return None


def _network_ops(
    kernels: pl.DataFrame,
    names: Sequence[str],
    hlo_op_codes: np.ndarray,
    launch_ns: np.ndarray,
) -> pl.DataFrame:
    """Builds the ``network_ops`` table of one rank from its JAX kernels.

    NCCL kernels use ``api="nccl"``, non-NCCL XLA/TPU HLO collectives use
    ``api="xla"``, and ``MemcpyP2P`` transfers use ``api="p2p"``.
    """
    if kernels.is_empty():
        return pl.DataFrame(schema=table_schemas.NETWORK_OPS)

    comm_mask = (kernels["category"] == "comm").to_numpy()
    comm_indices = np.flatnonzero(comm_mask)
    collective: pl.DataFrame | None = None
    if len(comm_indices):
        kernel_names = kernels["name"].to_list()
        coll_rows: list[int] = []
        coll_ops: list[str] = []
        for idx in comm_indices.tolist():
            hlo_op = names[int(hlo_op_codes[idx])]
            logical = _xla_collective_op(kernel_names[idx], hlo_op)
            if logical is not None:
                coll_rows.append(idx)
                coll_ops.append(logical)
        if coll_rows:
            n_coll = len(coll_rows)
            collective = pl.DataFrame(
                {
                    "row": pl.Series(coll_rows, dtype=pl.UInt32),
                    "collective": pl.Series(coll_ops, dtype=pl.String),
                    "dtype": pl.repeat("", n_coll, dtype=pl.String, eager=True),
                    "group": pl.repeat("", n_coll, dtype=pl.String, eager=True),
                    "in_elems": np.zeros(n_coll, dtype=np.int64),
                    "out_elems": np.zeros(n_coll, dtype=np.int64),
                    "group_size": np.zeros(n_coll, dtype=np.int64),
                }
            )

    ops = network.ops_from_kernels(kernels, collective, launch_ns)
    if not len(comm_indices):
        return ops

    # ops_from_kernels emits the comm rows first (in kernels order), followed
    # by peer_copies. Label non-NCCL XLA collectives as api="xla" and preserve
    # any kernel bytes recorded on comm rows.
    n_comm = len(comm_indices)
    comm_slice = ops.slice(0, n_comm)
    p2p_slice = ops.slice(n_comm)
    comm_kernels = kernels.filter(pl.col("category") == "comm")
    is_nccl = comm_kernels["name"].str.contains(r"(?i)nccl")
    comm_bytes = comm_kernels["bytes"]
    comm_slice = comm_slice.with_columns(
        pl.when(is_nccl)
        .then(pl.lit("nccl"))
        .otherwise(pl.lit("xla"))
        .alias("api"),
        pl.when(pl.col("bytes") > 0)
        .then(pl.col("bytes"))
        .otherwise(comm_bytes)
        .alias("bytes"),
    )
    return pl.concat([comm_slice, p2p_slice]).cast(table_schemas.NETWORK_OPS)


def _ingest_shard(
    columns: dict[str, Any],
    rank: int,
    source: str,
    runtime_by_name: dict[str, list[int]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Builds kernels, steps, network_ops and runtime_calls for one host.

    Args:
        columns: Native columns returned by ``native.read_jax``.
        rank: The rank to give the host.
        source: The trace's file name, for the warning.
        runtime_by_name: Per host-side API call name, the [total duration ns,
            call count] of the calls made inside a step; updated in place.

    Returns:
        The kernels, steps, network_ops, and runtime_calls frames.
    """
    runtime_by_name = {} if runtime_by_name is None else runtime_by_name
    names: list[str] = columns["names"]
    empty_steps = pl.DataFrame(schema=table_schemas.STEPS)
    if not len(columns["op_name"]):
        calls = runtime_calls.runtime_calls_frame(
            names,
            columns["rcall_name"],
            columns["rcall_start"],
            columns["rcall_end"],
            columns["rcall_thread"],
            columns["rcall_correlation"],
            rank,
            empty_steps,
        )
        return (
            pl.DataFrame(schema=table_schemas.KERNELS),
            empty_steps,
            pl.DataFrame(schema=table_schemas.NETWORK_OPS),
            calls,
        )

    order = np.argsort(columns["op_start"], kind="stable")
    op_name_codes = columns["op_name"][order]
    starts_ns = columns["op_start"][order]
    ends_ns = columns["op_end"][order]
    durations_ns = ends_ns - starts_ns
    op_launch_ns = columns["op_launch_ns"][order]
    op_hlo_op = columns["op_hlo_op"][order]
    op_hlo_module = columns["op_hlo_module"][order]
    row_count = len(op_name_codes)

    names_arr = np.array(names, dtype=object)
    used_codes, name_rows = np.unique(op_name_codes, return_inverse=True)
    distinct = alignment.enrich_ops_df(
        pl.DataFrame(
            {"name": pl.Series([names[c] for c in used_codes.tolist()])}
        ),
        "name",
    )
    kernels = pl.DataFrame(
        {
            "name": distinct["name"].gather(name_rows),
            "scope": pl.Series(
                names_arr[columns["op_scope"][order]], dtype=pl.String
            ),
            "scope_path": pl.Series(
                names_arr[columns["op_scope_path"][order]], dtype=pl.String
            ),
            "category": pl.Series(columns["op_category"][order]).replace_strict(
                table_schemas.GPU_CATEGORIES, return_dtype=pl.String
            ),
            "start_ns": starts_ns,
            "end_ns": ends_ns,
            "duration_ns": durations_ns,
            "bytes": columns["op_bytes"][order],
            "stream": columns["op_stream"][order],
            "device": columns["op_device"][order],
            "rank": np.full(row_count, rank, dtype=np.int64),
        }
    ).with_columns(
        distinct["demangled_name"].gather(name_rows),
        distinct["canonical_name"].gather(name_rows),
        distinct["op_family"].gather(name_rows),
    )

    # An op belongs to the step whose host code launched it; ops without a
    # known launch use their own start.
    launch_ns = np.where(
        np.isnan(op_launch_ns), starts_ns, op_launch_ns
    ).astype(np.int64)

    if len(columns["step_id"]):
        annotated = list(
            zip(
                columns["step_id"].tolist(),
                columns["step_start"].tolist(),
                columns["step_end"].tolist(),
                strict=True,
            )
        )
        step_spans = sorted(
            set(annotated), key=lambda span: (span[1], span[0], span[2])
        )
    else:
        warnings.warn(
            f"{source}: no jax.profiler.StepTraceAnnotation step markers; "
            "each host-side execution of the busiest jitted module is "
            "treated as a step (mark steps with "
            "StepTraceAnnotation(name, step_num=i) for exact boundaries).",
            RuntimeWarning,
            stacklevel=3,
        )
        step_spans = _infer_module_steps(
            names,
            op_hlo_module,
            op_hlo_op,
            durations_ns,
            columns["exec_module"],
            columns["exec_start"],
            columns["exec_end"],
        )

    step_columns: dict[str, Any]
    if step_spans:
        step_of_op, step_columns = _assign_steps(launch_ns, step_spans)
    else:
        step_of_op = np.zeros(row_count, dtype=np.int32)
        first_start_ns, last_end_ns = int(starts_ns.min()), int(ends_ns.max())
        step_columns = {
            "step": [0],
            "start_ns": [first_start_ns],
            "end_ns": [last_end_ns],
            "duration_ns": [last_end_ns - first_start_ns],
        }
    steps = table_schemas.conform(
        pl.DataFrame(step_columns).with_columns(pl.lit(rank).alias("rank")),
        table_schemas.STEPS,
    )
    kernels = kernels.with_columns(
        pl.Series("step", step_of_op, dtype=pl.Int32)
    )

    # Like the kernels, only the API calls made inside a step count in runtime.
    rcall_starts_ns = columns["rcall_start"]
    if len(rcall_starts_ns):
        is_in_step = _is_in_any_span(rcall_starts_ns, step_spans)
        in_step_codes = columns["rcall_name"][is_in_step]
        if len(in_step_codes):
            in_step_durs = (
                columns["rcall_end"][is_in_step] - rcall_starts_ns[is_in_step]
            )
            uniq_codes, first_pos, inv = np.unique(
                in_step_codes, return_index=True, return_inverse=True
            )
            dur_sums = np.bincount(inv, weights=in_step_durs).astype(np.int64)
            call_counts = np.bincount(inv).astype(np.int64)
            for idx in np.argsort(first_pos).tolist():
                api_name = names[int(uniq_codes[idx])]
                totals = runtime_by_name.setdefault(api_name, [0, 0])
                totals[0] += int(dur_sums[idx])
                totals[1] += int(call_counts[idx])

    calls = runtime_calls.runtime_calls_frame(
        names,
        columns["rcall_name"],
        columns["rcall_start"],
        columns["rcall_end"],
        columns["rcall_thread"],
        columns["rcall_correlation"],
        rank,
        steps,
    )

    if step_spans:
        # Compile/autotune work and transfers outside every step are not
        # step work.
        keep = step_of_op >= 0
        kernels = kernels.filter(pl.Series(keep))
        kept_hlo_op = op_hlo_op[keep]
        kept_launch_ns = op_launch_ns[keep]
    else:
        kept_hlo_op = op_hlo_op
        kept_launch_ns = op_launch_ns

    kernels = table_schemas.conform(kernels, table_schemas.KERNELS)
    net_ops = _network_ops(kernels, names, kept_hlo_op, kept_launch_ns)
    return kernels, steps, net_ops, calls


def _infer_module_steps(
    names: Sequence[str],
    op_hlo_module: np.ndarray,
    op_hlo_op: np.ndarray,
    durations_ns: np.ndarray,
    exec_module: np.ndarray,
    exec_start: np.ndarray,
    exec_end: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Infers steps from the host-side executions of the busiest module.

    The busiest jitted module is the one whose HLO ops take the most device
    time; each of its executions becomes a step, numbered in time order. Ops
    map to these steps by launch time.

    Args:
        names: Dictionary of interned names from the native reader.
        op_hlo_module: Module name code per op.
        op_hlo_op: HLO op name code per op.
        durations_ns: Duration in nanoseconds per op.
        exec_module: Module name code per host-side PJRT Execute span.
        exec_start: Start timestamps of executions in nanoseconds.
        exec_end: End timestamps of executions in nanoseconds.

    Returns:
        The (step, start_ns, end_ns) spans, sorted by start; empty if no
        execution of a module with HLO ops was recorded.
    """
    if not len(exec_module):
        return []
    device_ns_by_module: dict[str, int] = {}
    for mod_code, op_code, dur_ns in zip(
        op_hlo_module.tolist(),
        op_hlo_op.tolist(),
        durations_ns.tolist(),
        strict=True,
    ):
        module = names[mod_code]
        if module and names[op_code]:
            device_ns_by_module[module] = (
                device_ns_by_module.get(module, 0) + dur_ns
            )
    known_executions = [
        (names[mod_code], start_ns, end_ns)
        for mod_code, start_ns, end_ns in zip(
            exec_module.tolist(),
            exec_start.tolist(),
            exec_end.tolist(),
            strict=True,
        )
        if names[mod_code] in device_ns_by_module
    ]
    if not known_executions:
        return []
    busiest_module = max(
        sorted({execution[0] for execution in known_executions}),
        key=lambda module: device_ns_by_module[module],
    )
    busiest_executions = sorted(
        execution
        for execution in known_executions
        if execution[0] == busiest_module
    )
    return [
        (step, start_ns, end_ns)
        for step, (_, start_ns, end_ns) in enumerate(busiest_executions)
    ]
