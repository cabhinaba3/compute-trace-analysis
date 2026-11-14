"""JAX profiler (XProf) adapter: ``jax.profiler.trace`` log directories.

A trace session (``<logdir>/plugins/profile/<run>/``) holds
``<host>.xplane.pb`` and, with ``create_perfetto_trace=True``,
``<host>.trace.json.gz`` and ``perfetto_trace.json.gz``. The three describe
the same profile, so exactly one file per host is read: the XProf Chrome trace
if present, else the Perfetto one, else the ``xplane.pb`` (converted with the
optional ``xprof`` package: ``pip install cta[jax]``).

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
import dataclasses
import gzip
import json
import pathlib
import re
from typing import Any
import warnings

import numpy as np
import polars as pl

from cta.adapters.nsys import reader as nsys_reader
from cta.core import alignment
from cta.core import models
from cta.core import registry

_XPLANE_SUFFIX = ".xplane.pb"
_HOST_TRACE_SUFFIX = ".trace.json.gz"
_PERFETTO_NAME = "perfetto_trace.json.gz"
_COMM_RE = re.compile(
    r"all[-_]?reduce|all[-_]?gather|reduce[-_]?scatter|all[-_]?to[-_]?all"
    r"|collective[-_]?permute|nccl|^(send|recv)[-.\d]",
    re.IGNORECASE,
)
_COPY_RE = re.compile(
    r"memcpy|memset|copy[-_]?(start|done)|infeed|outfeed", re.IGNORECASE
)
# Name prefix of the host event of each execution of a compiled module (PJRT).
_EXECUTE_PREFIX = "CommonPjRtLoadedExecutable::Execute ("
_BYTES_RE = re.compile(r"(?:size|num_bytes):(\d+)")
_DEVICE_RE = re.compile(r"/device:(?:[A-Za-z]+):(\d+)")


def _read_head(path: pathlib.Path, num_bytes: int = 512) -> str:
    """Returns the first bytes of a (possibly gzipped) file as text.

    Returns an empty string if the file cannot be read.
    """
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            head: bytes = fh.read(num_bytes)
            return head.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _host_stem(name: str) -> str | None:
    """Returns the host part of an XProf xplane/trace file name, else None."""
    for suffix in (_XPLANE_SUFFIX, _HOST_TRACE_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def select_sources(paths: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    """Selects one file per host and trace session directory.

    In each directory the XProf Chrome traces are read if there are any, else
    the Perfetto export, else the xplane.pb files. Next to XProf traces, the
    xplane.pb files of hosts without a converted trace are read too.

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
        converted_hosts = {_host_stem(path.name) for path in host_traces}
        unconverted_xplanes = [
            path
            for path in session_paths
            if path.name.endswith(_XPLANE_SUFFIX)
            and _host_stem(path.name) not in converted_hosts
        ]
        selected += host_traces or perfetto_traces or unconverted_xplanes
        if host_traces:
            selected += unconverted_xplanes
    return sorted(set(selected)) or sorted(paths)


def _xplane_to_events(path: pathlib.Path) -> list[dict[str, Any]]:
    """Converts an XPlane protobuf to trace events with the ``xprof`` package.

    Raises:
        ImportError: If the optional ``xprof`` package is not installed.
    """
    try:
        from xprof.convert import raw_to_tool_data  # optional dependency
    except ImportError as exc:
        raise ImportError(
            f"{path.name} is an XPlane protobuf; converting it needs xprof: "
            "pip install 'cta[jax]'"
        ) from exc
    tool_data, _ = raw_to_tool_data.xspace_to_tool_data(
        [str(path)], "trace_viewer", {}
    )
    trace_json = (
        tool_data if isinstance(tool_data, (str, bytes)) else str(tool_data)
    )
    trace = json.loads(trace_json)
    return list(trace.get("traceEvents", []))


def _load_events(path: pathlib.Path) -> list[dict[str, Any]]:
    """Returns the trace events of an XProf/Perfetto JSON trace or xplane.pb."""
    if path.name.endswith(_XPLANE_SUFFIX):
        return _xplane_to_events(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        trace = json.load(fh)
    if isinstance(trace, dict):
        return list(trace.get("traceEvents", []))
    return list(trace)


@registry.register_adapter("jax", version="1.0")
class JaxAdapter:
    """Reads JAX profiler traces into kernels, steps and runtime tables.

    Attributes:
        format_name: The registered trace format name, ``"jax"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "jax"
    version = "1.0"

    def detect(self, path: pathlib.Path) -> bool:
        """Checks whether a file is a JAX profiler trace.

        Args:
            path: The candidate file.

        Returns:
            True for an xplane.pb file, or a JSON (optionally gzipped) trace
            that starts with ``"displayTimeUnit": "ns"``.
        """
        name = path.name
        if name.endswith(_XPLANE_SUFFIX):
            return True
        is_json = (
            name.endswith(_HOST_TRACE_SUFFIX)
            or name == _PERFETTO_NAME
            or name.endswith(".json")
            or name.endswith(".json.gz")
        )
        if not is_json:
            return False
        head = _read_head(path).lstrip()
        # XProf and jax's Perfetto export both start with displayTimeUnit
        # "ns"; PyTorch/Kineto traces start with schemaVersion.
        return re.match(r'\{\s*"displayTimeUnit"\s*:\s*"ns"', head) is not None

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the artifact's traces to kernels/steps/runtime tables.

        One file per host is read (see ``select_sources``); each becomes a
        rank, numbered in sorted path order. The runtime table sums the
        host-side API calls made inside a step per call name.

        Args:
            artifact: The JAX artifact to read.
            out_dir: The directory to write the Parquet tables to.

        Returns:
            The table names ``"kernels"``, ``"steps"`` and ``"runtime"``
            mapped to their Parquet paths.
        """
        kernel_frames: list[pl.DataFrame] = []
        step_frames: list[pl.DataFrame] = []
        runtime_by_name: dict[str, list[int]] = {}
        for rank, path in enumerate(select_sources(artifact.paths)):
            shard_kernels, shard_steps = _ingest_shard(
                _load_events(path), rank, path.name, runtime_by_name
            )
            if shard_kernels.height:
                kernel_frames.append(shard_kernels)
            if shard_steps.height:
                step_frames.append(shard_steps)
        kernels = (
            pl.concat(kernel_frames).sort("rank", "start_ns")
            if kernel_frames
            else pl.DataFrame(schema=nsys_reader.KERNELS_SCHEMA)
        )
        steps = (
            pl.concat(step_frames).sort("rank", "start_ns")
            if step_frames
            else pl.DataFrame(schema=nsys_reader.STEPS_SCHEMA)
        )
        runtime = pl.DataFrame(
            [
                {"name": name, "duration_ns": totals[0], "calls": totals[1]}
                for name, totals in runtime_by_name.items()
            ],
            schema=nsys_reader.RUNTIME_SCHEMA,
        )
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


def _us_to_ns(us: Any) -> int:
    """Converts a trace time in microseconds to whole nanoseconds."""
    return int(round(float(us) * 1000.0))


def _scope_path(args: dict[str, Any], module: str) -> list[str]:
    """Returns an op's HLO name stack as a list of scopes.

    The stack looks like ``jit(train_step)/jvp()/dot_general``: GPU ops carry
    it in ``name``, others in ``tf_op``. Without one, the op's HLO module
    stands in, spelled like a name stack entry (``jit_f`` -> ``jit(f)``).
    """
    for key in ("name", "tf_op"):
        value = str(args.get(key) or "")
        if "/" in value or value.startswith("jit("):
            return [segment for segment in value.split("/") if segment]
    if module.startswith(("jit_", "pjit_", "pmap_")):
        transform, _, function_name = module.partition("_")
        return [f"{transform}({function_name})"]
    return [module] if module else []


def _op_category(name: str, hlo_op: str) -> str:
    """Classifies a device op as ``comm``, ``memcpy`` or ``compute``."""
    if _COMM_RE.search(name) or _COMM_RE.search(hlo_op):
        return "comm"
    if _COPY_RE.search(name):
        return "memcpy"
    return "compute"


@dataclasses.dataclass
class _DeviceLines:
    """The device processes of one trace and the event lines read on each.

    Attributes:
        devices_by_pid: The device ordinal of each ``/device:*`` process id.
        thread_names_by_id: The name of each (pid, tid) event line.
        line_prefixes_by_pid: The name prefix of the lines read on each device
            process.
    """

    devices_by_pid: dict[Any, int]
    thread_names_by_id: dict[tuple[Any, Any], str]
    line_prefixes_by_pid: dict[Any, str]

    def is_read(self, pid: Any, tid: Any) -> bool:
        """Returns whether events on a device's (pid, tid) line are read."""
        thread_name = self.thread_names_by_id.get((pid, tid), "")
        return thread_name.startswith(self.line_prefixes_by_pid.get(pid, ""))


def _device_lines(events: list[dict[str, Any]]) -> _DeviceLines:
    """Finds a trace's device processes and the lines to read on each."""
    process_names_by_pid: dict[Any, str] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_names_by_pid[event.get("pid")] = str(
                (event.get("args") or {}).get("name", "")
            )
    devices_by_pid = {
        pid: int(device_match.group(1))
        for pid, name in process_names_by_pid.items()
        if (device_match := _DEVICE_RE.search(name))
    }
    thread_names_by_id: dict[tuple[Any, Any], str] = {
        (event.get("pid"), event.get("tid")): str(
            (event.get("args") or {}).get("name", "")
        )
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    # A device process may also carry XProf's derived lines (XLA Ops, XLA
    # Modules, Steps, ...) that repeat the raw stream events: read the raw GPU
    # streams, or on TPU the XLA Ops line.
    line_prefixes_by_pid: dict[Any, str] = {}
    for pid in devices_by_pid:
        line_names = [
            name
            for (line_pid, _), name in thread_names_by_id.items()
            if line_pid == pid
        ]
        if any(name.startswith("Stream #") for name in line_names):
            line_prefixes_by_pid[pid] = "Stream #"
        elif "XLA Ops" in line_names:
            line_prefixes_by_pid[pid] = "XLA Ops"
        else:
            line_prefixes_by_pid[pid] = ""
    return _DeviceLines(
        devices_by_pid, thread_names_by_id, line_prefixes_by_pid
    )


@dataclasses.dataclass
class _ShardEvents:
    """The events of one host's trace that its kernels and steps come from.

    Attributes:
        op_rows: One kernels-table row per device op, in trace order.
        op_details: The (HLO module, HLO op, correlation id) of each op row;
            the correlation id is -1 if unknown.
        annotated_steps: The (step_num, start_ns, end_ns) of each
            StepTraceAnnotation.
        launch_ns_by_correlation: The host launch time of each correlation id.
        api_calls: The (name, start_ns, duration_ns) of each host-side API
            call (a host event with a correlation id).
        executions: The (module, start_ns, end_ns) of each host-side
            execution of a jitted module.
    """

    op_rows: list[dict[str, object]] = dataclasses.field(default_factory=list)
    op_details: list[tuple[str, str, int]] = dataclasses.field(
        default_factory=list
    )
    annotated_steps: list[tuple[int, int, int]] = dataclasses.field(
        default_factory=list
    )
    launch_ns_by_correlation: dict[int, int] = dataclasses.field(
        default_factory=dict
    )
    api_calls: list[tuple[str, int, int]] = dataclasses.field(
        default_factory=list
    )
    executions: list[tuple[str, int, int]] = dataclasses.field(
        default_factory=list
    )


def _scan_events(events: list[dict[str, Any]], rank: int) -> _ShardEvents:
    """Splits one host's complete ("X") events into ops, steps and launches."""
    lines = _device_lines(events)
    scan = _ShardEvents()
    for event in events:
        if event.get("ph") != "X":
            continue
        duration_ns = _us_to_ns(event.get("dur", 0.0))
        if duration_ns <= 0:
            continue
        start_ns = _us_to_ns(event.get("ts", 0.0))
        end_ns = start_ns + duration_ns
        raw_args = event.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        pid = event.get("pid")
        correlation_id = (
            int(args["correlation_id"])
            if str(args.get("correlation_id", "")).isdigit()
            else -1
        )
        is_on_device = pid in lines.devices_by_pid
        if (
            "step_num" in args
            and str(args["step_num"]).lstrip("-").isdigit()
            and not is_on_device
        ):
            step_num = int(args["step_num"])
            scan.annotated_steps.append((step_num, start_ns, end_ns))
            continue
        if (
            not is_on_device
            and str(event.get("name", "")).startswith(_EXECUTE_PREFIX)
            and args.get("name")
        ):
            scan.executions.append((str(args["name"]), start_ns, end_ns))
        if not is_on_device and "hlo_op" not in args:
            if correlation_id >= 0:
                scan.launch_ns_by_correlation.setdefault(
                    correlation_id, start_ns
                )
                api_name = str(event.get("name", ""))
                scan.api_calls.append((api_name, start_ns, duration_ns))
            continue
        if is_on_device and not lines.is_read(pid, event.get("tid")):
            continue

        name = str(event.get("name", ""))
        hlo_op = str(args.get("hlo_op") or name)
        module = str(args.get("hlo_module") or args.get("module") or "")
        scope_path = _scope_path(args, module)
        transfer_details = str(
            args.get("memcpy_details") or args.get("Memset_details") or ""
        )
        bytes_match = _BYTES_RE.search(transfer_details)
        default_device = (
            int(args["device_ordinal"])
            if str(args.get("device_ordinal", "")).isdigit()
            else 0
        )
        scan.op_rows.append(
            {
                "name": name,
                "scope": scope_path[-1] if scope_path else "",
                "scope_path": " > ".join(scope_path),
                "category": _op_category(name, hlo_op),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
                "bytes": int(bytes_match.group(1)) if bytes_match else 0,
                "stream": int(event.get("tid", 0) or 0) % 2**31,
                "device": lines.devices_by_pid.get(pid, default_device),
                "rank": rank,
            }
        )
        scan.op_details.append(
            (module, str(args.get("hlo_op") or ""), correlation_id)
        )
    return scan


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


def _ingest_shard(
    events: list[dict[str, Any]],
    rank: int,
    source: str,
    runtime_by_name: dict[str, list[int]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds the kernels and steps tables of one host's trace.

    Args:
        events: The host's trace events.
        rank: The rank to give the host.
        source: The trace's file name, for the warning.
        runtime_by_name: Per host-side API call name, the [total duration ns,
            call count] of the calls made inside a step; updated in place.

    Returns:
        The kernels and steps frames.
    """
    runtime_by_name = {} if runtime_by_name is None else runtime_by_name
    scan = _scan_events(events, rank)
    if not scan.op_rows:
        return pl.DataFrame(schema=nsys_reader.KERNELS_SCHEMA), pl.DataFrame(
            schema=nsys_reader.STEPS_SCHEMA
        )

    order = np.argsort([row["start_ns"] for row in scan.op_rows], kind="stable")
    kernels = alignment.enrich_ops_df(
        pl.DataFrame(scan.op_rows)[order.tolist()], "name"
    )
    op_details = [scan.op_details[i] for i in order]
    starts_ns = kernels["start_ns"].to_numpy()
    ends_ns = kernels["end_ns"].to_numpy()
    # An op belongs to the step whose host code launched it; ops without a
    # known launch use their own start.
    launch_ns = np.array(
        [
            scan.launch_ns_by_correlation.get(correlation_id, int(starts_ns[i]))
            if correlation_id >= 0
            else int(starts_ns[i])
            for i, (_, _, correlation_id) in enumerate(op_details)
        ],
        dtype=np.int64,
    )

    if scan.annotated_steps:
        step_spans = sorted(set(scan.annotated_steps), key=lambda span: span[1])
    else:
        warnings.warn(
            f"{source}: no jax.profiler.StepTraceAnnotation step markers; "
            "each host-side execution of the busiest jitted module is "
            "treated as a step (mark steps with "
            "StepTraceAnnotation(name, step_num=i) for exact boundaries).",
            RuntimeWarning,
            stacklevel=3,
        )
        step_spans = _infer_module_steps(kernels, op_details, scan.executions)

    step_columns: dict[str, Any]
    if step_spans:
        step_of_op, step_columns = _assign_steps(launch_ns, step_spans)
    else:
        step_of_op = np.zeros(kernels.height, dtype=np.int32)
        first_start_ns, last_end_ns = int(starts_ns.min()), int(ends_ns.max())
        step_columns = {
            "step": [0],
            "start_ns": [first_start_ns],
            "end_ns": [last_end_ns],
            "duration_ns": [last_end_ns - first_start_ns],
        }
    steps = nsys_reader.conform(
        pl.DataFrame(step_columns).with_columns(pl.lit(rank).alias("rank")),
        nsys_reader.STEPS_SCHEMA,
    )
    kernels = kernels.with_columns(
        pl.Series("step", step_of_op, dtype=pl.Int32)
    )
    # Like the kernels, only the API calls made inside a step count.
    for api_name, call_start_ns, call_duration_ns in scan.api_calls:
        if not step_spans or any(
            span_start_ns <= call_start_ns <= span_end_ns
            for _, span_start_ns, span_end_ns in step_spans
        ):
            totals = runtime_by_name.setdefault(api_name, [0, 0])
            totals[0] += call_duration_ns
            totals[1] += 1
    if step_spans:
        # Compile/autotune work and transfers outside every step are not
        # step work.
        kernels = kernels.filter(pl.col("step") >= 0)
    return nsys_reader.conform(kernels, nsys_reader.KERNELS_SCHEMA), steps


def _infer_module_steps(
    kernels: pl.DataFrame,
    op_details: list[tuple[str, str, int]],
    executions: list[tuple[str, int, int]],
) -> list[tuple[int, int, int]]:
    """Infers steps from the host-side executions of the busiest module.

    The busiest jitted module is the one whose HLO ops take the most device
    time; each of its executions becomes a step, numbered in time order. Ops
    map to these steps by launch time.

    Args:
        kernels: The host's ops, in the order of ``op_details``.
        op_details: The (HLO module, HLO op, correlation id) of each op.
        executions: The (module, start_ns, end_ns) of each host-side
            execution of a jitted module.

    Returns:
        The (step, start_ns, end_ns) spans, sorted by start; empty if no
        execution of a module with HLO ops was recorded.
    """
    durations_ns = kernels["duration_ns"].to_numpy()
    device_ns_by_module: dict[str, int] = {}
    for (module, hlo_op, _), duration_ns in zip(
        op_details, durations_ns, strict=True
    ):
        if module and hlo_op:
            previous_ns = device_ns_by_module.get(module, 0)
            device_ns_by_module[module] = previous_ns + int(duration_ns)
    known_executions = [
        execution
        for execution in executions
        if execution[0] in device_ns_by_module
    ]
    if not known_executions:
        return []
    busiest_module = max(
        {execution[0] for execution in known_executions},
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
