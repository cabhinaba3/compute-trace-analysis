"""PyTorch Profiler / Kineto Chrome Trace (.json and .json.gz) adapter.

Every file of an artifact is one rank's trace with its own clock. GPU kernels,
memcpys and memsets are attributed to the CPU op and ``ProfilerStep#N`` range
that launched them, matched through the CUDA runtime call's correlation id.
"""

from __future__ import annotations

from collections.abc import Sequence
import gzip
import pathlib
import re
from typing import Any

import numpy as np
import polars as pl

from cta import _native as native
from cta.adapters.nsys import reader as nsys_reader
from cta.core import alignment
from cta.core import bundle
from cta.core import intervals
from cta.core import models
from cta.core import registry

_XPROF_HEAD_RE = re.compile(r'\{\s*"displayTimeUnit"\s*:\s*"ns"')


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
            columns = native.read_kineto_trace(str(path))
            rank = _shard_rank(columns, path, shard_index)
            shard_kernels, shard_steps = _ingest_shard(
                columns, rank, runtime_totals
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


_CATEGORY_NAMES = {0: "compute", 1: "comm", 2: "memcpy"}


class _CodedNames:
    """Names looked up through a code array, without building a list of them."""

    def __init__(self, names: Sequence[str], codes: np.ndarray) -> None:
        self._names = names
        self._codes = codes

    def __len__(self) -> int:
        return len(self._codes)

    def __getitem__(self, index: int) -> str:
        return self._names[int(self._codes[index])]


def _shard_rank(columns: dict[str, Any], path: pathlib.Path, index: int) -> int:
    """Returns the rank of a shard.

    The rank is the trace's own ``distributedInfo.rank``, else a rank in the
    file name, else the shard's index.
    """
    if columns["rank"] is not None:
        return int(columns["rank"])
    return bundle.infer_rank(path, index)


def _ingest_shard(
    columns: dict[str, Any], rank: int, runtime_totals: dict[str, list[int]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds the kernels and steps tables of one rank's trace.

    All timestamps of a shard share one clock. CUDA runtime call durations
    and counts are added to ``runtime_totals`` as a side effect.

    Args:
        columns: The shard as read by ``native.read_kineto_trace``.
        rank: The rank written to every row.
        runtime_totals: Runtime API name -> [total duration in ns, calls];
            updated in place.

    Returns:
        The kernels and steps tables, conformed to the nsys schemas.
    """
    names: list[str] = columns["names"]
    for code, duration_ns, calls in zip(
        columns["runtime_name"].tolist(),
        columns["runtime_duration"].tolist(),
        columns["runtime_calls"].tolist(),
        strict=True,
    ):
        totals = runtime_totals.setdefault(names[code], [0, 0])
        totals[0] += duration_ns
        totals[1] += calls

    if not len(columns["gpu_name"]):
        return (
            pl.DataFrame(schema=nsys_reader.KERNELS_SCHEMA),
            pl.DataFrame(schema=nsys_reader.STEPS_SCHEMA),
        )

    # Sort by GPU start time; the launch arrays are permuted alongside so
    # they stay aligned with the sorted rows.
    order = np.argsort(columns["gpu_start"], kind="stable")
    codes = columns["gpu_name"][order]
    gpu_start_ns = columns["gpu_start"][order]
    gpu_end_ns = columns["gpu_end"][order]
    launch_ns = columns["gpu_launch_ns"][order]
    launch_thread = columns["gpu_launch_thread"][order]
    row_count = len(codes)

    # Demangling and classifying are per name, so they run on the distinct
    # kernel names and are spread over the rows afterwards.
    used_codes, name_rows = np.unique(codes, return_inverse=True)
    distinct = pl.DataFrame(
        {"name": pl.Series([names[code] for code in used_codes.tolist()])}
    )
    distinct = alignment.enrich_ops_df(distinct, "name")
    kernels = pl.DataFrame(
        {
            "name": distinct["name"].gather(name_rows),
            "scope": pl.repeat("", row_count, dtype=pl.String, eager=True),
            "category": pl.Series(
                columns["gpu_category"][order]
            ).replace_strict(_CATEGORY_NAMES, return_dtype=pl.String),
            "start_ns": gpu_start_ns,
            "end_ns": gpu_end_ns,
            "duration_ns": gpu_end_ns - gpu_start_ns,
            "bytes": columns["gpu_bytes"][order],
            "stream": columns["gpu_stream"][order],
            "device": columns["gpu_device"][order],
            "rank": np.full(row_count, rank, dtype=np.int64),
        }
    ).with_columns(
        distinct["demangled_name"].gather(name_rows),
        distinct["canonical_name"].gather(name_rows),
        distinct["op_family"].gather(name_rows),
    )

    # Scope = the innermost CPU op that encloses the launch *on the launching
    # thread* (a finished sibling that started later does not hide the
    # enclosing op). Kernels without a known launch get no scope.
    op_count = len(columns["op_name"])
    if op_count:
        op_thread = columns["op_thread"]
        by_thread = np.argsort(op_thread, kind="stable")
        thread_ids, first = np.unique(op_thread[by_thread], return_index=True)
        ranges_by_thread = {}
        for thread_id, members in zip(
            thread_ids.tolist(), np.split(by_thread, first[1:]), strict=True
        ):
            ranges_by_thread[thread_id] = intervals.build_thread_ranges_arrays(
                columns["op_start"][members],
                columns["op_end"][members],
                members,
            )
        op_ids = intervals.assign_ranges(
            ranges_by_thread,
            columns["op_start"],
            launch_ns,
            launch_thread,
            gpu_start_ns,
        )
        # Range id -1 (no enclosing op) indexes the trailing "".
        names_with_blank = np.array([*names, ""], dtype=object)
        op_codes = np.append(columns["op_name"], -1)
        has_launch = launch_thread >= 0
        scopes = np.where(has_launch, names_with_blank[op_codes[op_ids]], "")
        scope_paths = np.where(
            has_launch,
            intervals.range_paths(
                ranges_by_thread,
                op_ids,
                _CodedNames(names, columns["op_name"]),
            ),
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
    if len(columns["step_id"]):
        step_order = np.argsort(columns["step_start"], kind="stable")
        step_ids = columns["step_id"][step_order].astype(np.int32)
        step_start_ns = columns["step_start"][step_order]
        step_end_ns = columns["step_end"][step_order]
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
        kernel_steps = np.zeros(row_count, dtype=np.int32)
        trace_start_ns = int(gpu_start_ns.min())
        trace_end_ns = int(gpu_end_ns.max())
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
