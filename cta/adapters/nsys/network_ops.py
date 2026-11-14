"""MPI and UCX calls of an Nsight Systems export as ``network_ops`` rows."""

from __future__ import annotations

import sqlite3

import numpy as np
import polars as pl

from cta.core import network

# Calls come from MPI_* tables (with the communicator's size) and UCP submits.
_MPI_QUERY = """
SELECT e.start AS start_ns, e."end" AS end_ns, s.value AS name,
       CAST(e.commUid AS TEXT) AS "group", COALESCE(c.size, 0) AS group_size,
       COALESCE(e.size, 0) AS size, COALESCE(e.recvSize, 0) AS recv_size
FROM ({calls}) e
JOIN StringIds s ON s.id = e.textId
LEFT JOIN (SELECT commUid, MAX(size) AS size FROM MPI_COMMUNICATORS
           GROUP BY commUid) c ON c.commUid = e.commUid
"""
_MPI_CALLS = {
    "MPI_P2P_EVENTS": (
        'SELECT start, "end", textId, commUid, size, NULL AS recvSize '
        "FROM MPI_P2P_EVENTS"
    ),
    "MPI_COLLECTIVES_EVENTS": (
        'SELECT start, "end", textId, commUid, size, recvSize '
        "FROM MPI_COLLECTIVES_EVENTS"
    ),
}
_UCX_QUERY = """
SELECT e.start AS start_ns, e."end" AS end_ns, s.value AS name,
       COALESCE(e.packedSize, 0) AS size
FROM UCP_SUBMIT_EVENTS e JOIN StringIds s ON s.id = e.textId
"""


def host_network_ops(
    conn: sqlite3.Connection,
    tables: set[str],
    rank: int,
    steps: pl.DataFrame,
) -> pl.DataFrame:
    """Reads the MPI and UCX calls of one rank's export.

    The message size is the larger of the send and receive size, which is the
    root's buffer for a broadcast and the full buffer for an allgather.
    MPI functions that only post a transfer (``MPI_Isend``) and every UCX
    submit are marked ``posted``: their duration is not a transfer time.

    Args:
        conn: A connection to the SQLite export.
        tables: The table names of the export.
        rank: The rank written to every row.
        steps: The rank's step ranges (``step``, ``start_ns``, ``end_ns``),
            used to number the calls; calls outside every step get -1.

    Returns:
        A frame conforming to ``network.NETWORK_OPS_SCHEMA``; empty when the
        export has no MPI or UCX tables.
    """
    frames = []
    calls = [sql for table, sql in _MPI_CALLS.items() if table in tables]
    if calls and "MPI_COMMUNICATORS" in tables:
        mpi = pl.read_database(
            _MPI_QUERY.format(calls=" UNION ALL ".join(calls)), conn
        )
        frames.append(_mpi_rows(mpi))
    if "UCP_SUBMIT_EVENTS" in tables:
        frames.append(_ucx_rows(pl.read_database(_UCX_QUERY, conn)))
    if not frames:
        return pl.DataFrame(schema=network.NETWORK_OPS_SCHEMA)
    ops = pl.concat(frames, how="vertical_relaxed")
    return (
        ops.with_columns(
            pl.lit(rank, dtype=pl.Int32).alias("rank"),
            pl.Series("step", _step_of(ops["start_ns"].to_numpy(), steps)),
        )
        .select(list(network.NETWORK_OPS_SCHEMA))
        .cast(network.NETWORK_OPS_SCHEMA)
    )


def _mpi_rows(mpi: pl.DataFrame) -> pl.DataFrame:
    """Gives MPI rows the shared columns."""
    kinds = {name: network.mpi_op(name) for name in mpi["name"].unique()}
    return mpi.select(
        pl.lit("mpi").alias("api"),
        pl.col("name")
        .replace_strict(
            {name: op for name, (op, _) in kinds.items()},
            return_dtype=pl.String,
        )
        .alias("op"),
        pl.col("name")
        .replace_strict(
            {name: posted for name, (_, posted) in kinds.items()},
            return_dtype=pl.Boolean,
        )
        .alias("posted"),
        "group",
        "group_size",
        pl.max_horizontal("size", "recv_size").alias("bytes"),
        "start_ns",
        "end_ns",
    ).pipe(_with_host_columns)


def _ucx_rows(ucx: pl.DataFrame) -> pl.DataFrame:
    """Gives UCX submit rows the shared columns."""
    ops = {name: network.ucx_op(name) for name in ucx["name"].unique()}
    return ucx.select(
        pl.lit("ucx").alias("api"),
        pl.col("name").replace_strict(ops, return_dtype=pl.String).alias("op"),
        pl.lit(True).alias("posted"),
        pl.lit("").alias("group"),
        pl.lit(0).alias("group_size"),
        pl.col("size").alias("bytes"),
        "start_ns",
        "end_ns",
    ).pipe(_with_host_columns)


def _with_host_columns(rows: pl.DataFrame) -> pl.DataFrame:
    """Adds what host calls lack: no device, no stream, no algorithm."""
    return rows.with_columns(
        (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"),
        pl.lit(-1).alias("device"),
        pl.lit(-1).alias("stream"),
        pl.lit("").alias("algo"),
        pl.lit("").alias("proto"),
        pl.lit("").alias("dtype"),
    )


def _step_of(start_ns: np.ndarray, steps: pl.DataFrame) -> np.ndarray:
    """Returns the step each start time falls in, or -1."""
    if steps.is_empty():
        return np.full(len(start_ns), -1, dtype=np.int32)
    ordered = steps.sort("start_ns")
    begin = ordered["start_ns"].to_numpy()
    index = np.searchsorted(begin, start_ns, side="right") - 1
    inside = (index >= 0) & (
        start_ns <= ordered["end_ns"].to_numpy()[np.maximum(index, 0)]
    )
    return np.where(
        inside, ordered["step"].to_numpy()[np.maximum(index, 0)], -1
    ).astype(np.int32)
