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
       COALESCE(e.size, 0) AS size, COALESCE(e.recvSize, 0) AS recv_size,
       COALESCE(e.remoteRank, -1) AS peer
FROM ({calls}) e
JOIN StringIds s ON s.id = e.textId
LEFT JOIN (SELECT commUid, MAX(size) AS size FROM MPI_COMMUNICATORS
           GROUP BY commUid) c ON c.commUid = e.commUid
"""
# The peer of a point-to-point call is the remote rank in its communicator.
_MPI_CALLS = {
    "MPI_P2P_EVENTS": (
        'SELECT start, "end", textId, commUid, size, NULL AS recvSize, '
        "{remote_rank} AS remoteRank FROM MPI_P2P_EVENTS"
    ),
    "MPI_COLLECTIVES_EVENTS": (
        'SELECT start, "end", textId, commUid, size, recvSize, '
        "NULL AS remoteRank FROM MPI_COLLECTIVES_EVENTS"
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

    The message size follows nccl-tests' convention for the operation; see
    ``_message_bytes`` for how it is recovered from what nsys records.
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
    # Older exports have no remoteRank column.
    has_remote = any(
        row[1] == "remoteRank"
        for row in conn.execute("PRAGMA table_info(MPI_P2P_EVENTS)")
    )
    remote_rank = "remoteRank" if has_remote else "NULL"
    calls = [
        sql.format(remote_rank=remote_rank)
        for table, sql in _MPI_CALLS.items()
        if table in tables
    ]
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
            pl.Series("step", step_of(ops["start_ns"].to_numpy(), steps)),
        )
        .select(list(network.NETWORK_OPS_SCHEMA))
        .cast(network.NETWORK_OPS_SCHEMA)
    )


def _mpi_rows(mpi: pl.DataFrame) -> pl.DataFrame:
    """Gives MPI rows the shared columns."""
    kinds = {name: network.mpi_op(name) for name in mpi["name"].unique()}
    rows = mpi.with_columns(
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
    )
    return rows.select(
        pl.lit("mpi").alias("api"),
        "op",
        "posted",
        "group",
        "group_size",
        "peer",
        _message_bytes().alias("bytes"),
        "start_ns",
        "end_ns",
    ).pipe(_with_host_columns)


def _message_bytes() -> pl.Expr:
    """Recovers the message size nccl-tests would report from nsys's numbers.

    nsys records ``size`` and ``recvSize`` as the buffer times the number of
    ranks (one less for ``MPI_IN_PLACE``) for most collectives, and as the
    plain per-rank buffer for the others. Measured on Open MPI 5 with 2, 3 and
    4 ranks:

    - allgather, alltoall, reduce_scatter_block: ``size`` is already the
      total (nccl-tests' size);
    - allreduce: ``size`` is the buffer times the group size, so it is
      divided by it;
    - broadcast: ``recvSize`` is the buffer on every rank;
    - reduce: ``size`` is the buffer; at an in-place root it is missing and
      ``recvSize`` is the buffer times (ranks - 1);
    - gather, scatter: the chunk each rank holds, times the group size.

    An ``MPI_IN_PLACE`` allreduce, allgather or alltoall cannot be told from
    a plain one, so its size is read as a plain call's. Other functions, and
    a communicator of unknown size, take the larger of the two numbers.
    """
    op, ranks = pl.col("op"), pl.col("group_size")
    send, recv = pl.col("size"), pl.col("recv_size")
    largest = pl.max_horizontal(send, recv)
    return (
        pl.when(ranks < 2)
        .then(largest)
        .when(op == "allreduce")
        .then(largest // ranks)
        .when(op == "broadcast")
        .then(recv)
        .when(op == "reduce")
        .then(pl.when(send > 0).then(send).otherwise(recv // (ranks - 1)))
        .when(op == "scatter")
        .then(recv * ranks)
        .when(op == "gather")
        .then(send * ranks)
        .otherwise(largest)
    )


def _ucx_rows(ucx: pl.DataFrame) -> pl.DataFrame:
    """Gives UCX submit rows the shared columns."""
    ops = {name: network.ucx_op(name) for name in ucx["name"].unique()}
    return ucx.select(
        pl.lit("ucx").alias("api"),
        pl.col("name").replace_strict(ops, return_dtype=pl.String).alias("op"),
        pl.lit(True).alias("posted"),
        pl.lit("").alias("group"),
        pl.lit(0).alias("group_size"),
        pl.lit(-1).alias("peer"),
        pl.col("size").alias("bytes"),
        "start_ns",
        "end_ns",
    ).pipe(_with_host_columns)


def _with_host_columns(rows: pl.DataFrame) -> pl.DataFrame:
    """Adds what host calls lack: no device, no stream, no algorithm."""
    return rows.with_columns(
        (pl.col("end_ns") - pl.col("start_ns")).alias("duration_ns"),
        pl.lit(-1).alias("launch_ns"),
        pl.lit(-1).alias("device"),
        pl.lit(-1).alias("stream"),
        pl.lit("").alias("algo"),
        pl.lit("").alias("proto"),
        pl.lit("").alias("dtype"),
    )


def step_of(start_ns: np.ndarray, steps: pl.DataFrame) -> np.ndarray:
    """Returns the step each start time falls in, or -1.

    Args:
        start_ns: Start times.
        steps: Step ranges with ``step``, ``start_ns`` and ``end_ns``.

    Returns:
        One step number per start time; -1 outside every step.
    """
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
