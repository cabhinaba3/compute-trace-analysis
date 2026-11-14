"""Network-API helpers for the trace adapters and the ``network_api`` aspect.

A ``network_ops`` table has one row per communication call (an NCCL kernel, a
GPU-to-GPU copy, an MPI or UCX call), in the same shape whichever API made
it, so calls of different APIs on the same workload can be matched by logical
operation and message size.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

NETWORK_OPS_SCHEMA = pl.Schema(
    {
        "rank": pl.Int32(),
        "device": pl.Int32(),
        "stream": pl.Int32(),
        "step": pl.Int32(),
        # Communication library that issued the call, e.g. "nccl".
        "api": pl.String(),
        # Logical operation, lower case without separators: "allreduce".
        "op": pl.String(),
        # Algorithm and protocol NCCL picked ("RING", "LL"); "" if unknown.
        "algo": pl.String(),
        "proto": pl.String(),
        "dtype": pl.String(),
        # Process group the call ran in; "" if unknown.
        "group": pl.String(),
        "group_size": pl.Int32(),
        # Message size in bytes as nccl-tests counts it; 0 when the trace does
        # not record it.
        "bytes": pl.Int64(),
        # True when the call only posts the transfer (MPI_Isend, a UCX
        # submit): its duration says nothing about bandwidth.
        "posted": pl.Boolean(),
        "start_ns": pl.Int64(),
        "end_ns": pl.Int64(),
        "duration_ns": pl.Int64(),
    }
)

# Collective name as PyTorch or a kernel spells it -> logical operation.
_OP_ALIASES = {
    "allgatherbase": "allgather",
    "allgathercoalesced": "allgather",
    "allgatherintotensor": "allgather",
    "reducescatterbase": "reducescatter",
    "reducescattertensor": "reducescatter",
    "alltoallbase": "alltoall",
    "allreducecoalesced": "allreduce",
    "sendrecv": "sendrecv",
}

# Bus-bandwidth factors of nccl-tests: busbw = algbw * factor(ranks).
_BUSBW_FACTOR = {
    "allreduce": lambda n: 2.0 * (n - 1) / n,
    "allgather": lambda n: (n - 1) / n,
    "reducescatter": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
}

DTYPE_BYTES = {
    "float": 4, "float32": 4, "double": 8, "float64": 8, "half": 2,
    "float16": 2, "bfloat16": 2, "long": 8, "int64": 8, "int": 4,
    "int32": 4, "short": 2, "char": 1, "byte": 1, "bool": 1,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}  # fmt: skip

# Upper edges of the message-size buckets, in bytes.
SIZE_BUCKET_EDGES = (1 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20)
SIZE_BUCKET_LABELS = (
    "<1KB", "1-64KB", "64KB-1MB", "1-16MB", "16-256MB", ">256MB",
)  # fmt: skip

_KERNEL_RE = (
    r"nccl(?:Dev)?Kernel_(?<op>[A-Za-z]+)"
    r"(?:_(?:Sum|Prod|Max|Min|PreMulSum|SumPostDiv))?"
    r"(?:_(?:f|bf|u|i)[0-9][a-z0-9]*)?"
    r"(?:_(?<algo>RING|TREE|COLLNET_DIRECT|COLLNET_CHAIN|NVLS_TREE|NVLS|PAT))?"
    r"(?:_(?<proto>LL128|LL|SIMPLE))?"
)


def normalize_op(name: pl.Expr) -> pl.Expr:
    """Maps a collective name to its logical operation.

    Args:
        name: A string expression such as ``all_gather`` or ``AllReduce``.

    Returns:
        The lower-cased name without underscores, with PyTorch's variants
        (``_allgather_base``, ``alltoall_base``) folded onto the base one.
    """
    base = name.str.to_lowercase().str.replace_all("_", "")
    return base.replace(_OP_ALIASES)


def bucket_bytes(size: np.ndarray) -> np.ndarray:
    """Maps message sizes in bytes to bucket labels; 0 bytes is ``unknown``."""
    labels = np.array([*SIZE_BUCKET_LABELS, "unknown"], dtype=object)
    index = np.searchsorted(SIZE_BUCKET_EDGES, size, side="right")
    return labels[np.where(size <= 0, len(SIZE_BUCKET_LABELS), index)]


def busbw_factor(op: str, ranks: int) -> float:
    """Returns the nccl-tests bus-bandwidth factor of an operation.

    Args:
        op: Logical operation.
        ranks: Number of ranks in the group.

    Returns:
        2(n-1)/n for allreduce, (n-1)/n for allgather, reducescatter and
        alltoall, 1 for everything else (including one rank).
    """
    if ranks < 2 or op not in _BUSBW_FACTOR:
        return 1.0
    return _BUSBW_FACTOR[op](ranks)


def ops_from_kernels(
    kernels: pl.DataFrame, collective: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Builds the NCCL rows of a ``network_ops`` table from a kernels table.

    Args:
        kernels: A kernels table; rows of category ``comm`` become calls.
        collective: Optional collective metadata of PyTorch traces: ``row``
            (index into ``kernels``), ``collective``, ``dtype``, ``group``,
            ``in_elems``, ``out_elems`` and ``group_size``.

    Returns:
        A frame conforming to ``NETWORK_OPS_SCHEMA``.
    """
    rows = kernels.with_row_index("row").filter(pl.col("category") == "comm")
    rows = rows.with_columns(
        pl.col("name").str.extract_groups(_KERNEL_RE).alias("parsed")
    ).unnest("parsed")
    if collective is None:
        collective = pl.DataFrame(
            schema={
                "row": pl.UInt32(),
                "collective": pl.String(),
                "dtype": pl.String(),
                "group": pl.String(),
                "in_elems": pl.Int64(),
                "out_elems": pl.Int64(),
                "group_size": pl.Int64(),
            }
        )
    rows = rows.join(
        collective.cast({"row": pl.UInt32()}), on="row", how="left"
    )
    # PyTorch's name wins; the kernel name covers hand-written NCCL and
    # kernels that only say "Generic".
    op = normalize_op(
        pl.coalesce(
            pl.col("collective"),
            pl.when(pl.col("op") != "Generic").then(pl.col("op")),
            pl.lit("other"),
        )
    )
    elems = (
        pl.when(op == "allgather")
        .then(pl.col("out_elems"))
        .otherwise(pl.col("in_elems"))
    )
    width = (
        pl.col("dtype")
        .str.to_lowercase()
        .replace_strict(DTYPE_BYTES, default=0)
    )
    nccl = rows.select(
        "rank",
        "device",
        "stream",
        "step",
        pl.lit("nccl").alias("api"),
        op.alias("op"),
        pl.col("algo").fill_null(""),
        pl.col("proto").fill_null(""),
        pl.col("dtype").fill_null(""),
        pl.col("group").fill_null(""),
        pl.col("group_size").fill_null(0),
        (elems.fill_null(0) * width).alias("bytes"),
        pl.lit(False).alias("posted"),
        "start_ns",
        "end_ns",
        "duration_ns",
    ).cast(NETWORK_OPS_SCHEMA)
    return pl.concat([nccl, peer_copies(kernels)])


def peer_copies(kernels: pl.DataFrame) -> pl.DataFrame:
    """Returns the GPU-to-GPU copies of a kernels table as network calls.

    Args:
        kernels: A kernels table; memcpys named ``P2P``, ``PtoP`` or
            ``peer`` are taken.

    Returns:
        A ``network_ops`` frame with api ``p2p`` and op ``copy``.
    """
    copies = kernels.filter(
        pl.col("category") == "memcpy",
        pl.col("name").str.contains(r"(?i)p2p|ptop|peer"),
    )
    return copies.select(
        "rank",
        "device",
        "stream",
        "step",
        pl.lit("p2p").alias("api"),
        pl.lit("copy").alias("op"),
        pl.lit("").alias("algo"),
        pl.lit("").alias("proto"),
        pl.lit("").alias("dtype"),
        pl.lit("").alias("group"),
        pl.lit(2).alias("group_size"),
        "bytes",
        pl.lit(False).alias("posted"),
        "start_ns",
        "end_ns",
        "duration_ns",
    ).cast(NETWORK_OPS_SCHEMA)


_MPI_OPS = {
    "bcast": "broadcast",
    "allgatherv": "allgather",
    "alltoallv": "alltoall",
    "alltoallw": "alltoall",
    "reduce_scatter_block": "reducescatter",
    "reduce_scatter": "reducescatter",
    "ssend": "send",
    "bsend": "send",
    "rsend": "send",
}
_MPI_POSTS = re.compile(
    r"^I(send|recv|bsend|ssend|rsend|allreduce|bcast|allgatherv?|alltoall[vw]?"
    r"|reduce|reduce_scatter(_block)?|gather|scatter|barrier)$"
)


def mpi_op(name: str) -> tuple[str, bool]:
    """Maps an MPI function name to its logical operation.

    Args:
        name: For example ``MPI_Allreduce`` or ``MPI_Isend``.

    Returns:
        The logical operation (``MPI_Bcast`` is ``broadcast``, ``MPI_Isend``
        is ``send``) and whether the function only posts the transfer.
    """
    base = name.removeprefix("MPI_")
    posted = bool(_MPI_POSTS.match(base))
    base = base[1:] if posted else base
    lowered = base.lower()
    return _MPI_OPS.get(lowered, lowered.replace("_", "")), posted


def ucx_op(name: str) -> str:
    """Maps a UCP function name such as ``ucp_tag_send_nbx`` to an operation."""
    for op in ("send", "recv", "put", "get"):
        if op in name:
            return op
    return "other"
