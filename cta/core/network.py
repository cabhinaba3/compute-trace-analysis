"""Network-API helpers for the trace adapters and the ``network_api`` aspect.

A ``network_ops`` table has one row per communication call as it ran on a GPU,
in the same shape whichever API made it, so calls of different APIs on the
same workload can be matched by logical operation and message size.
"""

from __future__ import annotations

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
    return rows.select(
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
        "start_ns",
        "end_ns",
        "duration_ns",
    ).cast(NETWORK_OPS_SCHEMA)
