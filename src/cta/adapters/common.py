"""File helpers shared by the trace adapters.

Sniffing the start of a (possibly gzip-compressed) file, writing an adapter's
tables as Parquet (single-shot or bounded-memory streaming), converting native
streaming graph chunks to Polars frames, and reading the rank out of a
per-rank file name.
"""

from __future__ import annotations

import gzip
import pathlib
import re
from typing import Any
import zlib

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

_RANK_RE = re.compile(r"(?:rank[_.-]?|\.r?)(\d+)(?:\.[^.]+)+$", re.IGNORECASE)

# Canonical native streaming OpClass and EdgeKind codes (streaming/batch.hpp).
STREAM_OP_CLASSES = {
    0: "other",
    1: "memory",
    2: "compute",
    3: "comm_send",
    4: "comm_recv",
    5: "comm_coll",
    6: "storage",
    7: "data_op",
}
STREAM_EDGE_KINDS = {
    0: "data",
    1: "ctrl",
    2: "requires",
    3: "irequires",
    4: "p2p",
    5: "coll_barrier",
}

_RING_COLLECTIVES = frozenset(
    (
        "ALL_GATHER",
        "REDUCE_SCATTER",
        "REDUCE_SCATTER_BLOCK",
        "BROADCAST",
        "REDUCE",
    )
)


def peek_text(path: pathlib.Path, max_bytes: int = 8192) -> str:
    """Returns the first decompressed text bytes of a file, or '' on error."""
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as gz:
                return gz.read(max_bytes).decode("utf-8", errors="ignore")
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="ignore")
    except (OSError, EOFError, zlib.error):
        return ""


def write_tables(
    out_dir: pathlib.Path, tables: dict[str, pl.DataFrame]
) -> dict[str, pathlib.Path]:
    """Writes tables as compressed Parquet files in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_paths: dict[str, pathlib.Path] = {}
    for table_name, table in tables.items():
        table_path = out_dir / f"{table_name}.parquet"
        table.write_parquet(table_path, compression="zstd")
        table_paths[table_name] = table_path
    return table_paths


def infer_rank(path: pathlib.Path, fallback: int) -> int:
    """Returns the rank encoded in a per-rank trace file name.

    Recognises names such as ``..._rank3.sqlite`` or ``trace.r3.json``.

    Args:
        path: The trace file.
        fallback: Rank to return when the name encodes none.

    Returns:
        The encoded rank, else ``fallback``.
    """
    match = _RANK_RE.search(path.name)
    return int(match.group(1)) if match else fallback


def strings_from_ends(ends: np.ndarray, byte_buf: np.ndarray) -> pl.Series:
    """Decodes flat UTF-8 string bytes with end offsets into a Polars Series."""
    if len(ends) == 0:
        return pl.Series([], dtype=pl.String)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), ends.astype(np.int64))
    )
    array = pa.Array.from_buffers(
        pa.large_string(),
        len(ends),
        [None, pa.py_buffer(offsets), pa.py_buffer(byte_buf)],
    )
    return pl.Series(pl.from_arrow(array))


def chunk_to_graph_frames(
    chunk: dict[str, Any],
    nodes_schema: pl.Schema,
    edges_schema: pl.Schema,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Converts one native ``GraphStream`` chunk dict into node/edge frames."""
    op_codes = chunk["op_class"]
    if len(op_codes) > 0:
        nodes = pl.DataFrame(
            {
                "rank": chunk["rank"],
                "node_id": chunk["node_id"],
                "name": strings_from_ends(
                    chunk["name_end"], chunk["name_bytes"]
                ),
                "op_class": pl.Series(op_codes).replace_strict(
                    STREAM_OP_CLASSES, return_dtype=pl.String
                ),
                "comm_type": strings_from_ends(
                    chunk["comm_type_end"], chunk["comm_type_bytes"]
                ),
                "duration_ns": chunk["duration_ns"],
                "size_bytes": chunk["size_bytes"],
                "src_rank": chunk["src_rank"],
                "dst_rank": chunk["dst_rank"],
                "message_tag": chunk["message_tag"],
                "num_ops": chunk["num_ops"],
            },
            schema=nodes_schema,
        )
    else:
        nodes = pl.DataFrame(schema=nodes_schema)

    edge_kinds = chunk["edge_kind"]
    if len(edge_kinds) > 0:
        edges = pl.DataFrame(
            {
                "rank": chunk["edge_rank"],
                "src_id": chunk["edge_src"],
                "dst_id": chunk["edge_dst"],
                "dep_kind": pl.Series(edge_kinds).replace_strict(
                    STREAM_EDGE_KINDS, return_dtype=pl.String
                ),
            },
            schema=edges_schema,
        )
    else:
        edges = pl.DataFrame(schema=edges_schema)
    return nodes, edges


class StreamingParquetWriter:
    """Writes Polars DataFrame chunks to a zstd Parquet file in bounded RAM."""

    def __init__(self, path: pathlib.Path, schema: pl.Schema) -> None:
        """Initialises the writer for ``path`` with Polars ``schema``."""
        self.path = path
        self._schema = schema
        self._arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema
        self._writer: pq.ParquetWriter | None = None
        self._rows_written = 0

    def write_frame(self, frame: pl.DataFrame) -> None:
        """Appends one non-empty Polars frame as a Parquet row group."""
        if frame.is_empty():
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                str(self.path), self._arrow_schema, compression="zstd"
            )
        self._writer.write_table(frame.to_arrow())
        self._rows_written += frame.height

    def close(self) -> pathlib.Path:
        """Finalises the file, writing an empty table if no rows arrived."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(schema=self._schema).write_parquet(
                self.path, compression="zstd"
            )
        return self.path

    def __enter__(self) -> StreamingParquetWriter:
        """Enters the context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Closes the underlying Parquet writer on exit."""
        self.close()


class OnlineTrafficAccumulator:
    """Accumulates P2P and collective traffic across chunks in O(P^2) memory."""

    def __init__(self, *, include_collectives: bool) -> None:
        """Initialises empty P2P and collective traffic maps."""
        self._include_collectives = include_collectives
        self._p2p: dict[tuple[int, int], list[int]] = {}
        self._colls: dict[tuple[int, str, int], int] = {}
        self._max_rank = -1
        self._has_nodes = False

    def update(self, nodes: pl.DataFrame) -> None:
        """Incorporates one chunk of ``nodes`` rows."""
        if nodes.is_empty():
            return
        self._has_nodes = True
        chunk_max = int(nodes["rank"].max())  # type: ignore[arg-type]
        if chunk_max > self._max_rank:
            self._max_rank = chunk_max

        sends = nodes.filter(
            (pl.col("op_class") == "comm_send") & (pl.col("dst_rank") >= 0)
        )
        if not sends.is_empty():
            grouped = sends.group_by(
                [
                    pl.col("rank").alias("src_rank"),
                    pl.col("dst_rank").cast(pl.UInt32),
                ]
            ).agg(
                pl.len().cast(pl.UInt64).alias("message_count"),
                pl.col("size_bytes").sum().cast(pl.UInt64).alias("total_bytes"),
            )
            for src_r, dst_r, cnt, nbytes in grouped.iter_rows():
                key = (int(src_r), int(dst_r))
                slot = self._p2p.get(key)
                if slot is None:
                    self._p2p[key] = [int(cnt), int(nbytes)]
                else:
                    slot[0] += int(cnt)
                    slot[1] += int(nbytes)

        if self._include_collectives:
            colls = nodes.filter(
                (pl.col("op_class") == "comm_coll") & (pl.col("size_bytes") > 0)
            )
            if not colls.is_empty():
                grouped = colls.group_by(
                    "rank", "comm_type", "size_bytes"
                ).len()
                for rank_v, ctype, sz, cnt in grouped.iter_rows():
                    ckey = (int(rank_v), str(ctype), int(sz))
                    self._colls[ckey] = self._colls.get(ckey, 0) + int(cnt)

    def finish(
        self, traffic_schema: pl.Schema, min_world_size: int = 1
    ) -> pl.DataFrame:
        """Folds collectives using final world size and returns traffic."""
        if not self._has_nodes:
            return pl.DataFrame(schema=traffic_schema)

        world_size = max(min_world_size, self._max_rank + 1, 1)
        merged: dict[tuple[int, int], list[int]] = {
            k: [v[0], v[1]] for k, v in self._p2p.items()
        }

        def _add(src_r: int, dst_r: int, msgs: int, nbytes: int) -> None:
            slot = merged.get((src_r, dst_r))
            if slot is None:
                merged[(src_r, dst_r)] = [msgs, nbytes]
            else:
                slot[0] += msgs
                slot[1] += nbytes

        if self._include_collectives and world_size > 1:
            for (rank_v, ctype, size), count in self._colls.items():
                next_rank = (rank_v + 1) % world_size
                if ctype == "ALL_REDUCE":
                    _add(
                        rank_v,
                        next_rank,
                        count * (2 * (world_size - 1)),
                        count * ((size * (2 * (world_size - 1))) // world_size),
                    )
                elif ctype in _RING_COLLECTIVES:
                    _add(
                        rank_v,
                        next_rank,
                        count * (world_size - 1),
                        count * ((size * (world_size - 1)) // world_size),
                    )
                elif ctype == "ALL_TO_ALL":
                    per_peer = count * (size // world_size)
                    for dst_r in range(world_size):
                        if dst_r != rank_v:
                            _add(rank_v, dst_r, count, per_peer)

        if not merged:
            return pl.DataFrame(schema=traffic_schema)

        keys = sorted(merged)
        return pl.DataFrame(
            {
                "src_rank": [k[0] for k in keys],
                "dst_rank": [k[1] for k in keys],
                "message_count": [merged[k][0] for k in keys],
                "total_bytes": [merged[k][1] for k in keys],
            },
            schema=traffic_schema,
        )
