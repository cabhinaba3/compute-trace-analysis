"""File helpers shared by the trace adapters.

Sniffing the start of a (possibly gzip-compressed) file, writing an adapter's
tables as Parquet, and reading the rank out of a per-rank file name.
"""

from __future__ import annotations

import gzip
import pathlib
import re
from typing import TYPE_CHECKING
import zlib

if TYPE_CHECKING:
    import polars as pl

_RANK_RE = re.compile(r"(?:rank[_.-]?|\.r?)(\d+)(?:\.[^.]+)+$", re.IGNORECASE)


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
