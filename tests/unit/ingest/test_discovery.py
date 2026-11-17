"""Unit tests for trace bundle resolution, file sniffing, and table writing."""

from __future__ import annotations

import gzip
import pathlib
import zlib

import polars as pl
import pytest

from cta.adapters import common
from cta.adapters.jax import reader as jax_reader
from cta.adapters.kineto import reader as kineto_reader
from cta.ingest import discovery


def test_peek_text_plain_text(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello trace analysis world", encoding="utf-8")
    assert common.peek_text(path, max_bytes=5) == "hello"
    assert common.peek_text(path, max_bytes=100) == "hello trace analysis world"


def test_peek_text_valid_gz(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as gz:
        gz.write('{"traceEvents": [{"name": "kernel"}]}')
    assert common.peek_text(path, max_bytes=14) == '{"traceEvents"'
    assert '{"traceEvents": [{"name": "kernel"}]}' in common.peek_text(path)


def test_peek_text_nonexistent_and_empty(tmp_path: pathlib.Path) -> None:
    nonexistent = tmp_path / "does_not_exist.json.gz"
    assert common.peek_text(nonexistent) == ""

    empty_txt = tmp_path / "empty.txt"
    empty_txt.touch()
    assert common.peek_text(empty_txt) == ""

    empty_gz = tmp_path / "empty.json.gz"
    empty_gz.touch()
    assert common.peek_text(empty_gz) == ""


def test_peek_text_directory(tmp_path: pathlib.Path) -> None:
    assert common.peek_text(tmp_path) == ""


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("trunc_1.gz", b"\x1f"),
        ("trunc_2.gz", b"\x1f\x8b"),
        ("trunc_header.gz", b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"),
        (
            "trunc_partial.gz",
            zlib.compressobj(wbits=31).compress(
                b'{"traceEvents": [{"name": "kernel"}]}' * 5
            )[:18],
        ),
    ],
)
def test_peek_text_truncated_gz(
    tmp_path: pathlib.Path, filename: str, data: bytes
) -> None:
    path = tmp_path / filename
    path.write_bytes(data)
    assert common.peek_text(path) == ""


def test_peek_text_corrupted_gz_stream(tmp_path: pathlib.Path) -> None:
    valid_gz = gzip.compress(b'{"traceEvents": [{"name": "kernel"}]}' * 10)
    # Corrupt the deflate payload after the 10-byte header
    corrupt_data = valid_gz[:12] + b"\xff\xff\xff\xff\xff\xff" + valid_gz[18:]
    path = tmp_path / "corrupted.json.gz"
    path.write_bytes(corrupt_data)
    assert common.peek_text(path) == ""


def test_adapters_survive_corrupted_and_truncated_traces(
    tmp_path: pathlib.Path,
) -> None:
    kineto = kineto_reader.KinetoAdapter()
    jax = jax_reader.JaxAdapter()

    # 1. Truncated gzip trace
    p_trunc = tmp_path / "incomplete_trace.json.gz"
    p_trunc.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff")

    # 2. Corrupted gzip trace
    valid_gz = gzip.compress(b'{"traceEvents": []}')
    p_corrupt = tmp_path / "corrupt_trace.json.gz"
    p_corrupt.write_bytes(valid_gz[:12] + b"\xff\xff\xff\xff" + valid_gz[16:])

    for p in (p_trunc, p_corrupt):
        assert kineto.detect(p) is False
        assert jax.detect(p) is False
        assert discovery.detect_format(p) is None

    # resolve_bundle on a directory containing only corrupted/truncated traces
    # should raise ValueError because no valid traces are detected, rather than
    # an unhandled EOFError or zlib.error
    with pytest.raises(ValueError, match="No recognized trace files found"):
        discovery.resolve_bundle(tmp_path)


def test_write_tables(tmp_path: pathlib.Path) -> None:
    out_dir = tmp_path / "tables_output"
    tables = {
        "metrics": pl.DataFrame(
            {"metric": ["latency", "throughput"], "value": [1.2, 450.0]}
        ),
        "nodes": pl.DataFrame({"id": [1, 2], "name": ["fwd", "bwd"]}),
    }
    result = common.write_tables(out_dir, tables)

    assert set(result.keys()) == {"metrics", "nodes"}
    assert result["metrics"] == out_dir / "metrics.parquet"
    assert result["nodes"] == out_dir / "nodes.parquet"
    assert result["metrics"].exists()
    assert result["nodes"].exists()

    df_read = pl.read_parquet(result["metrics"])
    assert df_read["metric"].to_list() == ["latency", "throughput"]
