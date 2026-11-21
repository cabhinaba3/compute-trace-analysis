"""Table work goes through `_device.collect`.

The Polars GPU engine runs it when asked for and present.

`cuda` runs a query with Polars' GPU engine (cudf-polars); anything else, a
missing cudf-polars or a query the GPU engine fails on, gives the CPU result.
Polars has no Metal engine, so on a Mac tables stay on the CPU.
"""

from __future__ import annotations

import polars as pl
import polars.testing as pl_testing
import pytest

from cta import _device
from cta.aspects.device import kernels


def _frame(rows: int = 1000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": [f"k{i % 7}" for i in range(rows)],
            "duration_ns": [(i * 37) % 1000 + 1 for i in range(rows)],
        }
    )


def _query(frame: pl.DataFrame) -> pl.LazyFrame:
    return (
        frame.lazy()
        .group_by("name")
        .agg(pl.col("duration_ns").sum().alias("total"), pl.len().alias("n"))
        .sort("name")
    )


@pytest.fixture(autouse=True)
def _fresh() -> None:
    _device.reset_warnings()


def test_collect_on_the_cpu_is_a_plain_collect() -> None:
    frame = _frame()

    with _device.use("cpu"):
        got = _device.collect(_query(frame), rows=frame.height)

    pl_testing.assert_frame_equal(got, _query(frame).collect())


def test_a_missing_cudf_polars_gives_the_cpu_result_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setattr(_device, "backends", lambda: ["cuda"])
    monkeypatch.setattr(_device, "polars_gpu_available", lambda: False)

    with _device.use("cuda"), pytest.warns(UserWarning, match="cudf-polars"):
        got = _device.collect(_query(frame), rows=frame.height)

    pl_testing.assert_frame_equal(got, _query(frame).collect())


def test_a_gpu_engine_failure_gives_the_cpu_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setattr(_device, "backends", lambda: ["cuda"])
    monkeypatch.setattr(_device, "polars_gpu_available", lambda: True)

    def boom(lazy: pl.LazyFrame) -> pl.DataFrame:
        raise RuntimeError("out of device memory")

    monkeypatch.setattr(_device, "_collect_on_gpu", boom)

    with _device.use("cuda"), pytest.warns(UserWarning, match="out of device"):
        got = _device.collect(_query(frame), rows=frame.height)

    pl_testing.assert_frame_equal(got, _query(frame).collect())


def test_auto_keeps_small_tables_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setattr(_device, "backends", lambda: ["cuda"])
    monkeypatch.setattr(_device, "polars_gpu_available", lambda: True)
    monkeypatch.setattr(
        _device,
        "_collect_on_gpu",
        lambda lazy: pytest.fail("a small table went to the GPU"),
    )

    with _device.use("auto"):
        got = _device.collect(_query(frame), rows=frame.height)

    assert got.height == 7


def test_the_real_gpu_engine_agrees_with_the_cpu() -> None:
    if not (_device.backends() and _device.polars_gpu_available()):
        pytest.skip("no CUDA device with cudf-polars here")
    frame = _frame(200_000)

    with _device.use("cuda"):
        got = _device.collect(_query(frame), rows=frame.height)

    pl_testing.assert_frame_equal(got, _query(frame).collect())


def test_the_kernel_totals_are_the_same_through_collect() -> None:
    frame = (
        _frame()
        .rename({"name": "kernel"})
        .with_columns(pl.lit("a").alias("category"))
    )

    got = kernels._time_by(frame, "kernel").sort("kernel")

    want = (
        frame.group_by("kernel")
        .agg(
            (pl.col("duration_ns").sum().cast(pl.Float64) / 1e6).alias(
                "duration_ms"
            ),
            pl.len().alias("launches"),
        )
        .sort("kernel")
    )
    pl_testing.assert_frame_equal(got, want)
