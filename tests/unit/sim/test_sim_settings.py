"""Tests of settings, presets, compute models, sweeps and Monte Carlo runs."""

from __future__ import annotations

import pathlib

import numpy as np
import polars as pl
import pytest

from cta.sim import compute
from cta.sim import config as config_lib
from cta.sim import engine
from cta.sim import graph as graph_lib
from cta.sim import presets
from cta.sim import sweep
from cta.sim import units


@pytest.mark.parametrize(
    ("text", "parse", "expected"),
    [
        ("400Gbps", units.bandwidth, 50.0),
        ("400Gb/s", units.bandwidth, 50.0),
        ("50GB/s", units.bandwidth, 50.0),
        ("64GiB/s", units.bandwidth, 2**36 / 1e9),
        (12.5, units.bandwidth, 12.5),
        ("2.5us", units.time_ns, 2500),
        ("64KiB", units.size_bytes, 65536),
        ("50GB/s", units.ns_per_byte, 0.02),
        ("6", units.ns_per_byte, 6),
        ("2kHz", units.rate_per_second, 2000),
        ("989TFLOPS", units.flops, 989e12),
    ],
)
def test_units(text: str | float, parse: object, expected: float) -> None:
    assert parse(text) == pytest.approx(expected)  # type: ignore[operator]


def test_bad_units() -> None:
    with pytest.raises(ValueError, match="unknown time unit"):
        units.time_ns("3 fortnights")
    with pytest.raises(ValueError, match="bandwidth"):
        units.bandwidth("fast")


def test_overrides() -> None:
    settings = config_lib.SimConfig().with_overrides(
        {
            "loggops.L": "1.2us",
            "loggops.G": "50GB/s",
            "collectives.algorithms.ALL_REDUCE": "halving_doubling",
            "compute.rank_scale.3": "1.5",
            "topology.dims": "4x4",
            "noise.nic_scale.2": 0.5,
        }
    )
    assert (settings["loggops.L"], settings["loggops.G"]) == (
        1200,
        pytest.approx(0.02),
    )
    assert settings["compute.rank_scale"] == {3: 1.5}
    native = settings.to_native(4)
    assert native["topology.dims"] == [4, 4]
    assert native["noise.nic_scale"] == [1.0, 1.0, 0.5, 1.0]
    assert native["collectives.algorithms.ALL_REDUCE"] == "halving_doubling"
    assert (
        "compute.rank_scale" not in native
        and native["collectives.completion"] == "all"
    )
    for key, message in (
        ("loggop.L", "did you mean 'loggops.L'"),
        ("collectives.algorithms.ALLREDUCE", "unknown setting"),
        ("collectives.algorithms.ALL_REDUCE", "not one of"),
    ):
        with pytest.raises(ValueError, match=message):
            config_lib.SimConfig().with_overrides({key: "rings"})


def test_load_config_layers(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "sim.toml"
    path.write_text(
        'preset = "dgx-h100"\nmodel = "flow"\n[topology]\nkind = "fat_tree"\n'
        '[noise]\nfrequency = "1kHz"\n[compute.rank_scale]\n3 = 1.25\n'
    )
    settings = config_lib.load_config(path, overrides=["topology.spines=4"])
    assert (
        settings["model"],
        settings["ranks_per_node"],
        settings["topology.spines"],
    ) == ("flow", 8, 4)
    assert settings["loggops.intra_G"] == pytest.approx(1 / 450)
    assert settings["compute.rank_scale"] == {3: 1.25}
    assert settings.to_native(16)["noise.frequency"] == 1000


def test_presets() -> None:
    for name, preset in presets.PRESETS.items():
        settings = presets.apply(config_lib.SimConfig(), name)
        assert settings["loggops.G"] == pytest.approx(
            settings["alpha_beta.beta"]
        )
        assert settings["topology.link_bandwidth"] == preset.bandwidth
    with pytest.raises(ValueError, match="unknown preset"):
        presets.apply(config_lib.SimConfig(), "ib-xdr")


def tiny_graph() -> graph_lib.SimGraph:
    nodes = pl.DataFrame(
        {
            "rank": [0, 0, 1],
            "node_id": [0, 1, 0],
            "name": ["attn", "mlp", "attn"],
            "op_class": "compute",
            "comm_type": "",
            "duration_ns": [1000, 2000, 1000],
            "size_bytes": [4000, 0, 0],
            "src_rank": -1,
            "dst_rank": -1,
            "message_tag": 0,
            "num_ops": [2_000_000, 0, 0],
        }
    ).cast(
        {
            "rank": pl.UInt32,
            "node_id": pl.UInt64,
            "src_rank": pl.Int32,
            "dst_rank": pl.Int32,
        }
    )
    edges = pl.DataFrame(
        {"rank": [0], "src_id": [0], "dst_id": [1], "dep_kind": ["data"]}
    ).cast({"rank": pl.UInt32, "src_id": pl.UInt64, "dst_id": pl.UInt64})
    return graph_lib.from_tables(
        nodes, edges, trace_format="chakra", label="tiny"
    )


def durations(**overrides: object) -> list[int]:
    return compute.durations(
        tiny_graph(), config_lib.SimConfig().with_overrides(overrides)
    ).tolist()


def test_compute_models() -> None:
    assert durations(
        **{
            "compute.scale": 2,
            "compute.rank_scale.1": 1.5,
            "compute.name_scale.^mlp": 0.25,
        }
    ) == [
        2000,
        1000,
        3000,
    ]
    # 2 MFLOP at 1 TFLOP/s is 2 us, 4000 bytes at 1 GB/s 4 us.
    roofline = {
        "compute.model": "roofline",
        "compute.peak_flops": 1e12,
        "compute.memory_bandwidth": 1,
    }
    assert durations(**roofline) == [4000, 2000, 1000]
    assert durations(
        **{"compute.model": "roofline_if_missing", "compute.peak_flops": 1e12}
    ) == [1000, 2000, 1000]
    with pytest.raises(ValueError, match="roofline needs"):
        durations(**{"compute.model": "roofline"})
    jittered = durations(**{"compute.jitter": 0.2, "seed": 5})
    assert (
        jittered
        == durations(**{"compute.jitter": 0.2, "seed": 5})
        != durations(**{"compute.jitter": 0.2})
    )
    straggler = config_lib.SimConfig().with_overrides(
        {"compute.rank_scale.1": 4}
    )
    assert (
        engine.run_graph(tiny_graph()).makespan_ns,
        engine.run_graph(tiny_graph(), straggler).makespan_ns,
    ) == (
        3000,
        4000,
    )


def test_sweep_and_monte_carlo(
    training_traces: dict[str, pathlib.Path],
) -> None:
    graph = graph_lib.load_graph(
        training_traces["goal"], cache_dir=training_traces["cache"]
    )
    table = sweep.sweep(
        graph,
        config_lib.SimConfig(),
        sweep.parse_grid(["loggops.G=6,3", "loggops.L=1us,bad"]),
    )
    ran = table.filter(pl.col("error").is_null()).sort("loggops.G")
    assert (table.height, ran.height) == (4, 2)
    assert (
        ran["makespan_ms"][0] < ran["makespan_ms"][1]
    )  # doubling the bandwidth helps
    noisy = config_lib.SimConfig().with_overrides(
        {"noise.frequency": "10kHz", "noise.duration": "50us"}
    )
    trials, stats = sweep.monte_carlo(graph, noisy, 3)
    assert trials["seed"].to_list() == [1, 2, 3]
    assert (
        engine.run_graph(graph).makespan_ns / 1e6
        < stats["min_ms"]
        <= stats["p50_ms"]
        <= stats["max_ms"]
    )
    assert np.isfinite(stats["std_ms"])
