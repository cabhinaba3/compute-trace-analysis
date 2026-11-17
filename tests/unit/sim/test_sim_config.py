"""Tests of simulation settings: units, presets, TOML files and overrides."""

from __future__ import annotations

import pathlib

import pytest

from cta.sim import config as config_lib
from cta.sim import presets
from cta.sim import units


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("400Gbps", 50.0),
        ("400Gb/s", 50.0),
        ("50GB/s", 50.0),
        ("900 GB/s", 900.0),
        (12.5, 12.5),
    ],
)
def test_bandwidths(text: str | float, expected: float) -> None:
    assert units.bandwidth(text) == pytest.approx(expected)


def test_other_units() -> None:
    assert units.time_ns("2.5us") == 2500
    assert units.time_ns(10) == 10
    assert units.size_bytes("64KiB") == 65536
    assert units.size_bytes("8MB") == 8_000_000
    assert units.ns_per_byte("50GB/s") == pytest.approx(0.02)
    assert units.ns_per_byte(6) == 6
    assert units.rate_per_second("2kHz") == 2000
    assert units.flops("989TFLOPS") == pytest.approx(989e12)
    with pytest.raises(ValueError, match="unknown time unit"):
        units.time_ns("3 fortnights")
    with pytest.raises(ValueError, match="bandwidth"):
        units.bandwidth("fast")


def test_overrides_parse_units_and_reject_unknown_keys() -> None:
    settings = config_lib.SimConfig().with_overrides(
        {
            "loggops.L": "1.2us",
            "loggops.G": "50GB/s",
            "collectives.algorithms.ALL_REDUCE": "halving_doubling",
            "compute.rank_scale.3": "1.5",
            "topology.dims": "4x4",
        }
    )
    assert settings.loggops.L == 1200
    assert pytest.approx(0.02) == settings.loggops.G
    assert settings.collectives.algorithms["ALL_REDUCE"] == "halving_doubling"
    assert settings.compute.rank_scale == {3: 1.5}
    assert settings.topology.dims == (4, 4)
    with pytest.raises(ValueError, match="did you mean 'loggops'"):
        config_lib.SimConfig().with_overrides({"loggop.L": 1})
    with pytest.raises(ValueError, match="not one of"):
        config_lib.SimConfig().with_overrides(
            {"collectives.algorithms.ALL_REDUCE": "rings"}
        )
    with pytest.raises(ValueError, match="unknown collective type"):
        config_lib.SimConfig().with_overrides(
            {"collectives.algorithms.ALLREDUCE": "ring"}
        )


def test_load_config_layers_preset_file_and_overrides(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "sim.toml"
    path.write_text(
        'preset = "dgx-h100"\nmodel = "flow"\n\n[topology]\n'
        'kind = "fat_tree"\nhosts_per_leaf = 32\n\n[noise]\n'
        'frequency = "1kHz"\nduration = "20us"\n'
    )
    settings = config_lib.load_config(path, overrides=["topology.spines=4"])
    assert settings.model == "flow"
    assert settings.ranks_per_node == 8
    assert settings.loggops.intra_G == pytest.approx(1 / 450)
    assert settings.topology.kind == "fat_tree"
    assert (settings.topology.hosts_per_leaf, settings.topology.spines) == (
        32,
        4,
    )
    params = settings.to_native(16)
    assert params["noise_frequency"] == pytest.approx(1e-6)
    assert params["noise_duration"] == 20_000
    assert params["intra_bandwidth"] == 450.0


def test_presets_describe_one_machine_in_every_model() -> None:
    for name in presets.PRESETS:
        settings = presets.apply(config_lib.SimConfig(), name)
        preset = presets.PRESETS[name]
        assert pytest.approx(1 / preset.bandwidth) == settings.loggops.G
        assert settings.alpha_beta.beta == pytest.approx(1 / preset.bandwidth)
        assert settings.topology.link_bandwidth == preset.bandwidth
    with pytest.raises(ValueError, match="unknown preset"):
        presets.apply(config_lib.SimConfig(), "ib-xdr")


def test_collective_defaults() -> None:
    params = config_lib.SimConfig().to_native(4)
    # A collective waits for its last sends, as MPI and NCCL do; all-to-all
    # sizes are whole buffers, as in Chakra.
    assert params["data_completion"] is False
    assert params["all_to_all_per_peer"] is False
    other = config_lib.SimConfig().with_overrides(
        {
            "collectives.completion": "outputs",
            "collectives.all_to_all_size": "per_peer",
        }
    )
    assert other.to_native(4)["data_completion"] is True
    assert other.to_native(4)["all_to_all_per_peer"] is True


def test_nic_scale_becomes_a_per_rank_list() -> None:
    params = (
        config_lib.SimConfig()
        .with_overrides({"noise.nic_scale.2": 0.5})
        .to_native(4)
    )
    assert params["nic_scale"] == [1.0, 1.0, 0.5, 1.0]


def test_describe_lists_every_setting() -> None:
    keys = [key for key, _, _ in config_lib.describe()]
    assert (
        "loggops.L" in keys
        and "topology.kind" in keys
        and "compute.jitter" in keys
    )
