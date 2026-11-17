"""Named machine descriptions that fill in every network model at once.

A preset describes a fabric by a few measurable numbers (bandwidth per
direction, end-to-end latency, per-message CPU overhead and gap, eager limit,
and optionally a faster intra-node level), and ``apply`` derives each model's
parameters from them, so the three models describe the same machine:

* LogGOPS: L = latency, o = overhead, g = gap, G = 1 / bandwidth, S = eager.
* Alpha-beta: alpha = L + 2o, beta = 1 / bandwidth.
* Flow: link bandwidth = bandwidth; per-link latency = latency / hops of the
  default topology (a switch: two links); o comes from LogGOPS.

Bandwidths come from vendor datasheets. Latencies and overheads are nominal
values of the kind small-message benchmarks (osu_latency, nccl-tests) show;
they are meant for first-order studies. Measure your own machine and set the
numbers (``--set loggops.L=...``) when the absolute times matter.
"""

from __future__ import annotations

import dataclasses

from cta.sim import config as config_lib


@dataclasses.dataclass(frozen=True, slots=True)
class Preset:
    """A fabric, in the numbers ``apply`` turns into model parameters.

    Attributes:
        name: Key of the preset.
        description: One line about what it models and where its numbers
            come from.
        bandwidth: Per-direction bandwidth per rank, GB/s.
        latency: End-to-end latency of a small message, ns.
        overhead: CPU (or proxy) overhead per message, ns.
        gap: Minimum time between two messages from one NIC, ns.
        eager: Largest eager message, bytes.
        ranks_per_node: Ranks per node; 1 for no intra-node level.
        intra_bandwidth: Per-direction intra-node bandwidth per rank, GB/s.
        intra_latency: Intra-node small-message latency, ns.
    """

    name: str
    description: str
    bandwidth: float
    latency: float
    overhead: float
    gap: float
    eager: int
    ranks_per_node: int = 1
    intra_bandwidth: float = 0.0
    intra_latency: float = 0.0


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset(
            "loggopsim",
            "LogGOPSim's command-line defaults (L=2.5us, o=1.5us, g=1us, "
            "G=6ns/B, S=65535); the defaults of this simulator too.",
            bandwidth=1.0 / 6.0,
            latency=2500.0,
            overhead=1500.0,
            gap=1000.0,
            eager=65535,
        ),
        Preset(
            "ib-hdr200",
            "InfiniBand HDR, 200 Gb/s per port (25 GB/s); ~1.1us MPI latency.",
            bandwidth=25.0,
            latency=1100.0,
            overhead=300.0,
            gap=100.0,
            eager=16384,
        ),
        Preset(
            "ib-ndr400",
            "InfiniBand NDR, 400 Gb/s per port (50 GB/s); ~1us MPI latency.",
            bandwidth=50.0,
            latency=1000.0,
            overhead=250.0,
            gap=60.0,
            eager=16384,
        ),
        Preset(
            "roce-400g",
            "RoCEv2 on 400GbE (50 GB/s); ~2us latency through a "
            "lossless Ethernet fabric.",
            bandwidth=50.0,
            latency=2000.0,
            overhead=400.0,
            gap=80.0,
            eager=16384,
        ),
        Preset(
            "eth-100g",
            "TCP over 100GbE (12.5 GB/s); ~10us latency including the "
            "kernel network stack.",
            bandwidth=12.5,
            latency=10000.0,
            overhead=3000.0,
            gap=500.0,
            eager=65536,
        ),
        Preset(
            "dgx-h100",
            "DGX H100 nodes: 8 GPUs per node on NVLink 4 (900 GB/s "
            "bidirectional, 450 GB/s per direction), one 400 Gb/s NDR NIC "
            "per GPU between nodes.",
            bandwidth=50.0,
            latency=2500.0,
            overhead=1000.0,
            gap=100.0,
            eager=65536,
            ranks_per_node=8,
            intra_bandwidth=450.0,
            intra_latency=1500.0,
        ),
        Preset(
            "gb200-nvl72",
            "GB200 NVL72: 72 GPUs in one NVLink 5 domain (1.8 TB/s "
            "bidirectional, 900 GB/s per direction), 800 Gb/s ConnectX-8 "
            "per GPU beyond it.",
            bandwidth=100.0,
            latency=2500.0,
            overhead=1000.0,
            gap=100.0,
            eager=65536,
            ranks_per_node=72,
            intra_bandwidth=900.0,
            intra_latency=1500.0,
        ),
    )
}


def apply(config: config_lib.SimConfig, name: str) -> config_lib.SimConfig:
    """Returns a copy of ``config`` with the preset's numbers in every model.

    Raises:
        ValueError: There is no preset called ``name``.
    """
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(
            f"unknown preset {name!r}; presets: {', '.join(PRESETS)}"
        )
    inverse = 1.0 / preset.bandwidth
    overrides: dict[str, object] = {
        "ranks_per_node": preset.ranks_per_node,
        "loggops.L": preset.latency,
        "loggops.o": preset.overhead,
        "loggops.g": preset.gap,
        "loggops.G": inverse,
        "loggops.S": preset.eager,
        "alpha_beta.alpha": preset.latency + 2 * preset.overhead,
        "alpha_beta.beta": inverse,
        "topology.link_bandwidth": preset.bandwidth,
        "topology.link_latency": preset.latency / 2,
    }
    if preset.intra_bandwidth > 0:
        intra_inverse = 1.0 / preset.intra_bandwidth
        overrides.update(
            {
                "loggops.intra_L": preset.intra_latency,
                "loggops.intra_G": intra_inverse,
                "alpha_beta.intra_alpha": preset.intra_latency
                + 2 * preset.overhead,
                "alpha_beta.intra_beta": intra_inverse,
                "topology.intra_bandwidth": preset.intra_bandwidth,
                "topology.intra_latency": preset.intra_latency / 2,
            }
        )
    return config.with_overrides(overrides)
