"""Machine presets: a few measured numbers that set every network model.

``apply`` derives LogGOPS (L, o, g, G = 1/bandwidth, S), alpha-beta
(alpha = L + 2o, beta = 1/bandwidth) and the flow model (links of the given
bandwidth, half the latency each) from one preset. Bandwidths are datasheet
values per rank and direction; latencies and overheads are typical
small-message figures, to be calibrated when absolute times matter.
"""

from __future__ import annotations

import dataclasses

from cta.sim import config as config_lib


@dataclasses.dataclass(frozen=True, slots=True)
class Preset:
    """GB/s, ns and bytes; intra_* describe the node level, if any."""

    description: str
    bandwidth: float
    latency: float
    overhead: float
    gap: float
    eager: int
    ranks_per_node: int = 1
    intra_bandwidth: float = 0.0
    intra_latency: float = 0.0


PRESETS = {
    "loggopsim": Preset(
        "LogGOPSim's defaults, which are this simulator's too.",
        1 / 6, 2500, 1500, 1000, 65535,
    ),
    "ib-hdr200": Preset("InfiniBand HDR, 200 Gb/s.", 25, 1100, 300, 100, 16384),
    "ib-ndr400": Preset("InfiniBand NDR, 400 Gb/s.", 50, 1000, 250, 60, 16384),
    "roce-400g": Preset("RoCEv2 on 400GbE.", 50, 2000, 400, 80, 16384),
    "eth-100g": Preset("TCP on 100GbE.", 12.5, 10000, 3000, 500, 65536),
    "dgx-h100": Preset(
        "8 H100 per node on NVLink 4 (450 GB/s each way), NDR 400 per GPU.",
        50, 2500, 1000, 100, 65536, 8, 450, 1500,
    ),
    "gb200-nvl72": Preset(
        "72 GPUs per NVLink 5 domain (900 GB/s), ConnectX-8 800 Gb/s per GPU.",
        100, 2500, 1000, 100, 65536, 72, 900, 1500,
    ),
}  # fmt: skip


def apply(config: config_lib.SimConfig, name: str) -> config_lib.SimConfig:
    """Returns ``config`` with the preset's numbers in every model."""
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; presets: {', '.join(PRESETS)}"
        )
    p = PRESETS[name]
    values: dict[str, object] = {
        "ranks_per_node": p.ranks_per_node,
        "loggops.L": p.latency,
        "loggops.o": p.overhead,
        "loggops.g": p.gap,
        "loggops.G": 1 / p.bandwidth,
        "loggops.S": p.eager,
        "alpha_beta.alpha": p.latency + 2 * p.overhead,
        "alpha_beta.beta": 1 / p.bandwidth,
        "topology.link_bandwidth": p.bandwidth,
        "topology.link_latency": p.latency / 2,
    }
    if p.intra_bandwidth:
        values |= {
            "loggops.intra_L": p.intra_latency,
            "loggops.intra_G": 1 / p.intra_bandwidth,
            "alpha_beta.intra_alpha": p.intra_latency + 2 * p.overhead,
            "alpha_beta.intra_beta": 1 / p.intra_bandwidth,
            "topology.intra_bandwidth": p.intra_bandwidth,
            "topology.intra_latency": p.intra_latency / 2,
        }
    return config.with_overrides(values)
