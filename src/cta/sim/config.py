"""Simulation settings.

Every setting has a dotted name shared by TOML files, ``--set``, presets and
the native engine (``native/src/sim/params.hpp``). ``load_config`` applies
the defaults (LogGOPSim's), a preset, a TOML file and overrides, in that
order. Values may carry units: times ``"1.5us"``, sizes ``"64KiB"``,
bandwidths ``"400Gbps"`` or ``"50GB/s"``, per-byte costs as ns or as a
bandwidth (``loggops.G = "50GB/s"``). Plain numbers are ns, bytes, GB/s and
Hz. A TOML file nests the names in tables::

    model = "flow"
    preset = "dgx-h100"
    [topology]
    kind = "fat_tree"
    [compute.rank_scale]
    3 = 1.25
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import copy
import dataclasses
import difflib
import pathlib
import tomllib
from typing import Any

from cta.sim import units

NETWORK_MODELS = ("loggops", "alpha_beta", "flow")
COLLECTIVE_LEVELS = ("lowered", "analytical")
TOPOLOGIES = ("switch", "fat_tree", "torus", "ring", "dragonfly")
ALGORITHMS = (
    "ring",
    "recursive_doubling",
    "halving_doubling",
    "binomial_tree",
    "double_binary_tree",
    "direct",
    "hierarchical",
)
# Chakra's CollectiveCommType names, in its numbering.
COLLECTIVE_TYPES = (
    "ALL_REDUCE",
    "REDUCE",
    "ALL_GATHER",
    "GATHER",
    "SCATTER",
    "BROADCAST",
    "ALL_TO_ALL",
    "REDUCE_SCATTER",
    "REDUCE_SCATTER_BLOCK",
    "BARRIER",
)
DEFAULT_ALGORITHMS = {
    "ALL_REDUCE": "ring",
    "REDUCE": "binomial_tree",
    "ALL_GATHER": "ring",
    "GATHER": "binomial_tree",
    "SCATTER": "binomial_tree",
    "BROADCAST": "binomial_tree",
    "ALL_TO_ALL": "direct",
    "REDUCE_SCATTER": "ring",
    "REDUCE_SCATTER_BLOCK": "ring",
    "BARRIER": "recursive_doubling",
}


def _suggestion(word: str, options: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _choice(*allowed: str) -> Callable[[Any], str]:
    def parse(value: Any) -> str:
        if str(value) not in allowed:
            raise ValueError(
                f"{value!r} is not one of {', '.join(allowed)}"
                + _suggestion(str(value), allowed)
            )
        return str(value)

    return parse


def _int(value: Any) -> int:
    if isinstance(value, bool) or float(value) != int(float(value)):
        raise ValueError(f"expected an integer, got {value!r}")
    return int(float(value))


def _float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _optional(parse: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda v: None if v is None or str(v).lower() == "none" else parse(v)


def _ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [v for v in value.replace("x", ",").split(",") if v.strip()]
    return tuple(_int(v) for v in value)


def _table(
    key: Callable[[Any], Any], value: Callable[[Any], Any]
) -> Callable[[Any], dict[Any, Any]]:
    return lambda table: {key(k): value(v) for k, v in dict(table).items()}


@dataclasses.dataclass(frozen=True, slots=True)
class Setting:
    """A setting's default, parser and help; ``native``: the engine reads it."""

    default: Any
    parse: Callable[[Any], Any]
    help: str
    native: bool = True


def _intra(name: str, parse: Callable[[Any], Any]) -> Setting:
    return Setting(None, _optional(parse), f"{name} between ranks of a node.")


def _local(
    default: Any, parse: Callable[[Any], Any], help_text: str
) -> Setting:
    return Setting(default, parse, help_text, native=False)


SETTINGS: dict[str, Setting] = {
    "model": Setting("loggops", _choice(*NETWORK_MODELS), "Network model."),
    "seed": Setting(1, _int, "Seed of noise, degraded links and jitter."),
    "world_size": Setting(0, _int, "Ranks; 0: as many as the trace has."),
    "ranks_per_node": Setting(1, _int, "Ranks sharing a node."),
    "max_events": Setting(0, _int, "Stop after this many events; 0: never."),
    "loggops.L": Setting(2500.0, units.time_ns, "Latency."),
    "loggops.o": Setting(1500.0, units.time_ns, "CPU overhead per message."),
    "loggops.g": Setting(1000.0, units.time_ns, "NIC gap per message."),
    "loggops.G": Setting(6.0, units.ns_per_byte, "NIC gap per byte."),
    "loggops.O": Setting(0.0, units.ns_per_byte, "CPU overhead per byte."),
    "loggops.S": Setting(65535, units.size_bytes, "Largest eager message."),
    "loggops.intra_L": _intra("L", units.time_ns),
    "loggops.intra_o": _intra("o", units.time_ns),
    "loggops.intra_g": _intra("g", units.time_ns),
    "loggops.intra_G": _intra("G", units.ns_per_byte),
    "loggops.intra_O": _intra("O", units.ns_per_byte),
    "alpha_beta.alpha": Setting(5500.0, units.time_ns, "Cost per message."),
    "alpha_beta.beta": Setting(6.0, units.ns_per_byte, "Cost per byte."),
    "alpha_beta.intra_alpha": _intra("alpha", units.time_ns),
    "alpha_beta.intra_beta": _intra("beta", units.ns_per_byte),
    "topology.kind": Setting("switch", _choice(*TOPOLOGIES), "Flow fabric."),
    "topology.link_bandwidth": Setting(1 / 6, units.bandwidth, "Link speed."),
    "topology.link_latency": Setting(1250.0, units.time_ns, "Per link."),
    "topology.intra_bandwidth": Setting(
        0.0,
        units.bandwidth,
        "Node switch; 0: same-node traffic uses the fabric.",
    ),
    "topology.intra_latency": Setting(0.0, units.time_ns, "Node switch."),
    "topology.global_bandwidth": Setting(
        0.0, units.bandwidth, "Dragonfly global links; 0: link_bandwidth."
    ),
    "topology.oversubscription": Setting(1.0, _float, "Fat-tree uplinks."),
    "topology.dims": Setting((), _ints, "Torus dimensions, e.g. 4x4x2."),
    "topology.hosts_per_leaf": Setting(16, _int, "Fat tree."),
    "topology.spines": Setting(8, _int, "Fat tree."),
    "topology.hosts_per_router": Setting(4, _int, "Dragonfly."),
    "topology.routers_per_group": Setting(8, _int, "Dragonfly."),
    "collectives.level": _local(
        "lowered", _choice(*COLLECTIVE_LEVELS), "Messages, or costed whole."
    ),
    "collectives.groups": _local(
        None, _optional(str), "JSON file {pg_name: [ranks]}."
    ),
    **{
        f"collectives.algorithms.{name}": Setting(
            algorithm, _choice(*ALGORITHMS), f"Algorithm of {name}."
        )
        for name, algorithm in DEFAULT_ALGORITHMS.items()
    },
    "collectives.completion": Setting(
        "all",
        _choice("all", "outputs"),
        "Done after the last send (MPI, NCCL) or with the result (GOAL).",
    ),
    "collectives.all_to_all_size": Setting(
        "total",
        _choice("total", "per_peer"),
        "All-to-all size: the whole buffer, or what each peer gets.",
    ),
    "collectives.cost": Setting(
        "model",
        _choice("model", "recorded"),
        "Analytical cost: the model's, or the trace's.",
    ),
    "collectives.reduce_ns_per_byte": Setting(
        0.0, units.ns_per_byte, "Reduction cost."
    ),
    "collectives.chunks": Setting(1, _int, "Pipeline chunks."),
    "compute.model": _local(
        "recorded",
        _choice("recorded", "roofline", "roofline_if_missing"),
        "Compute durations (cta.sim.compute).",
    ),
    "compute.peak_flops": _local(0.0, units.flops, "Roofline."),
    "compute.memory_bandwidth": _local(0.0, units.bandwidth, "Roofline."),
    "compute.efficiency": _local(1.0, _float, "Fraction of the roofline."),
    "compute.scale": _local(1.0, _float, "Scales all compute."),
    "compute.rank_scale": _local({}, _table(int, float), "Per-rank factor."),
    "compute.name_scale": _local(
        {}, _table(str, float), "Factor per op-name regex."
    ),
    "compute.jitter": _local(0.0, _float, "Sigma of mean-1 log-normal noise."),
    "noise.frequency": Setting(0.0, units.rate_per_second, "OS detours/s."),
    "noise.duration": Setting(0.0, units.time_ns, "Mean detour length."),
    "noise.distribution": Setting(
        "fixed", _choice("fixed", "exponential"), "Detour lengths."
    ),
    "noise.nic_scale": Setting({}, _table(int, float), "Per-rank NIC factor."),
    "noise.degraded_fraction": Setting(0.0, _float, "Share of slow links."),
    "noise.degraded_factor": Setting(1.0, _float, "Their bandwidth factor."),
}


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and name not in SETTINGS:
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


class SimConfig(Mapping[str, Any]):
    """Every setting's value by dotted name; ``with_overrides`` copies."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        """Takes parsed values; the other settings keep their defaults."""
        self._values = {
            k: copy.deepcopy(s.default) for k, s in SETTINGS.items()
        }
        self._values.update(values or {})

    def __getitem__(self, key: str) -> Any:
        """The value of a setting."""
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """The setting names."""
        return iter(self._values)

    def __len__(self) -> int:
        """The number of settings."""
        return len(self._values)

    def with_overrides(self, overrides: Mapping[str, Any]) -> SimConfig:
        """Returns a copy with the (dotted or nested) settings parsed and set.

        Raises:
            ValueError: An unknown key or a value that does not parse.
        """
        values = copy.deepcopy(self._values)
        for key, value in _flatten(overrides).items():
            table, _, entry = key.rpartition(".")
            try:
                if key in SETTINGS:
                    values[key] = SETTINGS[key].parse(value)
                elif table in SETTINGS and isinstance(values[table], dict):
                    parsed = SETTINGS[table].parse({entry: value})
                    values[table] = {**values[table], **parsed}
                else:
                    raise ValueError(
                        "unknown setting" + _suggestion(key, SETTINGS)
                    )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key}: {error}") from None
        return SimConfig(values)

    def to_native(self, world_size: int) -> dict[str, Any]:
        """Returns the parameters ``cta._native.simulate`` reads."""
        params = {
            k: v
            for k, v in self._values.items()
            if SETTINGS[k].native and v is not None
        }
        params["world_size"] = max(world_size, self["world_size"])
        params["topology.dims"] = list(self["topology.dims"])
        scale = self["noise.nic_scale"]
        ranks = max(params["world_size"], max(scale, default=-1) + 1)
        params["noise.nic_scale"] = (
            [scale.get(r, 1.0) for r in range(ranks)] if scale else []
        )
        return params


def load_config(
    path: str | pathlib.Path | None = None,
    *,
    preset: str | None = None,
    overrides: Mapping[str, Any] | Iterable[str] = (),
    model: str | None = None,
) -> SimConfig:
    """Builds a config from defaults, a preset, a TOML file, overrides, model.

    A TOML file's top-level ``preset`` applies unless ``preset`` is given;
    ``overrides`` is a dict or ``"key=value"`` strings.

    Raises:
        ValueError: An unknown preset, key or unparseable value.
    """
    from cta.sim import presets

    data: dict[str, Any] = {}
    if path is not None:
        with pathlib.Path(path).open("rb") as fh:
            data = tomllib.load(fh)
    preset = preset or data.get("preset")
    data.pop("preset", None)
    config = presets.apply(SimConfig(), preset) if preset else SimConfig()
    if not isinstance(overrides, Mapping):
        overrides = parse_assignments(overrides)
    config = config.with_overrides(data).with_overrides(overrides)
    return config.with_overrides({"model": model}) if model else config


def parse_assignments(assignments: Iterable[str]) -> dict[str, str]:
    """Reads ``key=value`` strings into a dict."""
    out: dict[str, str] = {}
    for item in assignments:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected key=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out
