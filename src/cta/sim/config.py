"""Simulation settings, and how presets, TOML files and overrides make them.

``load_config`` starts from the defaults (LogGOPSim's own: L=2.5us, o=1.5us,
g=1us, G=6ns/B, S=65535), applies a preset (``cta.sim.presets``), then a TOML
file, then ``section.key=value`` overrides, in that order. Every value may
carry a unit: times ``"1.5us"``, sizes ``"64KiB"``, bandwidths ``"400Gbps"``
or ``"50GB/s"``, per-byte costs as a number of ns or as a bandwidth
(``G = "50GB/s"`` is 0.02 ns/B), rates ``"1kHz"``, FLOP rates ``"989TFLOPS"``.
Plain numbers are nanoseconds, bytes, GB/s, ns per byte, Hz and FLOP/s.

A TOML file mirrors the dataclasses::

    model = "flow"
    preset = "ib-ndr400"        # optional, applied before the rest
    ranks_per_node = 8

    [loggops]
    L = "1.2us"
    G = "50GB/s"

    [topology]
    kind = "fat_tree"
    hosts_per_leaf = 32

    [collectives]
    level = "lowered"
    algorithms = { ALL_REDUCE = "halving_doubling", ALL_TO_ALL = "ring" }

    [compute]
    rank_scale = { 3 = 1.25 }   # rank 3 is a 25% straggler

    [noise]
    frequency = "1kHz"
    duration = "25us"
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
COMPUTE_MODELS = ("recorded", "roofline", "roofline_if_missing")


def _choice(*allowed: str) -> Callable[[Any], str]:
    """Returns a parser that accepts one of ``allowed``."""

    def parse(value: Any) -> str:
        text = str(value)
        if text not in allowed:
            raise ValueError(
                f"{text!r} is not one of {', '.join(allowed)}"
                + _suggestion(text, allowed)
            )
        return text

    return parse


def _int(value: Any) -> int:
    if isinstance(value, bool) or int(value) != float(value):
        raise ValueError(f"expected an integer, got {value!r}")
    return int(value)


def _float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def _optional(parse: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def parse_optional(value: Any) -> Any:
        if value is None or (
            isinstance(value, str) and value.lower() == "none"
        ):
            return None
        return parse(value)

    return parse_optional


def _ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [v for v in value.replace("x", ",").split(",") if v.strip()]
    return tuple(_int(v) for v in value)


def _suggestion(word: str, options: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _field(default: Any, parse: Callable[[Any], Any], help_text: str) -> Any:
    """A dataclass field whose metadata holds its parser and help text."""
    if isinstance(default, dict | list):
        return dataclasses.field(
            default_factory=lambda: copy.deepcopy(default),
            metadata={"parse": parse, "help": help_text},
        )
    return dataclasses.field(
        default=default, metadata={"parse": parse, "help": help_text}
    )


@dataclasses.dataclass(slots=True)
class LogGOPSConfig:
    """LogGOPS parameters (also the host overheads o, O of the flow model).

    The intra_* parameters, when set, apply to ranks of one node. The names
    are the model's own (L, o, g, G, O, S), hence the noqa marks.
    """

    L: float = _field(2500.0, units.time_ns, "Network latency.")
    o: float = _field(1500.0, units.time_ns, "CPU overhead per message.")
    g: float = _field(1000.0, units.time_ns, "NIC gap between messages.")
    G: float = _field(6.0, units.ns_per_byte, "NIC gap per byte (1/bandwidth).")
    O: float = _field(0.0, units.ns_per_byte, "CPU overhead per byte.")  # noqa: E741
    S: int = _field(65535, units.size_bytes, "Largest eager message.")
    rendezvous: str = _field(
        "sync",
        _choice("sync", "rts_cts"),
        "Messages above S: 'sync' (LogGOPSim) or 'rts_cts' handshake.",
    )
    intra_L: float | None = _field(None, _optional(units.time_ns), "")  # noqa: N815
    intra_o: float | None = _field(None, _optional(units.time_ns), "")
    intra_g: float | None = _field(None, _optional(units.time_ns), "")
    intra_G: float | None = _field(None, _optional(units.ns_per_byte), "")  # noqa: N815
    intra_O: float | None = _field(None, _optional(units.ns_per_byte), "")  # noqa: N815


@dataclasses.dataclass(slots=True)
class AlphaBetaConfig:
    """Alpha-beta parameters: a message costs alpha + bytes * beta."""

    alpha: float = _field(5500.0, units.time_ns, "Per-message cost.")
    beta: float = _field(6.0, units.ns_per_byte, "Per-byte cost.")
    intra_alpha: float | None = _field(None, _optional(units.time_ns), "")
    intra_beta: float | None = _field(None, _optional(units.ns_per_byte), "")


@dataclasses.dataclass(slots=True)
class TopologyConfig:
    """The flow model's topology (see native/src/sim/topology.hpp)."""

    kind: str = _field("switch", _choice(*TOPOLOGIES), "Fabric shape.")
    link_bandwidth: float = _field(
        1.0 / 6.0, units.bandwidth, "NIC and fabric link bandwidth."
    )
    link_latency: float = _field(1250.0, units.time_ns, "Latency per link.")
    intra_bandwidth: float = _field(
        0.0,
        units.bandwidth,
        "Node switch bandwidth; 0 sends same-node traffic through the fabric.",
    )
    intra_latency: float = _field(0.0, units.time_ns, "Node switch latency.")
    global_bandwidth: float = _field(
        0.0, units.bandwidth, "Dragonfly global links; 0 = link_bandwidth."
    )
    dims: tuple[int, ...] = _field((), _ints, "Torus dimensions, e.g. 4x4x2.")
    hosts_per_leaf: int = _field(0, _int, "fat_tree: ranks per leaf (16).")
    spines: int = _field(0, _int, "fat_tree: spine switches (8).")
    oversubscription: float = _field(1.0, _float, "fat_tree: leaf uplinks.")
    hosts_per_router: int = _field(0, _int, "dragonfly: ranks per router (4).")
    routers_per_group: int = _field(
        0, _int, "dragonfly: routers per group (8)."
    )


@dataclasses.dataclass(slots=True)
class CollectiveConfig:
    """How collectives run: lowered to messages or costed analytically."""

    level: str = _field(
        "lowered", _choice(*COLLECTIVE_LEVELS), "lowered or analytical."
    )
    algorithms: dict[str, str] = _field(
        DEFAULT_ALGORITHMS, _choice(*ALGORITHMS), "Algorithm per type."
    )
    cost: str = _field(
        "model",
        _choice("model", "recorded"),
        "analytical: from the model, or the trace's recorded duration.",
    )
    completion: str = _field(
        "outputs",
        _choice("outputs", "all"),
        "A member is done once it has its result (outputs, as GOAL "
        "schedules chain on the last recv) or after all its sends too (all).",
    )
    reduce_ns_per_byte: float = _field(
        0.0, units.ns_per_byte, "Reduction cost per byte (0 = free)."
    )
    chunks: int = _field(1, _int, "Pipeline chunks for chained algorithms.")
    groups: str | None = _field(
        None,
        _optional(str),
        "JSON file {pg_name: [ranks]}; unnamed collectives span all ranks.",
    )


@dataclasses.dataclass(slots=True)
class ComputeConfig:
    """How compute ops are timed."""

    model: str = _field(
        "recorded",
        _choice(*COMPUTE_MODELS),
        "recorded, roofline, or roofline_if_missing.",
    )
    peak_flops: float = _field(0.0, units.flops, "Roofline peak FLOP/s.")
    memory_bandwidth: float = _field(
        0.0, units.bandwidth, "Roofline memory bandwidth."
    )
    efficiency: float = _field(1.0, _float, "Fraction of peak reached.")
    scale: float = _field(1.0, _float, "Multiplies every compute duration.")
    rank_scale: dict[int, float] = _field({}, _float, "Per-rank factor.")
    name_scale: dict[str, float] = _field(
        {}, _float, "Factor for ops whose name matches the regex key."
    )
    jitter: float = _field(
        0.0, _float, "Sigma of mean-1 log-normal noise on durations."
    )


@dataclasses.dataclass(slots=True)
class NoiseConfig:
    """OS noise and network degradation."""

    frequency: float = _field(0.0, units.rate_per_second, "Detours per s.")
    duration: float = _field(0.0, units.time_ns, "Mean detour length.")
    distribution: str = _field(
        "fixed", _choice("fixed", "exponential"), "Detour lengths."
    )
    nic_scale: dict[int, float] = _field(
        {}, _float, "Per-rank NIC bandwidth factor."
    )
    degraded_fraction: float = _field(
        0.0, _float, "flow: share of links slowed down."
    )
    degraded_factor: float = _field(
        1.0, _float, "flow: their bandwidth factor."
    )


_SECTIONS = {
    "loggops": LogGOPSConfig,
    "alpha_beta": AlphaBetaConfig,
    "topology": TopologyConfig,
    "collectives": CollectiveConfig,
    "compute": ComputeConfig,
    "noise": NoiseConfig,
}


@dataclasses.dataclass(slots=True)
class SimConfig:
    """Everything a simulation run needs besides the graph."""

    model: str = _field("loggops", _choice(*NETWORK_MODELS), "Network model.")
    seed: int = _field(1, _int, "Seed of noise and jitter.")
    world_size: int = _field(0, _int, "Ranks; 0 = as many as the trace has.")
    ranks_per_node: int = _field(1, _int, "Ranks sharing a node.")
    max_events: int = _field(0, _int, "Stop after this many events (0 = no).")
    loggops: LogGOPSConfig = dataclasses.field(default_factory=LogGOPSConfig)
    alpha_beta: AlphaBetaConfig = dataclasses.field(
        default_factory=AlphaBetaConfig
    )
    topology: TopologyConfig = dataclasses.field(default_factory=TopologyConfig)
    collectives: CollectiveConfig = dataclasses.field(
        default_factory=CollectiveConfig
    )
    compute: ComputeConfig = dataclasses.field(default_factory=ComputeConfig)
    noise: NoiseConfig = dataclasses.field(default_factory=NoiseConfig)

    def with_overrides(self, overrides: Mapping[str, Any]) -> SimConfig:
        """Returns a copy with ``section.key`` (or ``key``) values set.

        Dict-valued fields take one more part: ``compute.rank_scale.3``.

        Raises:
            ValueError: An unknown key or a value that does not parse.
        """
        config = copy.deepcopy(self)
        for path, value in overrides.items():
            _set_path(config, path, value)
        return config

    def to_dict(self) -> dict[str, Any]:
        """Returns the settings as plain data (for reports and JSON)."""
        return dataclasses.asdict(self)

    def to_native(self, world_size: int) -> dict[str, Any]:
        """Returns the flat parameter dict ``cta._native.simulate`` reads."""
        world = max(world_size, self.world_size)
        lg, ab, topo = self.loggops, self.alpha_beta, self.topology
        params: dict[str, Any] = {
            "model": self.model,
            "world_size": world,
            "ranks_per_node": max(1, self.ranks_per_node),
            "L": lg.L,
            "o": lg.o,
            "g": lg.g,
            "G": lg.G,
            "O": lg.O,
            "S": lg.S,
            "rendezvous": lg.rendezvous,
            "alpha": ab.alpha,
            "beta": ab.beta,
            "topology": topo.kind,
            "link_bandwidth": topo.link_bandwidth,
            "link_latency": topo.link_latency,
            "intra_bandwidth": topo.intra_bandwidth,
            "intra_latency": topo.intra_latency,
            "global_bandwidth": topo.global_bandwidth,
            "dims": list(topo.dims),
            "hosts_per_leaf": topo.hosts_per_leaf,
            "spines": topo.spines,
            "oversubscription": topo.oversubscription,
            "hosts_per_router": topo.hosts_per_router,
            "routers_per_group": topo.routers_per_group,
            "algorithms": [
                self.collectives.algorithms.get(t, DEFAULT_ALGORITHMS[t])
                for t in COLLECTIVE_TYPES
            ],
            "reduce_ns_per_byte": self.collectives.reduce_ns_per_byte,
            "chunks": max(1, self.collectives.chunks),
            "data_completion": self.collectives.completion == "outputs",
            "collective_cost": self.collectives.cost,
            "noise_frequency": self.noise.frequency / 1e9,
            "noise_duration": self.noise.duration,
            "noise_exponential": self.noise.distribution == "exponential",
            "degraded_fraction": self.noise.degraded_fraction,
            "degraded_factor": self.noise.degraded_factor,
            "seed": self.seed,
            "max_events": self.max_events,
        }
        for name in ("L", "o", "g", "G", "O"):
            value = getattr(lg, f"intra_{name}")
            if value is not None:
                params[f"intra_{name}"] = value
        if ab.intra_alpha is not None:
            params["intra_alpha"] = ab.intra_alpha
        if ab.intra_beta is not None:
            params["intra_beta"] = ab.intra_beta
        if self.noise.nic_scale:
            ranks = max(world, max(self.noise.nic_scale) + 1)
            scale = [1.0] * ranks
            for rank, factor in self.noise.nic_scale.items():
                scale[rank] = factor
            params["nic_scale"] = scale
        return params


def _fields(obj: Any) -> dict[str, dataclasses.Field[Any]]:
    return {f.name: f for f in dataclasses.fields(obj)}


def _set_path(config: SimConfig, path: str, value: Any) -> None:
    """Sets one dotted key on the config, parsing the value."""
    parts = path.split(".")
    target: Any = config
    while True:
        fields = _fields(target)
        name = parts[0]
        if name not in fields:
            raise ValueError(
                f"unknown setting {path!r}" + _suggestion(name, fields)
            )
        field = fields[name]
        current = getattr(target, name)
        if dataclasses.is_dataclass(current):
            if len(parts) == 1:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path!r} is a section; set its keys")
                for key, item in value.items():
                    _set_path(config, f"{path}.{key}", item)
                return
            target, parts = current, parts[1:]
            continue
        parse = field.metadata.get("parse", lambda v: v)
        if isinstance(current, dict):
            key_type = int if name in ("rank_scale", "nic_scale") else str
            if len(parts) == 1:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path!r} takes a table of values")
                current.clear()
                items: list[tuple[Any, Any]] = list(value.items())
            elif len(parts) == 2:
                items = [(parts[1], value)]
            else:
                raise ValueError(f"unknown setting {path!r}")
            for key, item in items:
                if name == "algorithms" and key not in COLLECTIVE_TYPES:
                    raise ValueError(
                        f"unknown collective type {key!r}"
                        + _suggestion(str(key), COLLECTIVE_TYPES)
                    )
                current[key_type(key)] = parse(item)
            return
        if len(parts) != 1:
            raise ValueError(f"unknown setting {path!r}")
        try:
            setattr(target, name, parse(value))
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
        return


def load_config(
    path: str | pathlib.Path | None = None,
    *,
    preset: str | None = None,
    overrides: Mapping[str, Any] | Iterable[str] = (),
    model: str | None = None,
) -> SimConfig:
    """Builds a config: defaults, preset, TOML file, overrides, model.

    Args:
        path: A TOML file; its top-level ``preset`` key, if any, is applied
            before its other settings (an explicit ``preset`` argument wins).
        preset: Name of a preset (``cta simulate presets`` lists them).
        overrides: ``{"loggops.L": "2us"}`` or ``["loggops.L=2us", ...]``.
        model: Network model to use, overriding the file and preset.

    Returns:
        The config.

    Raises:
        ValueError: Unknown preset, key or unparseable value.
        FileNotFoundError: ``path`` does not exist.
    """
    from cta.sim import presets  # presets builds on this module

    data: dict[str, Any] = {}
    if path is not None:
        with pathlib.Path(path).open("rb") as fh:
            data = tomllib.load(fh)
    preset_name = preset or data.pop("preset", None)
    data.pop("preset", None)
    config = SimConfig()
    if preset_name:
        config = presets.apply(config, str(preset_name))
    config = config.with_overrides(data)
    if not isinstance(overrides, Mapping):
        overrides = parse_assignments(overrides)
    config = config.with_overrides(overrides)
    if model is not None:
        config = config.with_overrides({"model": model})
    return config


def parse_assignments(assignments: Iterable[str]) -> dict[str, str]:
    """Reads ``key=value`` strings into a dict, in order.

    Raises:
        ValueError: An item has no ``=``.
    """
    out: dict[str, str] = {}
    for item in assignments:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected key=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def describe() -> list[tuple[str, str, str]]:
    """Returns (key, default, help) for every setting, for ``--help`` text."""
    rows: list[tuple[str, str, str]] = []

    def walk(obj: Any, prefix: str) -> None:
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if dataclasses.is_dataclass(value):
                walk(value, f"{prefix}{field.name}.")
            else:
                rows.append(
                    (
                        prefix + field.name,
                        repr(value),
                        field.metadata.get("help", ""),
                    )
                )

    walk(SimConfig(), "")
    return rows
