"""Parses quantities with units into ns, bytes, GB/s, Hz and FLOP/s.

Numbers pass through as already in those units.
"""

from __future__ import annotations

import re

_QUANTITY = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµ/]*)\s*$"
)
# A bandwidth unit: SI or binary prefix, b/bit (bits) or B/byte, ps or /s.
_BANDWIDTH = re.compile(r"^([kKmMgGtT]?)(i?)(bit|byte|b|B)(ps|/s)$")
_TIME = {"": 1.0, "ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}
_SIZE = {
    "": 1.0, "b": 1.0, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
    "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
}  # fmt: skip
_RATE = {"": 1.0, "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
_FLOPS = {
    f"{prefix}{unit}": scale
    for prefix, scale in (("", 1.0), ("g", 1e9), ("t", 1e12), ("p", 1e15))
    for unit in ("flops", "flop/s")
} | {"": 1.0}


def _split(value: float | int | str, kind: str) -> tuple[float, str]:
    if isinstance(value, bool):
        raise ValueError(f"expected a {kind}, got {value!r}")
    if isinstance(value, int | float):
        return float(value), ""
    match = _QUANTITY.match(str(value))
    if match is None:
        raise ValueError(f"cannot read {value!r} as a {kind}")
    return float(match.group(1)), match.group(2)


def _parse(
    value: float | int | str, units: dict[str, float], kind: str
) -> float:
    number, unit = _split(value, kind)
    if unit.lower() not in units:
        raise ValueError(
            f"unknown {kind} unit {unit!r}; use one of "
            + ", ".join(k for k in units if k)
        )
    return number * units[unit.lower()]


def time_ns(value: float | int | str) -> float:
    """``"2.5us"`` -> 2500.0."""
    return _parse(value, _TIME, "time")


def size_bytes(value: float | int | str) -> int:
    """``"64KiB"`` -> 65536."""
    return round(_parse(value, _SIZE, "size"))


def bandwidth(value: float | int | str) -> float:
    """``"400Gbps"`` (bits) or ``"50GB/s"`` (bytes) -> 50.0 GB/s."""
    number, unit = _split(value, "bandwidth")
    if not unit:
        return number
    match = _BANDWIDTH.match(unit)
    if match is None:
        raise ValueError(f"cannot read {value!r} as a bandwidth, e.g. 400Gbps")
    prefix, binary, base, _ = match.groups()
    power = "kmgt".find(prefix.lower()) + 1 if prefix else 0
    scale = (1024.0 if binary else 1000.0) ** power
    return number * scale / (8 if base in ("b", "bit") else 1) / 1e9


def ns_per_byte(value: float | int | str) -> float:
    """Per-byte ns (``6``), or the inverse of a bandwidth (``"50GB/s"``)."""
    number, unit = _split(value, "per-byte time")
    return number if not unit else 1.0 / bandwidth(value)


def rate_per_second(value: float | int | str) -> float:
    """``"1kHz"`` -> 1000.0."""
    return _parse(value, _RATE, "frequency")


def flops(value: float | int | str) -> float:
    """``"989TFLOPS"`` -> 9.89e14."""
    return _parse(value, _FLOPS, "FLOP rate")
