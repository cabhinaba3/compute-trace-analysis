"""Chakra execution trace schema access, for writing .et files in tests.

Reading is done by the native reader (``cta._native.read_chakra_et``).

The official MLCommons Chakra schema comes from the pinned third_party/chakra
submodule (PyPI's mlc-chakra 1.0.0 predates STORAGE_NODE and DATA_OP_NODE).
The generated protobuf module is imported lazily: protobuf is only needed once
a .et file is written.

The protobuf message classes are also available as the module attributes
``NodeMsg``, ``GlobalMetadataMsg`` and ``AttributeProtoMsg``, resolved on first
access.
"""

from __future__ import annotations

import functools
from typing import Any

# NodeType enum numbers: stable protobuf wire values, asserted against the
# generated enum in the tests.
MEM_LOAD_NODE = 2
MEM_STORE_NODE = 3
COMP_NODE = 4
COMM_SEND_NODE = 5
COMM_RECV_NODE = 6
COMM_COLL_NODE = 7
STORAGE_NODE = 8
DATA_OP_NODE = 9


@functools.lru_cache(maxsize=1)
def et_def_pb2() -> Any:
    """Returns the generated Chakra protobuf module, importing it on first use.

    Returns:
        The ``chakra.schema.protobuf.et_def_pb2`` module.

    Raises:
        ImportError: The ``chakra`` extra (mlc-chakra and protobuf) is not
            installed.
    """
    try:
        from chakra.schema.protobuf import et_def_pb2 as generated_module
    except ImportError as exc:
        raise ImportError(
            "writing .et files needs the Chakra schema: "
            "pip install 'compute-trace-analysis[chakra]'"
        ) from exc

    return generated_module


# Lazily resolved module attributes (PEP 562): alias -> protobuf message name.
_MESSAGE_NAMES_BY_ALIAS = {
    "NodeMsg": "Node",
    "GlobalMetadataMsg": "GlobalMetadata",
    "AttributeProtoMsg": "AttributeProto",
}


def __getattr__(name: str) -> Any:
    """Resolves the lazy protobuf message aliases such as ``NodeMsg``.

    Args:
        name: The module attribute being looked up.

    Returns:
        The generated protobuf message class for the alias.

    Raises:
        AttributeError: If ``name`` is not one of the lazy message aliases.
    """
    if name in _MESSAGE_NAMES_BY_ALIAS:
        return getattr(et_def_pb2(), _MESSAGE_NAMES_BY_ALIAS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def encode_varint(value: int) -> bytes:
    """Encodes a non-negative integer as a protobuf base-128 varint.

    Args:
        value: The integer to encode.

    Returns:
        The varint bytes, least significant group first.
    """
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value & 0x7F)
    return bytes(encoded)
