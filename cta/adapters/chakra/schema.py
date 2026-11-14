"""Chakra execution trace schema access and length-delimited .et codec.

The official MLCommons Chakra schema comes from the pinned third_party/chakra
submodule (PyPI's mlc-chakra 1.0.0 predates STORAGE_NODE and DATA_OP_NODE).
The generated protobuf module is imported lazily: protobuf is only needed once
a .et file is actually parsed or written.

The protobuf message classes are also available as the module attributes
``NodeMsg``, ``GlobalMetadataMsg`` and ``AttributeProtoMsg``, resolved on first
access.
"""

from __future__ import annotations

from collections.abc import Iterator
import functools
import pathlib
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

NODE_TYPE_CLASS: dict[int, str] = {
    MEM_LOAD_NODE: "memory",
    MEM_STORE_NODE: "memory",
    COMP_NODE: "compute",
    COMM_SEND_NODE: "comm_send",
    COMM_RECV_NODE: "comm_recv",
    COMM_COLL_NODE: "comm_coll",
    STORAGE_NODE: "storage",
    DATA_OP_NODE: "data_op",
}


@functools.lru_cache(maxsize=1)
def et_def_pb2() -> Any:
    """Returns the generated Chakra protobuf module, importing it on first use.

    Returns:
        The ``chakra.schema.protobuf.et_def_pb2`` module.
    """
    from chakra.schema.protobuf import et_def_pb2 as generated_module

    return generated_module


@functools.lru_cache(maxsize=1)
def coll_type_names() -> dict[int, str]:
    """Returns the CollectiveCommType enum names keyed by their wire number.

    Returns:
        A mapping such as ``{0: "ALL_REDUCE", ...}`` read from the generated
        protobuf enum.
    """
    enum_values = et_def_pb2().CollectiveCommType.DESCRIPTOR.values
    return {enum_value.number: enum_value.name for enum_value in enum_values}


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


_SECTOR_BYTES = 512
_PAGE_BYTES = 4096


def storage_size_bytes(node: Any) -> int:
    """Returns the bytes moved by a STORAGE_NODE.

    The size comes from the node's storage detail: the ``fs`` op's
    ``size_bytes``, the ``block`` op's sector count or the ``page`` op's page
    count.

    Args:
        node: A Chakra ``Node`` message of type STORAGE_NODE.

    Returns:
        The number of bytes, or 0 if the node records no storage detail.
    """
    detail = node.storage.WhichOneof("detail")
    if detail == "fs":
        return int(node.storage.fs.size_bytes)
    if detail == "block":
        return int(node.storage.block.nr_sectors) * _SECTOR_BYTES
    if detail == "page":
        return int(node.storage.page.nr_pages) * _PAGE_BYTES
    return 0


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


def decode_varint(buf: memoryview | bytes, pos: int) -> tuple[int, int]:
    """Decodes a protobuf base-128 varint.

    Args:
        buf: The buffer holding the varint.
        pos: The offset of the varint's first byte in ``buf``.

    Returns:
        The decoded value and the offset of the first byte after the varint.

    Raises:
        EOFError: If ``buf`` ends before the varint's last byte.
    """
    result = 0
    shift = 0
    end = len(buf)
    while pos < end:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
    raise EOFError("Truncated protobuf varint in Chakra .et stream")


def extract_attr_value(attr: Any) -> Any:
    """Returns the value held by a Chakra ``AttributeProto``.

    Args:
        attr: An ``AttributeProto`` message.

    Returns:
        The value of whichever ``value`` oneof field is set, or None if none
        is.
    """
    value_field = attr.WhichOneof("value")
    return getattr(attr, value_field) if value_field else None


def iter_et_messages(path: pathlib.Path) -> tuple[str, Iterator[Any]]:
    """Reads a length-delimited Chakra .et file.

    The file holds a varint-prefixed ``GlobalMetadata`` message followed by
    varint-prefixed ``Node`` messages. The metadata is parsed eagerly; nodes
    are parsed lazily as the iterator is consumed.

    Args:
        path: The .et file to read.

    Returns:
        The schema version recorded in the metadata and an iterator over the
        ``Node`` messages. An empty file gives ``""`` and an empty iterator.

    Raises:
        EOFError: If the metadata length prefix is truncated. A truncated node
            length prefix raises it later, while iterating.
    """
    data = memoryview(path.read_bytes())
    if len(data) == 0:
        return ("", iter(()))
    metadata_length, pos = decode_varint(data, 0)
    metadata = et_def_pb2().GlobalMetadata()
    metadata.ParseFromString(bytes(data[pos : pos + metadata_length]))
    pos += metadata_length

    node_class = et_def_pb2().Node

    def _iter_nodes(offset: int) -> Iterator[Any]:
        end = len(data)
        while offset < end:
            message_length, offset = decode_varint(data, offset)
            node = node_class()
            node.ParseFromString(bytes(data[offset : offset + message_length]))
            offset += message_length
            yield node

    return (str(metadata.version), _iter_nodes(pos))
