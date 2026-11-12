from __future__ import annotations

import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Official MLCommons Chakra schema from the pinned third_party/chakra submodule (PyPI's mlc-chakra 1.0.0
# predates STORAGE_NODE / DATA_OP_NODE). The generated module is imported lazily: protobuf is only needed once a
# .et file is actually parsed or written.

# NodeType enum numbers (stable protobuf wire values; asserted against the generated enum in the tests).
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
    """The generated Chakra protobuf module (imported on first use)."""
    from chakra.schema.protobuf import et_def_pb2 as mod

    return mod


@functools.lru_cache(maxsize=1)
def coll_type_names() -> dict[int, str]:
    return {v.number: v.name for v in et_def_pb2().CollectiveCommType.DESCRIPTOR.values}


_LAZY_MESSAGES = {"NodeMsg": "Node", "GlobalMetadataMsg": "GlobalMetadata", "AttributeProtoMsg": "AttributeProto"}


def __getattr__(name: str) -> Any:  # PEP 562: NodeMsg / GlobalMetadataMsg / AttributeProtoMsg
    if name in _LAZY_MESSAGES:
        return getattr(et_def_pb2(), _LAZY_MESSAGES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_SECTOR_BYTES = 512
_PAGE_BYTES = 4096


def storage_size_bytes(node: Any) -> int:
    """Bytes moved by a STORAGE_NODE, from its typed fs/page/block detail (0 if not recorded)."""
    detail = node.storage.WhichOneof("detail")
    if detail == "fs":
        return int(node.storage.fs.size_bytes)
    if detail == "block":
        return int(node.storage.block.nr_sectors) * _SECTOR_BYTES
    if detail == "page":
        return int(node.storage.page.nr_pages) * _PAGE_BYTES
    return 0


def encode_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def decode_varint(buf: memoryview | bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    n = len(buf)
    while pos < n:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise EOFError("Truncated protobuf varint in Chakra .et stream")


def extract_attr_value(attr: Any) -> Any:
    which = attr.WhichOneof("value")
    return getattr(attr, which) if which else None


def iter_et_messages(path: Path) -> tuple[str, Iterator[Any]]:
    data = memoryview(path.read_bytes())
    if len(data) == 0:
        return ("", iter(()))
    meta_len, pos = decode_varint(data, 0)
    meta = et_def_pb2().GlobalMetadata()
    meta.ParseFromString(bytes(data[pos : pos + meta_len]))
    pos += meta_len

    node_cls = et_def_pb2().Node

    def _gen(cur: int) -> Iterator[Any]:
        end = len(data)
        while cur < end:
            msg_len, cur = decode_varint(data, cur)
            node = node_cls()
            node.ParseFromString(bytes(data[cur : cur + msg_len]))
            cur += msg_len
            yield node

    return (str(meta.version), _gen(pos))
