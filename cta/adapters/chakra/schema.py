from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chakra.schema.protobuf import et_def_pb2

# Official MLCommons Chakra schema from the pinned third_party/chakra submodule (PyPI's mlc-chakra 1.0.0
# predates STORAGE_NODE / DATA_OP_NODE).
NodeMsg = et_def_pb2.Node
GlobalMetadataMsg = et_def_pb2.GlobalMetadata
AttributeProtoMsg = et_def_pb2.AttributeProto

COMP_NODE = et_def_pb2.COMP_NODE
COMM_SEND_NODE = et_def_pb2.COMM_SEND_NODE
COMM_RECV_NODE = et_def_pb2.COMM_RECV_NODE
COMM_COLL_NODE = et_def_pb2.COMM_COLL_NODE
STORAGE_NODE = et_def_pb2.STORAGE_NODE
DATA_OP_NODE = et_def_pb2.DATA_OP_NODE

NODE_TYPE_CLASS: dict[int, str] = {
    et_def_pb2.MEM_LOAD_NODE: "memory",
    et_def_pb2.MEM_STORE_NODE: "memory",
    COMP_NODE: "compute",
    COMM_SEND_NODE: "comm_send",
    COMM_RECV_NODE: "comm_recv",
    COMM_COLL_NODE: "comm_coll",
    STORAGE_NODE: "storage",
    DATA_OP_NODE: "data_op",
}

COLL_TYPE_NAMES: dict[int, str] = {v.number: v.name for v in et_def_pb2.CollectiveCommType.DESCRIPTOR.values}

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
    meta = GlobalMetadataMsg()
    meta.ParseFromString(bytes(data[pos : pos + meta_len]))
    pos += meta_len

    def _gen(cur: int) -> Iterator[Any]:
        end = len(data)
        while cur < end:
            msg_len, cur = decode_varint(data, cur)
            node = NodeMsg()
            node.ParseFromString(bytes(data[cur : cur + msg_len]))
            cur += msg_len
            yield node

    return (str(meta.version), _gen(pos))
