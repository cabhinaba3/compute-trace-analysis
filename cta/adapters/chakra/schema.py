from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

# NodeType enum constants (MLCommons Chakra et_def.proto)
INVALID_NODE = 0
METADATA_NODE = 1
MEM_LOAD_NODE = 2
MEM_STORE_NODE = 3
COMP_NODE = 4
COMM_SEND_NODE = 5
COMM_RECV_NODE = 6
COMM_COLL_NODE = 7

NODE_TYPE_CLASS: dict[int, str] = {
    MEM_LOAD_NODE: "memory",
    MEM_STORE_NODE: "memory",
    COMP_NODE: "compute",
    COMM_SEND_NODE: "comm_send",
    COMM_RECV_NODE: "comm_recv",
    COMM_COLL_NODE: "comm_coll",
}

COLL_TYPE_NAMES: dict[int, str] = {
    0: "ALL_REDUCE",
    1: "REDUCE",
    2: "ALL_GATHER",
    3: "GATHER",
    4: "SCATTER",
    5: "BROADCAST",
    6: "ALL_TO_ALL",
    7: "REDUCE_SCATTER",
    8: "REDUCE_SCATTER_BLOCK",
    9: "BARRIER",
}


def _build_chakra_classes() -> tuple[Any, Any, Any]:
    """Build native C/upb-backed Protobuf classes dynamically (100% immune to protoc version skew)."""
    fd = descriptor_pb2.FileDescriptorProto()
    fd.name = "chakra_et_def.proto"
    fd.package = "ChakraProtoMsg"
    fd.syntax = "proto3"

    F = descriptor_pb2.FieldDescriptorProto

    def _add_field(
        msg: Any,
        name: str,
        num: int,
        ftype: Any,
        label: Any = F.LABEL_OPTIONAL,
        type_name: str = "",
        oneof_idx: int | None = None,
    ) -> None:
        f = msg.field.add()
        f.name = name
        f.number = num
        f.type = ftype
        f.label = label
        if type_name:
            f.type_name = type_name
        if oneof_idx is not None:
            f.oneof_index = oneof_idx

    attr = fd.message_type.add()
    attr.name = "AttributeProto"
    attr.oneof_decl.add().name = "value"
    _add_field(attr, "name", 1, F.TYPE_STRING)
    _add_field(attr, "doc_string", 2, F.TYPE_STRING)
    _add_field(attr, "double_val", 3, F.TYPE_DOUBLE, oneof_idx=0)
    _add_field(attr, "float_val", 5, F.TYPE_FLOAT, oneof_idx=0)
    _add_field(attr, "int32_val", 7, F.TYPE_INT32, oneof_idx=0)
    _add_field(attr, "int64_val", 9, F.TYPE_INT64, oneof_idx=0)
    _add_field(attr, "uint32_val", 11, F.TYPE_UINT32, oneof_idx=0)
    _add_field(attr, "uint64_val", 13, F.TYPE_UINT64, oneof_idx=0)
    _add_field(attr, "bool_val", 27, F.TYPE_BOOL, oneof_idx=0)
    _add_field(attr, "string_val", 29, F.TYPE_STRING, oneof_idx=0)

    gmeta = fd.message_type.add()
    gmeta.name = "GlobalMetadata"
    _add_field(gmeta, "version", 1, F.TYPE_STRING)
    _add_field(gmeta, "attr", 2, F.TYPE_MESSAGE, F.LABEL_REPEATED, ".ChakraProtoMsg.AttributeProto")

    io_info = fd.message_type.add()
    io_info.name = "IOInfo"
    _add_field(io_info, "values", 1, F.TYPE_STRING)
    _add_field(io_info, "shapes", 2, F.TYPE_STRING)
    _add_field(io_info, "types", 3, F.TYPE_STRING)

    node = fd.message_type.add()
    node.name = "Node"
    _add_field(node, "id", 1, F.TYPE_UINT64)
    _add_field(node, "name", 2, F.TYPE_STRING)
    _add_field(node, "type", 3, F.TYPE_INT32)
    _add_field(node, "ctrl_deps", 4, F.TYPE_UINT64, F.LABEL_REPEATED)
    _add_field(node, "data_deps", 5, F.TYPE_UINT64, F.LABEL_REPEATED)
    _add_field(node, "start_time_micros", 6, F.TYPE_UINT64)
    _add_field(node, "duration_micros", 7, F.TYPE_UINT64)
    _add_field(node, "inputs", 8, F.TYPE_MESSAGE, F.LABEL_OPTIONAL, ".ChakraProtoMsg.IOInfo")
    _add_field(node, "outputs", 9, F.TYPE_MESSAGE, F.LABEL_OPTIONAL, ".ChakraProtoMsg.IOInfo")
    _add_field(node, "attr", 10, F.TYPE_MESSAGE, F.LABEL_REPEATED, ".ChakraProtoMsg.AttributeProto")

    pool = descriptor_pool.DescriptorPool()
    file_desc = pool.AddSerializedFile(fd.SerializeToString())
    return (
        message_factory.GetMessageClass(file_desc.message_types_by_name["GlobalMetadata"]),
        message_factory.GetMessageClass(file_desc.message_types_by_name["Node"]),
        message_factory.GetMessageClass(file_desc.message_types_by_name["AttributeProto"]),
    )


GlobalMetadataMsg, NodeMsg, AttributeProtoMsg = _build_chakra_classes()


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
