"""Tests of the C++ Chakra .et reader (`cta._native.read_chakra`).

The tables the adapter builds from each fixture were hashed with the Python
reader this code replaced, so a mismatch means the adapter no longer produces
what it did. The hand-written fixture holds every case the generators never
write: each attribute value kind, repeated attributes, collectives given as a
number, a name or a mix, unknown enum values, storage nodes, negative and
64-bit values, and files that are empty or hold only metadata.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np
import polars as pl
import pytest

from cta import _native
from cta import models
from cta.adapters.chakra import reader
from cta.synth import chakra as synth_chakra
from cta.synth import chakra_proto
from cta.synth import training


def _attr(name: str, **value: Any) -> Any:
    return chakra_proto.AttributeProtoMsg(name=name, **value)


def _node(
    node_id: int,
    name: str,
    node_type: int,
    *,
    attrs: tuple[Any, ...] = (),
    data_deps: tuple[int, ...] = (),
    ctrl_deps: tuple[int, ...] = (),
    duration_micros: int = 1,
) -> Any:
    return chakra_proto.NodeMsg(
        id=node_id,
        name=name,
        type=node_type,
        duration_micros=duration_micros,
        attr=list(attrs),
        data_deps=list(data_deps),
        ctrl_deps=list(ctrl_deps),
    )


def adversarial_nodes() -> list[Any]:
    """Returns nodes that exercise every branch of the reader."""
    pb2 = chakra_proto.et_def_pb2()
    storage_fs = _node(60, "fs", chakra_proto.STORAGE_NODE)
    storage_fs.storage.fs.size_bytes = 4096
    storage_block = _node(61, "block", chakra_proto.STORAGE_NODE)
    storage_block.storage.block.nr_sectors = 8
    storage_page = _node(62, "page", chakra_proto.STORAGE_NODE)
    storage_page.storage.page.nr_pages = 3
    storage_sized = _node(
        63,
        "sized",
        chakra_proto.STORAGE_NODE,
        attrs=(_attr("comm_size", int64_val=100),),
    )
    storage_sized.storage.fs.size_bytes = 5000
    storage_empty = _node(64, "bare storage", chakra_proto.STORAGE_NODE)
    return [
        _node(
            1,
            "compute ü名前",
            chakra_proto.COMP_NODE,
            attrs=(
                _attr("num_ops", uint64_val=2**40),
                _attr("tensor_size", int32_val=1024),
            ),
            duration_micros=2**40,
        ),
        _node(
            2,
            "ar_int",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", int32_val=0),
                _attr("comm_size", int64_val=4096),
            ),
            data_deps=(1, 1, 5),
            ctrl_deps=(9, 1),
        ),
        _node(
            3,
            "a2a",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", sint64_val=6),
                _attr("comm_size", uint64_val=8000),
            ),
        ),
        _node(
            4,
            "unknown enum",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", fixed32_val=99),
                _attr("comm_size", sfixed64_val=777),
            ),
        ),
        _node(
            5,
            "text type",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", string_val="all_gather"),
                _attr("tensor_size", int64_val=500),
            ),
        ),
        _node(
            6,
            "zero comm size shadows tensor size",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_size", int32_val=0),
                _attr("tensor_size", int32_val=999),
            ),
        ),
        _node(
            7,
            "bool and float",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", bool_val=True),
                _attr("comm_size", double_val=123.9),
            ),
        ),
        _node(
            8,
            "float type, text size",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", double_val=1.0),
                _attr("comm_size", string_val="2048"),
            ),
        ),
        _node(
            9,
            "send",
            chakra_proto.COMM_SEND_NODE,
            attrs=(
                _attr("comm_dst", int32_val=3),
                _attr("comm_tag", uint32_val=77),
                _attr("comm_size", int64_val=64),
            ),
        ),
        _node(
            10,
            "recv",
            chakra_proto.COMM_RECV_NODE,
            attrs=(_attr("comm_src", sint32_val=2),),
        ),
        _node(
            11,
            "send to nowhere",
            chakra_proto.COMM_SEND_NODE,
            attrs=(
                _attr("comm_dst", int32_val=-5),
                _attr("comm_size", int32_val=8),
            ),
        ),
        _node(
            12,
            "repeated attributes",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", int32_val=0),
                _attr("comm_type", string_val="reduce"),
                _attr("comm_size", int32_val=10),
                _attr("comm_size", int32_val=20),
                _attr("doc", string_val="ignored", doc_string="x"),
            ),
        ),
        storage_fs,
        storage_block,
        storage_page,
        storage_sized,
        storage_empty,
        _node(70, "data op", chakra_proto.DATA_OP_NODE, data_deps=(60,)),
        _node(71, "load", chakra_proto.MEM_LOAD_NODE),
        _node(72, "store", chakra_proto.MEM_STORE_NODE),
        _node(73, "unknown type", 99),
        _node(74, "invalid type", 0),
        _node(
            80,
            "negative values",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_size", int64_val=-5),
                _attr("comm_tag", int32_val=-1),
                _attr("num_ops", int64_val=-3),
            ),
        ),
        _node(
            81,
            "wide values",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_size", uint64_val=2**63 + 5),
                _attr("num_ops", uint64_val=2**64 - 1),
            ),
        ),
        _node(2**63, "", chakra_proto.COMP_NODE),
        pb2.Node(id=90, name="no attributes, no type"),
    ]


def write_et(
    path: pathlib.Path, nodes: list[Any], version: str = "0.0.4"
) -> pathlib.Path:
    """Writes nodes as a length-delimited .et file."""
    out = bytearray()
    for message in (chakra_proto.GlobalMetadataMsg(version=version), *nodes):
        payload = message.SerializeToString()
        out += chakra_proto.encode_varint(len(payload)) + payload
    path.write_bytes(bytes(out))
    return path


def _fixtures(tmp_path: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Writes every fixture and returns its files by fixture name."""
    nodes = adversarial_nodes()
    adv = tmp_path / "adv"
    adv.mkdir()
    # Sorted by name, rank10 comes before rank2.
    files = [
        write_et(adv / "t.rank1.et", nodes),
        write_et(adv / "t.rank10.et", nodes[:5]),
        write_et(adv / "t.rank2.et", []),
        adv / "t.rank4.et",
    ]
    files[3].write_bytes(b"")
    compat = synth_chakra.write_run(
        tmp_path / "compat",
        world_size=3,
        steps=4,
        straggler_rank=1,
        straggler_factor=2.5,
    )
    spec = training.TrainingSpec(
        dp=3,
        pp=2,
        tp=2,
        microbatches=2,
        layers_per_stage=2,
        steps=2,
        moe_every=2,
        ep_size=3,
    )
    training.write_chakra(tmp_path / "train", spec)
    return {
        "adversarial": files,
        "compat": compat,
        "training": sorted((tmp_path / "train").glob("*.et")),
    }


def _tables(
    paths: list[pathlib.Path], out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests the files as one artifact and reads the tables back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(
        format="chakra", paths=tuple(paths), world_size=len(paths)
    )
    written = reader.ChakraAdapter().ingest(artifact, out_dir)
    return {name: pl.read_parquet(p) for name, p in written.items()}


def table_digest(tables: dict[str, pl.DataFrame]) -> str:
    """Returns a hash of the three tables, traffic sorted by rank pair."""
    digest = hashlib.sha256()
    for name in ("nodes", "edges", "traffic"):
        table = tables[name]
        if name == "traffic":
            table = table.sort("src_rank", "dst_rank")
        digest.update(name.encode())
        digest.update(table.write_csv().encode())
    return digest.hexdigest()


# Hashes of the tables the Python reader produced from each fixture.
GOLDEN: dict[str, str] = {
    "adversarial": (
        "e3a35a60a4f7166842f456f5b288b1c41e9615ec628a988024ecfe60cb7aa44d"
    ),
    "compat": (
        "3be9bee39c945854eae3cfc72198bead08a900673549b5b7df1f3c6744b1d2ae"
    ),
    "training": (
        "0425df17cc48df17cea13789f2490300209c5e382ab3d39159727b2d917a83a8"
    ),
}


def test_tables_match_the_python_reader_they_replaced(
    tmp_path: pathlib.Path,
) -> None:
    actual = {
        name: table_digest(_tables(files, tmp_path / f"out_{name}"))
        for name, files in _fixtures(tmp_path).items()
    }
    assert actual == GOLDEN


def test_rows_follow_the_rules_of_the_python_reader(
    tmp_path: pathlib.Path,
) -> None:
    files = _fixtures(tmp_path)["adversarial"]
    nodes = _tables(files, tmp_path / "out")["nodes"]
    first = nodes.filter(pl.col("rank") == 1)
    by_name = {row["name"]: row for row in first.iter_rows(named=True)}

    assert by_name["ar_int"]["comm_type"] == "ALL_REDUCE"
    assert by_name["a2a"]["comm_type"] == "ALL_TO_ALL"
    assert by_name["unknown enum"]["comm_type"] == "COLL_99"
    assert by_name["text type"]["comm_type"] == "ALL_GATHER"
    assert by_name["text type"]["size_bytes"] == 500
    # A present comm_size, even 0, hides tensor_size.
    assert by_name["zero comm size shadows tensor size"]["size_bytes"] == 0
    # A bool counts as an integer; a float is not a collective type.
    assert by_name["bool and float"]["comm_type"] == "REDUCE"
    assert by_name["bool and float"]["size_bytes"] == 123
    assert by_name["float type, text size"]["comm_type"] == ""
    assert by_name["float type, text size"]["size_bytes"] == 2048
    # The later of two equal attribute names wins.
    assert by_name["repeated attributes"]["comm_type"] == "REDUCE"
    assert by_name["repeated attributes"]["size_bytes"] == 20
    # Sends and receives default their own side to the file's rank.
    assert by_name["send"]["src_rank"] == 1
    assert by_name["send"]["dst_rank"] == 3
    assert by_name["send"]["message_tag"] == 77
    assert by_name["send"]["comm_type"] == "SEND"
    assert by_name["recv"]["dst_rank"] == 1
    assert by_name["recv"]["src_rank"] == 2
    assert by_name["recv"]["comm_type"] == "RECV"
    assert by_name["send to nowhere"]["dst_rank"] == -5
    # Storage nodes take their size from the storage detail unless an
    # attribute gives one.
    assert by_name["fs"]["size_bytes"] == 4096
    assert by_name["block"]["size_bytes"] == 8 * 512
    assert by_name["page"]["size_bytes"] == 3 * 4096
    assert by_name["sized"]["size_bytes"] == 100
    assert by_name["bare storage"]["size_bytes"] == 0
    # Node types.
    assert by_name["data op"]["op_class"] == "data_op"
    assert by_name["load"]["op_class"] == "memory"
    assert by_name["store"]["op_class"] == "memory"
    assert by_name["unknown type"]["op_class"] == "other"
    assert by_name["invalid type"]["op_class"] == "other"
    # Negative sizes, tags and counts become 0; wide ones stay exact.
    negative = by_name["negative values"]
    assert (negative["size_bytes"], negative["message_tag"]) == (0, 0)
    assert negative["num_ops"] == 0
    wide = by_name["wide values"]
    assert wide["size_bytes"] == 2**63 + 5
    assert wide["num_ops"] == 2**64 - 1
    assert by_name["compute ü名前"]["duration_ns"] == 2**40 * 1000
    # Ranks come from the file names, not their order.
    assert sorted(nodes["rank"].unique().to_list()) == [1, 10]


def test_data_dependencies_come_before_control_dependencies(
    tmp_path: pathlib.Path,
) -> None:
    files = _fixtures(tmp_path)["adversarial"]
    edges = _tables(files, tmp_path / "out")["edges"]
    into_two = edges.filter(
        (pl.col("dst_id") == 2) & (pl.col("rank") == 1)
    ).select("src_id", "dep_kind")
    assert into_two.rows() == [
        (1, "data"),
        (1, "data"),
        (5, "data"),
        (9, "ctrl"),
        (1, "ctrl"),
    ]


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    return all(
        np.array_equal(v, b[k]) if isinstance(v, np.ndarray) else v == b[k]
        for k, v in a.items()
    )


def test_result_does_not_depend_on_the_thread_count(
    tmp_path: pathlib.Path,
) -> None:
    for name, files in _fixtures(tmp_path).items():
        paths = [str(p) for p in files]
        ranks = list(range(len(paths)))
        one = _native.read_chakra(paths, ranks, 1)
        for threads in (2, 5, 16):
            assert _same(one, _native.read_chakra(paths, ranks, threads)), (
                name,
                threads,
            )


def test_an_empty_list_of_files_gives_no_rows() -> None:
    columns = _native.read_chakra([], [], 4)
    assert len(columns["node_id"]) == 0
    assert columns["names"] == []


@pytest.mark.parametrize(
    "damage",
    [
        lambda raw: raw[:-3],  # cut inside the last node
        lambda raw: raw[:1],  # cut inside the metadata
        lambda raw: raw + b"\xff",  # a length with no message
        lambda raw: b"\x05abc",  # a length longer than the file
    ],
)
def test_truncated_or_corrupt_files_are_rejected(
    tmp_path: pathlib.Path, damage: Any
) -> None:
    good = write_et(tmp_path / "good.et", adversarial_nodes()[:3])
    bad = tmp_path / "bad.et"
    bad.write_bytes(damage(good.read_bytes()))
    with pytest.raises(ValueError, match="bad.et"):
        _native.read_chakra([str(good), str(bad)], [0, 1], 2)


def test_a_missing_file_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError, match="cannot open"):
        _native.read_chakra([str(tmp_path / "none.et")], [0], 1)


def test_paths_and_ranks_must_match(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        _native.read_chakra([str(tmp_path / "a.et")], [], 1)


def test_collective_traffic_is_the_sum_over_collectives(
    tmp_path: pathlib.Path,
) -> None:
    """Counting collectives per size must give what summing each would."""
    nodes = [
        _node(
            i,
            f"c{i}",
            chakra_proto.COMM_COLL_NODE,
            attrs=(
                _attr("comm_type", int32_val=kind),
                _attr("comm_size", int64_val=size),
            ),
        )
        for i, (kind, size) in enumerate(
            [(0, 1000), (0, 1000), (0, 7), (2, 100), (6, 1003), (6, 1003)]
        )
    ]
    files = [write_et(tmp_path / f"t.rank{r}.et", nodes) for r in range(3)]
    traffic = _tables(files, tmp_path / "out")["traffic"]
    world = 3
    expected: dict[tuple[int, int], list[int]] = {}
    for rank in range(world):
        nxt = (rank + 1) % world
        cell = expected.setdefault((rank, nxt), [0, 0])
        for size in (1000, 1000, 7):  # ring all-reduce
            cell[0] += 2 * (world - 1)
            cell[1] += (2 * (world - 1) * size) // world
        cell[0] += world - 1  # ring all-gather
        cell[1] += ((world - 1) * 100) // world
        for peer in range(world):
            if peer != rank:  # all-to-all, twice
                peer_cell = expected.setdefault((rank, peer), [0, 0])
                peer_cell[0] += 2
                peer_cell[1] += 2 * (1003 // world)
    actual = {
        (r["src_rank"], r["dst_rank"]): [r["message_count"], r["total_bytes"]]
        for r in traffic.iter_rows(named=True)
    }
    assert actual == expected
    assert traffic.rows() == sorted(traffic.rows())  # sorted by rank pair
