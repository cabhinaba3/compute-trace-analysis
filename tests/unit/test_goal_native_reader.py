"""Tests of the C++ GOAL text reader (`cta._native.read_goal_text`).

The tables the adapter builds from each fixture were hashed with the Python
parser this code replaced, so a mismatch means the adapter no longer produces
what it did. The hand-written schedule holds what generated ones never do:
comments of every kind (one spanning a rank header), operations before the
first rank header and after a closing brace, labels reused across and within
ranks, forward and unknown dependencies, option orders the format does not
allow, upper-case keywords, every size unit, a rank declared twice, and each
kind of line ending.
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
from cta.adapters.goal import reader
from cta.synth import goal as synth_goal
from cta.synth import training

ADVERSARIAL = """\
num_ranks 5
l0: calc 5            // an op before any rank header: rank 0
rank 0 {
l1: calc 100 cpu 0
l2: send 1kb to 1 tag 7 cpu 1 nic 2
l3: recv 2MB from 1
l4: send 3gb to 2
l5: SEND 4tb TO 2 TAG 9
send 5b to 1
recv 6 from 1 tag 3
calc 50
l6: calc 7 // trailing comment
l7: calc 8 # hash comment
l8: calc 9 /* inline */
l9: calc /* mid */ 10
l10: calc 11 /* open
still a comment l11: calc 12
close */ l12: calc 13
lé世: calc 14
l2 requires l1
l3 irequires l1
l3 requires l9999
l1 requires l3
l20 requires l1
l21: calc 1
l21 requires l22
l22: calc 2
   /* only a comment */
x: bogus op line
l13: calc 1 tag 3
l14: send 8b to 1 nic 1 cpu 1
l15: send 8 b to 1
l16: send 8b to 1 tag 2 cpu 0 nic 1
l17: Calc 4 CPU 1 NIC 0
L18 REQUIRES l1
}
l19: calc 3
l19 requires l19
rank 1 {
l1: calc 7
l1: calc 8
l2: calc 9
l2 requires l1
l1 requires l1
}
rank 2{
l1: recv 100b from 0 tag 7
}
rank 3
l1: calc 1
rank 1 {
l1: calc 2
}
/* a comment that spans
rank 9 {
l1: calc 4
*/ rank 4 {
l1: calc 5
}
rank 5 { // the last line has no newline
l1: calc 6"""


def _write(path: pathlib.Path, text: str, newline: str = "\n") -> pathlib.Path:
    path.write_bytes(text.replace("\n", newline).encode())
    return path


def _fixtures(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Writes every fixture and returns them by name."""
    out = {
        "adversarial": _write(tmp_path / "a.goal", ADVERSARIAL),
        "adversarial_crlf": _write(tmp_path / "b.goal", ADVERSARIAL, "\r\n"),
        "adversarial_cr": _write(tmp_path / "c.goal", ADVERSARIAL, "\r"),
        "empty": _write(tmp_path / "e.goal", ""),
        "blank": _write(tmp_path / "f.goal", "\n  \n\t\n"),
        "comments_only": _write(
            tmp_path / "g.goal", "// nothing\n/* at all\n*/\n# here\n"
        ),
        "ring": synth_goal.write_goal_allreduce(
            tmp_path / "ring.goal", world_size=5, steps=3
        ),
        "direct": synth_goal.write_goal_allreduce(
            tmp_path / "direct.goal", world_size=4, steps=2, algorithm="direct"
        ),
    }
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
    training.write_goal_training(tmp_path / "train.goal", spec)
    out["training"] = tmp_path / "train.goal"
    return out


def _tables(
    path: pathlib.Path, out_dir: pathlib.Path
) -> dict[str, pl.DataFrame]:
    """Ingests one schedule and reads the tables back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = models.TraceArtifact(format="goal", paths=(path,), world_size=1)
    written = reader.GoalAdapter().ingest(artifact, out_dir)
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


# Hashes of the tables the Python parser produced from each fixture.
GOLDEN: dict[str, str] = {
    "adversarial": (
        "833a2fc5b21743525bb849ee8aef38a2ac60217f5686efd5092b24233f276a87"
    ),
    "adversarial_crlf": (
        "833a2fc5b21743525bb849ee8aef38a2ac60217f5686efd5092b24233f276a87"
    ),
    "adversarial_cr": (
        "833a2fc5b21743525bb849ee8aef38a2ac60217f5686efd5092b24233f276a87"
    ),
    "empty": (
        "999182b01da2ff92a939724e34aaee88e7ead485a4b9e5b48f1561dda40c63e7"
    ),
    "blank": (
        "999182b01da2ff92a939724e34aaee88e7ead485a4b9e5b48f1561dda40c63e7"
    ),
    "comments_only": (
        "999182b01da2ff92a939724e34aaee88e7ead485a4b9e5b48f1561dda40c63e7"
    ),
    "ring": (
        "a3fb1019dcc9679c9988ae38411cb840f725915e921615f8f2561c92adcd936a"
    ),
    "direct": (
        "f0a72ca92cdab4e0f5edaad8b91e4ded111ea67fb6efcd8131992fdab9d515a9"
    ),
    "training": (
        "25c021b823ded98b6b7711ebbcd1008869d8b8647b16d3ea832ac510732a6a6c"
    ),
}


def test_tables_match_the_python_parser_they_replaced(
    tmp_path: pathlib.Path,
) -> None:
    actual = {
        name: table_digest(_tables(path, tmp_path / f"out_{name}"))
        for name, path in _fixtures(tmp_path).items()
    }
    assert actual == GOLDEN


def test_rows_follow_the_rules_of_the_python_parser(
    tmp_path: pathlib.Path,
) -> None:
    nodes, edges = reader.parse_goal_text(_fixtures(tmp_path)["adversarial"])
    rank0 = nodes.filter(pl.col("rank") == 0)
    by_name = {row["name"]: row for row in rank0.iter_rows(named=True)}

    # An op before any header belongs to rank 0 and is numbered first.
    assert rank0["name"][0] == "l0"
    assert by_name["l2"]["size_bytes"] == 1024
    assert by_name["l2"]["message_tag"] == 7
    assert (by_name["l2"]["src_rank"], by_name["l2"]["dst_rank"]) == (0, 1)
    assert by_name["l3"]["size_bytes"] == 2 * 1024**2
    assert (by_name["l3"]["src_rank"], by_name["l3"]["dst_rank"]) == (1, 0)
    assert by_name["l4"]["size_bytes"] == 3 * 1024**3
    assert by_name["l5"]["size_bytes"] == 4  # "tb" counts as bytes
    assert by_name["l5"]["op_class"] == "comm_send"
    # Unlabelled ops are named after their class.
    assert rank0.filter(pl.col("name") == "comm_send").height == 1
    assert rank0.filter(pl.col("name") == "comm_recv").height == 1
    assert rank0.filter(pl.col("name") == "compute").height == 1
    # Comments come off every way a line can carry them.
    for label in ("l6", "l7", "l8", "l9", "l10", "l12", "lé世"):
        assert label in by_name, label
    assert "l11" not in by_name  # inside the block comment
    # Options in the wrong order, or not allowed, make the line no op.
    for label in ("l13", "l14", "l15"):
        assert label not in by_name, label
    assert by_name["l16"]["message_tag"] == 2
    assert by_name["l17"]["duration_ns"] == 4
    # After a closing brace the ids go on and the labels are forgotten.
    assert by_name["l19"]["node_id"] == rank0["node_id"].max()
    rank_one_ids = nodes.filter(pl.col("rank") == 1)["node_id"].to_list()
    assert rank_one_ids == [0, 1, 2, 0]  # the second "rank 1" starts over
    # A block comment may span a rank header, which then is no header.
    assert 9 not in nodes["rank"].to_list()
    assert 4 in nodes["rank"].to_list()
    assert 5 in nodes["rank"].to_list()

    rank0_edges = edges.filter(pl.col("rank") == 0)
    pairs = [
        (row["src_id"], row["dst_id"], row["dep_kind"])
        for row in rank0_edges.iter_rows(named=True)
    ]
    ids = {name: row["node_id"] for name, row in by_name.items()}
    assert (ids["l1"], ids["l2"], "requires") in pairs
    assert (ids["l1"], ids["l3"], "irequires") in pairs
    assert (ids["l3"], ids["l1"], "requires") in pairs  # backwards is allowed
    # A label defined after the closing brace can depend on itself.
    assert (ids["l19"], ids["l19"], "requires") in pairs
    # Unknown labels, forward references and upper-case keywords make no edge.
    assert len(pairs) == 4


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    return all(
        np.array_equal(v, b[k])
        if isinstance(v, np.ndarray)
        else k == "timing" or v == b[k]
        for k, v in a.items()
    )


def test_result_does_not_depend_on_the_thread_count_or_the_cut(
    tmp_path: pathlib.Path,
) -> None:
    for name, path in _fixtures(tmp_path).items():
        whole = _native.read_goal_text(str(path), 1, 1 << 40)
        for threads, chunk in ((1, 1), (3, 100), (8, 1000), (4, 1 << 20)):
            split = _native.read_goal_text(str(path), threads, chunk)
            assert _same(whole, split), (name, threads, chunk)


def test_an_empty_schedule_gives_empty_tables_of_the_right_types(
    tmp_path: pathlib.Path,
) -> None:
    path = _write(tmp_path / "e.goal", "")
    nodes, edges = reader.parse_goal_text(path)
    assert nodes.schema == reader.reader.NODES_SCHEMA
    assert edges.schema == reader.reader.EDGES_SCHEMA
    assert nodes.height == 0
    assert edges.height == 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("rank 0 {\nl1: calc 99999999999999999999999\n}\n", "64 bits"),
        ("rank 0 {\nl1: send 1b to 99999999999 \n}\n", "32-bit"),
        ("rank 0 {\nl1: send 1b to 1 tag 5000000000\n}\n", "32-bit"),
        ("rank 99999999999 {\nl1: calc 1\n}\n", "32-bit"),
        ("rank 0 {\nl1: send 99999999999999999kb to 1\n}\n", "64 bits"),
    ],
)
def test_numbers_that_do_not_fit_are_rejected(
    tmp_path: pathlib.Path, text: str, message: str
) -> None:
    path = _write(tmp_path / "n.goal", text.replace(" \n", "\n"))
    with pytest.raises(ValueError, match=message):
        _native.read_goal_text(str(path), 2, 4)


def test_text_that_is_not_utf8_is_rejected(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.goal"
    path.write_bytes(b"rank 0 {\nl\xff: calc 1\n}\n")
    with pytest.raises(ValueError, match="UTF-8"):
        _native.read_goal_text(str(path), 1, 1 << 20)


def test_a_missing_file_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError, match="cannot open"):
        _native.read_goal_text(str(tmp_path / "none.goal"), 1, 100)
