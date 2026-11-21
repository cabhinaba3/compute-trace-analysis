"""The streaming keys of a manifest's `config` reach the analysis config.

They used to be dropped, so `streaming: exact` or `memory_budget_bytes` in a
manifest silently ran with the defaults.
"""

from __future__ import annotations

import pathlib

import pytest
from typer import testing

from cta.analysis import whatif
from cta.cli import app as cli_app
from cta.ingest import manifest
from cta.synth import chakra as synth_chakra
from cta.synth import goal as synth_goal


def _write(tmp_path: pathlib.Path, config: str) -> pathlib.Path:
    trace = tmp_path / "t.goal"
    synth_goal.write_allreduce(trace)
    path = tmp_path / "m.yaml"
    path.write_text(
        "groups:\n"
        f"  - {{id: a, path: {trace.name}}}\n"
        f"  - {{id: b, path: {trace.name}}}\n"
        f"config:\n{config}"
    )
    return path


def test_the_streaming_keys_are_read(tmp_path: pathlib.Path) -> None:
    path = _write(
        tmp_path,
        "  streaming: Exact\n"
        "  graph_engine: graphchi\n"
        "  quantile_sketch: kll\n"
        "  collective_sync: barrier\n"
        "  goal_semantics: strict\n"
        "  memory_budget_bytes: 1000000\n"
        "  window_size: 4096\n"
        "  parquet_bypass_bytes: 5\n"
        "  fail_on_cycle: true\n"
        "  extract_critical_path_ops: true\n"
        "  delay_attribution: false\n"
        "  scratch_dir: scratch\n",
    )

    _, config = manifest.load_manifest(path)

    assert config.streaming == "exact"
    assert config.graph_engine == "graphchi"
    assert config.quantile_sketch == "kll"
    assert config.collective_sync == "barrier"
    assert config.goal_semantics == "strict"
    assert config.memory_budget_bytes == 1_000_000
    assert config.window_size == 4096
    assert config.parquet_bypass_bytes == 5
    assert config.fail_on_cycle is True
    assert config.extract_critical_path_ops is True
    assert config.delay_attribution is False
    assert config.scratch_dir == tmp_path / "scratch"


def test_missing_streaming_keys_keep_the_defaults(
    tmp_path: pathlib.Path,
) -> None:
    _, config = manifest.load_manifest(_write(tmp_path, "  warmup_steps: 1\n"))

    assert (config.streaming, config.graph_engine, config.quantile_sketch) == (
        "auto",
        "gridgraph",
        "ddsketch",
    )
    assert config.memory_budget_bytes == 0
    assert config.fail_on_cycle is False
    assert config.scratch_dir is None
    assert config.parquet_bypass_bytes == 2 * 1024**3


@pytest.mark.parametrize("value", ['"false"', "1", "maybe"])
def test_a_flag_that_is_not_a_boolean_is_refused(
    tmp_path: pathlib.Path, value: str
) -> None:
    """`bool("false")` is True, so a quoted flag must not slip through."""
    path = _write(tmp_path, f"  fail_on_cycle: {value}\n")

    with pytest.raises(ValueError, match="fail_on_cycle"):
        manifest.load_manifest(path)


def test_a_chakra_host_trace_is_a_streaming_what_if_source(
    tmp_path: pathlib.Path,
) -> None:
    """`load_streaming_model` knew only chakra and goal, not chakra_host."""
    trace = synth_chakra.write_host_trace(tmp_path / "host.json")

    model = whatif.load_streaming_model(trace, cache_dir=tmp_path / "c")

    assert model.num_nodes > 0


def test_whatif_streaming_off_refuses_options_only_the_streaming_model_has(
    tmp_path: pathlib.Path,
) -> None:
    trace = tmp_path / "t.goal"
    synth_goal.write_allreduce(trace)
    runner = testing.CliRunner()

    refused = runner.invoke(
        cli_app.app,
        [
            "whatif",
            str(trace),
            "--streaming",
            "off",
            "--goal-semantics",
            "strict",
        ],
    )
    unknown = runner.invoke(
        cli_app.app, ["whatif", str(trace), "--streaming", "sometimes"]
    )

    def text(result: testing.Result) -> str:
        """The output without the error box's frame and line breaks."""
        return " ".join(result.output.replace("│", " ").split())

    assert refused.exit_code != 0
    assert "drop --streaming off" in text(refused)
    assert unknown.exit_code != 0
    assert "auto, on, exact or off" in text(unknown)
