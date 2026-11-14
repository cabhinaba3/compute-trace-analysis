"""TensorRT trtexec timing, layer profile, and engine layer-info JSON adapters.

Three ``trtexec`` exports are read, each by its own adapter: per-query timings
(``--exportTimes``), per-layer profiles (``--exportProfile``) and engine layer
information (``--exportLayerInfo``).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import polars as pl

from cta.adapters.nsys import reader as nsys_reader
from cta.core import alignment
from cta.core import models
from cta.core import registry

TRT_PROFILE_SCHEMA = pl.Schema(
    {
        "name": pl.String(),
        "op_family": pl.String(),
        "avg_ms": pl.Float64(),
        "median_ms": pl.Float64(),
        "time_share_pct": pl.Float64(),
        "is_fused": pl.Boolean(),
    }
)

TRT_ENGINE_SCHEMA = pl.Schema(
    {
        "name": pl.String(),
        "layer_type": pl.String(),
        "precision": pl.String(),
        "tactic": pl.String(),
        "is_fused": pl.Boolean(),
    }
)


def _peek_json_head(path: pathlib.Path, max_bytes: int = 8192) -> str:
    """Returns the first bytes of a file as text, or "" if it cannot be read."""
    try:
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _is_fused_layer_name(name: str) -> bool:
    """Returns whether a TensorRT layer name denotes several fused layers."""
    return (
        "+" in name
        or "||" in name
        or "fused" in name.lower()
        or "myelin" in name.lower()
    )


def _ms_to_ns(value_ms: Any) -> int:
    """Converts a trtexec millisecond value to integer nanoseconds."""
    return int(round(float(value_ms) * 1e6))


def _optional_ms_to_ns(
    record: dict[str, Any], key: str, default_ns: int
) -> int:
    """Returns ``record[key]`` in nanoseconds, or ``default_ns`` if unset."""
    if record.get(key) is not None:
        return _ms_to_ns(record[key])
    return default_ns


def _query_event_row(
    step_id: int,
    name: str,
    category: str,
    start_ns: int,
    end_ns: int,
    stream: int,
) -> dict[str, object]:
    """Builds the kernels row of one query phase (H2D, compute or D2H)."""
    return {
        "step": step_id,
        "scope": "trtexec_query",
        "name": name,
        "category": category,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "bytes": 0,
        "stream": stream,
    }


@registry.register_adapter("trtexec_timing", version="1.0")
class TrtTimingAdapter:
    """Reads ``trtexec --exportTimes`` JSON: one step per inference query.

    Each query becomes a step spanning its host-to-device copy, compute and
    device-to-host copy; each non-empty phase becomes one kernels-table row.

    Attributes:
        format_name: The registered format name, ``"trtexec_timing"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "trtexec_timing"
    version = "1.0"

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file looks like a trtexec timing export.

        Args:
            path: The candidate file.

        Returns:
            True for a ``.json`` file whose head mentions both
            ``startComputeMs`` and ``endComputeMs``.
        """
        if path.suffix.lower() != ".json":
            return False
        head = _peek_json_head(path)
        return "startComputeMs" in head and "endComputeMs" in head

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the timing exports of an artifact to steps and kernels.

        Queries are numbered across all files in sorted path order. Queries
        without a positive overall span are skipped. A missing H2D (D2H)
        start defaults to the compute start (end), and a missing copy end to
        that copy's start.

        Args:
            artifact: The artifact whose JSON files are read.
            out_dir: The directory the Parquet tables are written to.

        Returns:
            The paths of the ``steps`` and ``kernels`` Parquet tables, by
            table name.
        """
        step_rows: list[dict[str, object]] = []
        kernel_rows: list[dict[str, object]] = []
        step_id = 0

        for path in sorted(artifact.paths):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for record in records:
                if (
                    not isinstance(record, dict)
                    or "startComputeMs" not in record
                    or "endComputeMs" not in record
                ):
                    continue
                compute_start_ns = _ms_to_ns(record["startComputeMs"])
                compute_end_ns = _ms_to_ns(record["endComputeMs"])
                h2d_start_ns = _optional_ms_to_ns(
                    record, "startH2dMs", compute_start_ns
                )
                h2d_end_ns = _optional_ms_to_ns(
                    record, "endH2dMs", h2d_start_ns
                )
                d2h_start_ns = _optional_ms_to_ns(
                    record, "startD2hMs", compute_end_ns
                )
                d2h_end_ns = _optional_ms_to_ns(
                    record, "endD2hMs", d2h_start_ns
                )

                query_start_ns = min(
                    h2d_start_ns, compute_start_ns, d2h_start_ns
                )
                query_end_ns = max(h2d_end_ns, compute_end_ns, d2h_end_ns)
                if query_end_ns <= query_start_ns:
                    continue

                step_rows.append(
                    {
                        "step": step_id,
                        "start_ns": query_start_ns,
                        "end_ns": query_end_ns,
                        "duration_ns": query_end_ns - query_start_ns,
                    }
                )
                if h2d_end_ns > h2d_start_ns:
                    kernel_rows.append(
                        _query_event_row(
                            step_id,
                            "Memcpy_H2D",
                            "memcpy",
                            h2d_start_ns,
                            h2d_end_ns,
                            stream=1,
                        )
                    )
                if compute_end_ns > compute_start_ns:
                    kernel_rows.append(
                        _query_event_row(
                            step_id,
                            "trt_compute",
                            "compute",
                            compute_start_ns,
                            compute_end_ns,
                            stream=0,
                        )
                    )
                if d2h_end_ns > d2h_start_ns:
                    kernel_rows.append(
                        _query_event_row(
                            step_id,
                            "Memcpy_D2H",
                            "memcpy",
                            d2h_start_ns,
                            d2h_end_ns,
                            stream=1,
                        )
                    )
                step_id += 1

        if step_rows:
            steps = nsys_reader.conform(
                pl.DataFrame(step_rows), nsys_reader.STEPS_SCHEMA
            )
        else:
            steps = pl.DataFrame(schema=nsys_reader.STEPS_SCHEMA)
        if kernel_rows:
            kernels = nsys_reader.conform(
                alignment.enrich_ops_df(pl.DataFrame(kernel_rows), "name"),
                nsys_reader.KERNELS_SCHEMA,
            )
        else:
            kernels = pl.DataFrame(schema=nsys_reader.KERNELS_SCHEMA)

        steps_path = out_dir / "steps.parquet"
        kernels_path = out_dir / "kernels.parquet"
        steps.write_parquet(steps_path, compression="zstd")
        kernels.write_parquet(kernels_path, compression="zstd")
        return {"steps": steps_path, "kernels": kernels_path}


@registry.register_adapter("trtexec_profile", version="1.1")
class TrtLayerProfileAdapter:
    """Reads ``trtexec --exportProfile`` JSON: per-layer timing statistics.

    Attributes:
        format_name: The registered format name, ``"trtexec_profile"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "trtexec_profile"
    version = "1.1"

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file looks like a trtexec layer profile.

        Args:
            path: The candidate file.

        Returns:
            True for a ``.json`` file whose head mentions ``percentage`` and
            either ``averageMs`` or ``timeMs``.
        """
        if path.suffix.lower() != ".json":
            return False
        head = _peek_json_head(path)
        return (
            "averageMs" in head or "timeMs" in head
        ) and "percentage" in head

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the layer profiles of an artifact to one layer table.

        Entries without a ``name`` (such as the leading ``count`` record)
        are skipped. The average falls back to ``timeMs`` and the median to
        the average.

        Args:
            artifact: The artifact whose JSON files are read.
            out_dir: The directory the Parquet table is written to.

        Returns:
            The path of the ``trt_profile`` Parquet table, by table name.
        """
        layer_rows: list[dict[str, object]] = []
        for path in sorted(artifact.paths):
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict) or "name" not in record:
                    continue
                name = str(record["name"])
                avg_ms = float(
                    record.get("averageMs", record.get("timeMs", 0.0)) or 0.0
                )
                median_ms = float(record.get("medianMs", avg_ms) or avg_ms)
                time_share_pct = float(record.get("percentage", 0.0) or 0.0)
                layer_rows.append(
                    {
                        "name": name,
                        "avg_ms": avg_ms,
                        "median_ms": median_ms,
                        "time_share_pct": time_share_pct,
                        "is_fused": _is_fused_layer_name(name),
                    }
                )

        if layer_rows:
            layers = (
                alignment.enrich_ops_df(pl.DataFrame(layer_rows), "name")
                .select(
                    "name",
                    "op_family",
                    "avg_ms",
                    "median_ms",
                    "time_share_pct",
                    "is_fused",
                )
                .cast(TRT_PROFILE_SCHEMA)
            )
        else:
            layers = pl.DataFrame(schema=TRT_PROFILE_SCHEMA)

        profile_path = out_dir / "trt_profile.parquet"
        layers.write_parquet(profile_path, compression="zstd")
        return {"trt_profile": profile_path}


def _extract_trt_precision(layer: dict[str, Any]) -> str:
    """Returns a layer's precision as INT8, FP8, FP16, BF16, FP32 or raw text.

    The precision comes from ``Precision`` or ``ParameterType``, else from the
    data type of the first input (then output) port. Unrecognised values are
    returned upper-cased, and a missing value as ``"UNKNOWN"``.
    """
    raw_precision = str(
        layer.get("Precision", layer.get("ParameterType", ""))
    ).upper()
    if not raw_precision:
        for port_key in ("Inputs", "Outputs"):
            ports = layer.get(port_key)
            if isinstance(ports, list) and ports and isinstance(ports[0], dict):
                first_port = ports[0]
                raw_precision = str(
                    first_port.get(
                        "Format/Datatype", first_port.get("Datatype", "")
                    )
                ).upper()
                if raw_precision:
                    break
    if "INT8" in raw_precision:
        return "INT8"
    if "FP8" in raw_precision:
        return "FP8"
    if "HALF" in raw_precision or "FP16" in raw_precision:
        return "FP16"
    if "BF16" in raw_precision or "BFLOAT16" in raw_precision:
        return "BF16"
    if "FLOAT" in raw_precision or "FP32" in raw_precision:
        return "FP32"
    return raw_precision or "UNKNOWN"


@registry.register_adapter("trtexec_engine", version="1.0")
class TrtEngineInfoAdapter:
    """Reads ``trtexec --exportLayerInfo`` JSON: the engine's layer list.

    Attributes:
        format_name: The registered format name, ``"trtexec_engine"``.
        version: The adapter version, part of the ingest cache key.
    """

    format_name = "trtexec_engine"
    version = "1.0"

    def detect(self, path: pathlib.Path) -> bool:
        """Returns whether a file looks like a trtexec engine layer-info export.

        Args:
            path: The candidate file.

        Returns:
            True for a ``.json`` file whose head has a ``"Layers"`` key and a
            ``"LayerType"``, ``"TacticValue"`` or ``"ParameterType"`` key.
        """
        if path.suffix.lower() != ".json":
            return False
        head = _peek_json_head(path)
        return '"Layers"' in head and (
            '"LayerType"' in head
            or '"TacticValue"' in head
            or '"ParameterType"' in head
        )

    def ingest(
        self, artifact: models.TraceArtifact, out_dir: pathlib.Path
    ) -> dict[str, pathlib.Path]:
        """Converts the engine layer lists of an artifact to one layer table.

        Layers given only by name get an ``"Unknown"`` type and
        ``"UNKNOWN"`` precision.

        Args:
            artifact: The artifact whose JSON files are read.
            out_dir: The directory the Parquet table is written to.

        Returns:
            The path of the ``trt_engine`` Parquet table, by table name.
        """
        layer_rows: list[dict[str, object]] = []
        for path in sorted(artifact.paths):
            engine_info = json.loads(path.read_text(encoding="utf-8"))
            layer_entries = (
                engine_info.get("Layers", [])
                if isinstance(engine_info, dict)
                else []
            )
            for layer in layer_entries:
                if isinstance(layer, str):
                    layer_rows.append(
                        {
                            "name": layer,
                            "layer_type": "Unknown",
                            "precision": "UNKNOWN",
                            "tactic": "",
                            "is_fused": _is_fused_layer_name(layer),
                        }
                    )
                elif isinstance(layer, dict):
                    name = str(layer.get("Name", ""))
                    layer_type = str(
                        layer.get(
                            "LayerType", layer.get("ParameterType", "Unknown")
                        )
                    )
                    tactic = str(
                        layer.get("TacticName", layer.get("TacticValue", ""))
                    )
                    layer_rows.append(
                        {
                            "name": name,
                            "layer_type": layer_type,
                            "precision": _extract_trt_precision(layer),
                            "tactic": tactic,
                            "is_fused": _is_fused_layer_name(name),
                        }
                    )

        if layer_rows:
            layers = pl.DataFrame(layer_rows, schema=TRT_ENGINE_SCHEMA)
        else:
            layers = pl.DataFrame(schema=TRT_ENGINE_SCHEMA)
        engine_path = out_dir / "trt_engine.parquet"
        layers.write_parquet(engine_path, compression="zstd")
        return {"trt_engine": engine_path}
