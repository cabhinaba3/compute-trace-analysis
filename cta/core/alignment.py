"""Kernel/op name alignment: demangling, canonical names and op families.

Traces from different profilers name the same kernel differently (mangled or
demangled, with or without template arguments). These helpers map raw names
to a canonical name and a coarse op family so they can be compared.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
import shutil
import subprocess

import polars as pl

_TEMPLATE_RE = re.compile(r"<[^<>]*>")
_LEADING_RETURN_TYPE_RE = re.compile(
    r"^(?:void|int|float|double|unsigned(?:\s+\w+)?)\s+"
)


def demangle_unique(names: Iterable[str]) -> dict[str, str]:
    """Demangles C++ symbol names with ``c++filt``, once per distinct name.

    Names that are not Itanium-mangled (no ``_Z`` prefix) map to themselves.
    When ``c++filt`` is missing, fails, or returns an unexpected number of
    lines, every name maps to itself.

    Args:
        names: Raw names; empty names and duplicates are skipped.

    Returns:
        Raw name -> demangled name, for every distinct non-empty name.
    """
    unique_names = [name for name in dict.fromkeys(names) if name]
    if not unique_names:
        return {}
    mangled = [name for name in unique_names if name.startswith("_Z")]
    demangled_by_name: dict[str, str] = {name: name for name in unique_names}
    cxxfilt_path = shutil.which("c++filt")
    if mangled and cxxfilt_path:
        try:
            completed = subprocess.run(
                [cxxfilt_path],
                input="\n".join(mangled),
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            )
            output_lines = completed.stdout.split("\n")
            if output_lines and output_lines[-1] == "":
                output_lines.pop()
            if len(output_lines) == len(mangled):
                for mangled_name, demangled_name in zip(
                    mangled, output_lines, strict=True
                ):
                    demangled_by_name[mangled_name] = (
                        demangled_name.strip() or mangled_name
                    )
        except (subprocess.SubprocessError, OSError):
            pass
    return demangled_by_name


def normalize_signature(name: str) -> str:
    """Reduces a demangled kernel signature to its bare qualified name.

    Drops a leading simple return type, every (nested) template argument
    list and the parameter list.

    Args:
        name: A demangled kernel or function name.

    Returns:
        The name without return type, template arguments and parameters.
    """
    signature = _LEADING_RETURN_TYPE_RE.sub("", name.strip())
    previous = None
    while previous != signature:
        previous = signature
        signature = _TEMPLATE_RE.sub("", signature)
    paren_index = signature.find("(")
    if paren_index > 0:
        signature = signature[:paren_index]
    return signature.strip()


def op_family_expr(column: str = "name") -> pl.Expr:
    """Builds an expression that classifies op names into op families.

    Matching is case-insensitive and the first matching family wins, e.g.
    ``NCCL_AllReduce``, ``GEMM``, ``FlashAttention`` or ``Memcpy_H2D``.
    Names that match no family get ``"Other"``.

    Args:
        column: Name of the string column holding the op names.

    Returns:
        A string expression with the op family of each row.
    """
    lowered = pl.col(column).str.to_lowercase()
    return (
        pl.when(
            lowered.str.contains(
                r"nccl.*allreduce|all_reduce|c10d::allreduce_?"
            )
        )
        .then(pl.lit("NCCL_AllReduce"))
        .when(
            lowered.str.contains(
                r"nccl.*allgather|all_gather|c10d::allgather_?"
            )
        )
        .then(pl.lit("NCCL_AllGather"))
        .when(
            lowered.str.contains(
                r"nccl.*reducescatter|reduce_scatter|c10d::reduce_scatter_?"
            )
        )
        .then(pl.lit("NCCL_ReduceScatter"))
        .when(
            lowered.str.contains(r"nccl.*alltoall|all_to_all|c10d::alltoall_?")
        )
        .then(pl.lit("NCCL_AllToAll"))
        .when(lowered.str.contains(r"nccl.*broadcast|c10d::broadcast_?"))
        .then(pl.lit("NCCL_Broadcast"))
        .when(
            lowered.str.contains(
                r"nccl.*send|nccl.*recv|comm_send|comm_recv|\bsend\b|\brecv\b"
            )
        )
        .then(pl.lit("Comm_P2P"))
        .when(
            lowered.str.contains(
                r"flash_?attn|flash_(?:fwd|bwd)|flash::|fmha"
                r"|scaled_dot_product|sdpa|xformers"
            )
        )
        .then(pl.lit("FlashAttention"))
        .when(
            lowered.str.contains(
                r"conv\d*d|convolution|cudnn.*conv|winograd|implicit_convolve"
                r"|implicit_gemm|fprop|dgrad|wgrad"
            )
        )
        .then(pl.lit("Convolution"))
        .when(
            lowered.str.contains(
                r"gemm|gemv|matmul|cublas|cutlass|wgmma|xmma|hmma|imma"
                r"|\bmma\b|_mma_|\bmm\b|bmm|addmm|linear"
            )
        )
        .then(pl.lit("GEMM"))
        .when(
            lowered.str.contains(
                r"layernorm|rmsnorm|batchnorm|groupnorm|bn_(?:fw|bw)|norm"
            )
        )
        .then(pl.lit("Normalization"))
        .when(lowered.str.contains(r"softmax"))
        .then(pl.lit("Softmax"))
        .when(lowered.str.contains(r"memcpy.*htod|h2d|hosttodevice"))
        .then(pl.lit("Memcpy_H2D"))
        .when(lowered.str.contains(r"memcpy.*dtoh|d2h|devicetohost"))
        .then(pl.lit("Memcpy_D2H"))
        .when(lowered.str.contains(r"memcpy|d2d|devicetodevice"))
        .then(pl.lit("Memcpy_D2D"))
        .when(lowered.str.contains(r"memset"))
        .then(pl.lit("Memset"))
        .when(lowered.str.contains(r"embed|index_?select|gather"))
        .then(pl.lit("Embedding"))
        .when(lowered.str.contains(r"reduce|triton_red|devicescan|\bscan\b"))
        .then(pl.lit("Reduction"))
        .when(
            lowered.str.contains(
                r"relu|gelu|silu|swish|elementwise|vectorized_elementwise"
                r"|(?:^|[^a-z])(?:add|mul|cast|copy)(?:[^a-z]|$)"
            )
        )
        .then(pl.lit("Elementwise"))
        .otherwise(pl.lit("Other"))
    )


def enrich_ops_df(ops: pl.DataFrame, name_column: str = "name") -> pl.DataFrame:
    """Adds demangled name, canonical name and op family columns to ops.

    Args:
        ops: Any frame with a string column of raw op or kernel names.
        name_column: Name of that column.

    Returns:
        ``ops`` with three added columns: ``demangled_name``,
        ``canonical_name`` (see ``normalize_signature``) and ``op_family``
        (see ``op_family_expr``). Names that cannot be demangled are kept
        as they are.
    """
    if ops.is_empty():
        return ops.with_columns(
            pl.lit("").alias("demangled_name"),
            pl.lit("").alias("canonical_name"),
            pl.lit("Other").alias("op_family"),
        )
    raw_names = ops[name_column].to_list()
    demangled_by_name = demangle_unique(raw_names)
    canonical_by_name = {
        name: normalize_signature(demangled)
        for name, demangled in demangled_by_name.items()
    }
    return ops.with_columns(
        pl.col(name_column)
        .replace_strict(demangled_by_name, default=pl.col(name_column))
        .alias("demangled_name"),
        pl.col(name_column)
        .replace_strict(canonical_by_name, default=pl.col(name_column))
        .alias("canonical_name"),
    ).with_columns(op_family_expr("demangled_name").alias("op_family"))
