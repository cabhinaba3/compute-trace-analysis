from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable

import polars as pl

_TEMPLATE_RE = re.compile(r"<[^<>]*>")
_LEADING_RET_RE = re.compile(r"^(?:void|int|float|double|unsigned(?:\s+\w+)?)\s+")


def demangle_unique(names: Iterable[str]) -> dict[str, str]:
    unique = [n for n in dict.fromkeys(names) if n]
    if not unique:
        return {}
    mangled = [n for n in unique if n.startswith("_Z")]
    resolved: dict[str, str] = {n: n for n in unique}
    cxxfilt = shutil.which("c++filt")
    if mangled and cxxfilt:
        try:
            proc = subprocess.run(
                [cxxfilt],
                input="\n".join(mangled),
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            )
            lines = proc.stdout.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if len(lines) == len(mangled):
                for src, dst in zip(mangled, lines, strict=True):
                    resolved[src] = dst.strip() or src
        except (subprocess.SubprocessError, OSError):
            pass
    return resolved


def normalize_signature(name: str) -> str:
    s = _LEADING_RET_RE.sub("", name.strip())
    prev = None
    while prev != s:
        prev = s
        s = _TEMPLATE_RE.sub("", s)
    paren = s.find("(")
    if paren > 0:
        s = s[:paren]
    return s.strip()


def op_family_expr(col: str = "name") -> pl.Expr:
    c = pl.col(col).str.to_lowercase()
    return (
        pl.when(c.str.contains(r"nccl.*allreduce|all_reduce|c10d::allreduce_?"))
        .then(pl.lit("NCCL_AllReduce"))
        .when(c.str.contains(r"nccl.*allgather|all_gather|c10d::allgather_?"))
        .then(pl.lit("NCCL_AllGather"))
        .when(c.str.contains(r"nccl.*reducescatter|reduce_scatter|c10d::reduce_scatter_?"))
        .then(pl.lit("NCCL_ReduceScatter"))
        .when(c.str.contains(r"nccl.*alltoall|all_to_all|c10d::alltoall_?"))
        .then(pl.lit("NCCL_AllToAll"))
        .when(c.str.contains(r"nccl.*broadcast|c10d::broadcast_?"))
        .then(pl.lit("NCCL_Broadcast"))
        .when(c.str.contains(r"nccl.*send|nccl.*recv|comm_send|comm_recv|\bsend\b|\brecv\b"))
        .then(pl.lit("Comm_P2P"))
        .when(c.str.contains(r"flash_?attn|flash_(?:fwd|bwd)|flash::|fmha|scaled_dot_product|sdpa|xformers"))
        .then(pl.lit("FlashAttention"))
        .when(c.str.contains(r"conv\d*d|convolution|cudnn.*conv|winograd|implicit_convolve|implicit_gemm|fprop|dgrad|wgrad"))
        .then(pl.lit("Convolution"))
        .when(c.str.contains(r"gemm|gemv|matmul|cublas|cutlass|wgmma|xmma|hmma|imma|\bmma\b|_mma_|\bmm\b|bmm|addmm|linear"))
        .then(pl.lit("GEMM"))
        .when(c.str.contains(r"layernorm|rmsnorm|batchnorm|groupnorm|bn_(?:fw|bw)|norm"))
        .then(pl.lit("Normalization"))
        .when(c.str.contains(r"softmax"))
        .then(pl.lit("Softmax"))
        .when(c.str.contains(r"memcpy.*htod|h2d|hosttodevice"))
        .then(pl.lit("Memcpy_H2D"))
        .when(c.str.contains(r"memcpy.*dtoh|d2h|devicetohost"))
        .then(pl.lit("Memcpy_D2H"))
        .when(c.str.contains(r"memcpy|d2d|devicetodevice"))
        .then(pl.lit("Memcpy_D2D"))
        .when(c.str.contains(r"memset"))
        .then(pl.lit("Memset"))
        .when(c.str.contains(r"embed|index_?select|gather"))
        .then(pl.lit("Embedding"))
        .when(c.str.contains(r"reduce|triton_red|devicescan|\bscan\b"))
        .then(pl.lit("Reduction"))
        .when(c.str.contains(r"relu|gelu|silu|swish|elementwise|vectorized_elementwise|(?:^|[^a-z])(?:add|mul|cast|copy)(?:[^a-z]|$)"))
        .then(pl.lit("Elementwise"))
        .otherwise(pl.lit("Other"))
    )


def enrich_ops_df(df: pl.DataFrame, name_col: str = "name") -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(
            pl.lit("").alias("demangled_name"),
            pl.lit("").alias("canonical_name"),
            pl.lit("Other").alias("op_family"),
        )
    raw_names = df[name_col].to_list()
    demap = demangle_unique(raw_names)
    cmap = {k: normalize_signature(v) for k, v in demap.items()}
    return df.with_columns(
        pl.col(name_col).replace_strict(demap, default=pl.col(name_col)).alias("demangled_name"),
        pl.col(name_col).replace_strict(cmap, default=pl.col(name_col)).alias("canonical_name"),
    ).with_columns(op_family_expr("demangled_name").alias("op_family"))
