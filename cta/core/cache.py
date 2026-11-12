from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import polars as pl

from cta.core.models import TraceArtifact
from cta.core.registry import TraceAdapter

_FULL_HASH_LIMIT = 4 * 1024 * 1024  # 4 MiB
_INTERIOR_SAMPLES = 14
_SAMPLE_CHUNK = 64 * 1024  # 64 KiB head + tail for multi-GB files


def compute_artifact_digest(artifact: TraceArtifact, adapter_version: str) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(f"{artifact.format}:{adapter_version}:{artifact.world_size}".encode())
    for p in sorted(artifact.paths):
        st = p.stat()
        hasher.update(f"{p.name}:{st.st_size}".encode())  # no mtime: copies/checkouts must hit the cache
        with p.open("rb") as fh:
            if st.st_size <= _FULL_HASH_LIMIT:
                while chunk := fh.read(262144):
                    hasher.update(chunk)
            else:
                hasher.update(fh.read(_SAMPLE_CHUNK))
                # evenly spaced interior samples so in-place edits in the middle are (probably) noticed
                for i in range(1, _INTERIOR_SAMPLES + 1):
                    fh.seek(st.st_size * i // (_INTERIOR_SAMPLES + 1))
                    hasher.update(fh.read(_SAMPLE_CHUNK))
                fh.seek(max(0, st.st_size - _SAMPLE_CHUNK))
                hasher.update(fh.read(_SAMPLE_CHUNK))
    return hasher.hexdigest()


class CacheStore:
    def __init__(self, root: Path = Path(".cta_cache")) -> None:
        self.root = root

    def entry_dir(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def get_or_ingest(self, artifact: TraceArtifact, adapter: TraceAdapter) -> dict[str, Path]:
        digest = artifact.digest or compute_artifact_digest(artifact, adapter.version)
        target = self.entry_dir(digest)
        meta_file = target / "_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                paths = {k: target / v for k, v in meta["tables"].items()}
            except (ValueError, KeyError, AttributeError, OSError):
                paths = {}  # truncated/corrupt metadata: re-ingest
            if paths and all(p.exists() for p in paths.values()):
                return paths

        target.mkdir(parents=True, exist_ok=True)
        tables = adapter.ingest(artifact, target)
        rel_tables: dict[str, str] = {}
        final_paths: dict[str, Path] = {}
        for name, src_path in tables.items():
            dst = target / f"{name}.parquet"
            if src_path.resolve() != dst.resolve():
                shutil.move(str(src_path), str(dst))
            rel_tables[name] = dst.name
            final_paths[name] = dst

        tmp_meta = meta_file.with_name(f"_meta.json.tmp.{os.getpid()}")
        tmp_meta.write_text(
            json.dumps(
                {
                    "format": artifact.format,
                    "version": adapter.version,
                    "world_size": artifact.world_size,
                    "sources": [str(p) for p in artifact.paths],
                    "tables": rel_tables,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_meta.replace(meta_file)
        return final_paths

    def write_parquet(self, df: pl.DataFrame, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dst, compression="zstd", compression_level=3)
        return dst

    def status(self) -> dict[str, int]:
        if not self.root.exists():
            return {"entries": 0, "bytes": 0}
        metas = list(self.root.glob("*/*/_meta.json"))
        total_bytes = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        return {"entries": len(metas), "bytes": total_bytes}

    def clear(self) -> int:
        st = self.status()
        if self.root.exists():
            shutil.rmtree(self.root)
        return st["entries"]
