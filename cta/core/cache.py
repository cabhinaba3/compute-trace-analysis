"""Content-addressed Parquet cache of ingested trace tables.

Each artifact is ingested once per content digest. An entry directory holds
one Parquet file per table plus a ``_meta.json`` that lists them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil

import polars as pl

from cta.core import models
from cta.core import registry

_FULL_HASH_LIMIT = 4 * 1024 * 1024  # 4 MiB
_INTERIOR_SAMPLES = 14
# 64 KiB samples (head, interior, tail) stand in for multi-GB files.
_SAMPLE_CHUNK = 64 * 1024
_READ_CHUNK = 262144  # 256 KiB


def compute_artifact_digest(
    artifact: models.TraceArtifact, adapter_version: str
) -> str:
    """Computes the cache key of an artifact from its files and adapter.

    Files up to 4 MiB are hashed in full. Larger files are sampled: the
    head, 14 evenly spaced interior chunks and the tail. File names and
    sizes are hashed, modification times are not, so copies and fresh
    checkouts still hit the cache.

    Args:
        artifact: The trace artifact to fingerprint.
        adapter_version: Version of the adapter that ingests the artifact,
            so an adapter change invalidates old entries.

    Returns:
        A 32-character hex BLAKE2b digest.
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(
        f"{artifact.format}:{adapter_version}:{artifact.world_size}".encode()
    )
    for path in sorted(artifact.paths):
        size = path.stat().st_size
        # No mtime: copies and checkouts must hit the cache.
        hasher.update(f"{path.name}:{size}".encode())
        with path.open("rb") as fh:
            if size <= _FULL_HASH_LIMIT:
                while chunk := fh.read(_READ_CHUNK):
                    hasher.update(chunk)
            else:
                hasher.update(fh.read(_SAMPLE_CHUNK))
                # Evenly spaced interior samples, so in-place edits in the
                # middle of the file are (probably) noticed.
                for i in range(1, _INTERIOR_SAMPLES + 1):
                    fh.seek(size * i // (_INTERIOR_SAMPLES + 1))
                    hasher.update(fh.read(_SAMPLE_CHUNK))
                fh.seek(max(0, size - _SAMPLE_CHUNK))
                hasher.update(fh.read(_SAMPLE_CHUNK))
    return hasher.hexdigest()


class CacheStore:
    """A directory of ingested trace tables, keyed by artifact digest.

    Attributes:
        root: Root directory of the cache.
    """

    def __init__(self, root: pathlib.Path = pathlib.Path(".cta_cache")) -> None:
        """Initialises the store.

        Args:
            root: Root directory of the cache; created on first ingest.
        """
        self.root = root

    def entry_dir(self, digest: str) -> pathlib.Path:
        """Returns the entry directory for a digest.

        Args:
            digest: Artifact digest.

        Returns:
            ``<root>/<first two digest characters>/<digest>``.
        """
        return self.root / digest[:2] / digest

    def get_or_ingest(
        self, artifact: models.TraceArtifact, adapter: registry.TraceAdapter
    ) -> dict[str, pathlib.Path]:
        """Returns the artifact's cached tables, ingesting them on a miss.

        A cached entry is reused when its ``_meta.json`` is readable and
        every table it lists exists; otherwise the artifact is ingested
        again and the metadata rewritten atomically.

        Args:
            artifact: The artifact to ingest. Its ``digest`` is used as the
                cache key when set.
            adapter: The adapter that reads the artifact's format.

        Returns:
            Table name -> Parquet path.
        """
        digest = artifact.digest or compute_artifact_digest(
            artifact, adapter.version
        )
        target_dir = self.entry_dir(digest)
        meta_path = target_dir / "_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cached_paths = {
                    name: target_dir / file_name
                    for name, file_name in meta["tables"].items()
                }
            except (ValueError, KeyError, AttributeError, OSError):
                # Truncated or corrupt metadata: re-ingest.
                cached_paths = {}
            if cached_paths and all(
                path.exists() for path in cached_paths.values()
            ):
                return cached_paths

        target_dir.mkdir(parents=True, exist_ok=True)
        ingested_paths = adapter.ingest(artifact, target_dir)
        file_names: dict[str, str] = {}
        table_paths: dict[str, pathlib.Path] = {}
        for name, ingested_path in ingested_paths.items():
            parquet_path = target_dir / f"{name}.parquet"
            if ingested_path.resolve() != parquet_path.resolve():
                shutil.move(str(ingested_path), str(parquet_path))
            file_names[name] = parquet_path.name
            table_paths[name] = parquet_path

        tmp_meta_path = meta_path.with_name(f"_meta.json.tmp.{os.getpid()}")
        tmp_meta_path.write_text(
            json.dumps(
                {
                    "format": artifact.format,
                    "version": adapter.version,
                    "world_size": artifact.world_size,
                    "sources": [str(path) for path in artifact.paths],
                    "tables": file_names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_meta_path.replace(meta_path)
        return table_paths

    def write_parquet(
        self, frame: pl.DataFrame, path: pathlib.Path
    ) -> pathlib.Path:
        """Writes a frame as zstd-compressed Parquet.

        Args:
            frame: The frame to write.
            path: Destination file; its parent directories are created.

        Returns:
            ``path``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression="zstd", compression_level=3)
        return path

    def status(self) -> dict[str, int]:
        """Returns the number of cache entries and their total size.

        Returns:
            ``{"entries": <entry count>, "bytes": <total file size>}``.
        """
        if not self.root.exists():
            return {"entries": 0, "bytes": 0}
        meta_paths = list(self.root.glob("*/*/_meta.json"))
        total_bytes = sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file()
        )
        return {"entries": len(meta_paths), "bytes": total_bytes}

    def clear(self) -> int:
        """Deletes the whole cache directory.

        Returns:
            The number of entries that were removed.
        """
        entry_count = self.status()["entries"]
        if self.root.exists():
            shutil.rmtree(self.root)
        return entry_count
