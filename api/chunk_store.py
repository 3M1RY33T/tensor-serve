"""
The chunk text and metadata for one collection, stored once.

The vector index and the keyword index both need the chunk text, and each used
to persist its own copy: on a 10,399-chunk corpus that is 9.6MB of duplicated
text on disk, loaded twice into RAM at boot. They now share this store.

Sharing is by object, not just by file — ``load_shared`` returns the same
instance to every caller reading the same path in a process, so the text exists
once in memory as well as once on disk. The cache is keyed on the file's
identity and mtime, so a re-ingest is picked up rather than served stale.

The format holds no pickles. An index file is derived from a ZIM someone
downloaded, and ``pickle.load`` on such a file executes whatever it contains;
text is stored as one UTF-8 blob plus an offsets array, metadata as JSON.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from api.durability import atomic_write

STORE_SUFFIX = ".chunks"
STORE_FORMAT_VERSION = 2

_MAGIC = b"TSCHUNK\x00"


class ChunkStore:
    """Chunk text and per-chunk metadata for one collection."""

    def __init__(self, texts: Optional[List[str]] = None, metadata: Optional[List[dict]] = None):
        self.texts: List[str] = texts if texts is not None else []
        self.metadata: List[dict] = metadata if metadata is not None else []

    def extend(self, chunks: List[str], metadata: Optional[List[dict]] = None) -> None:
        self.texts.extend(chunks)
        self.metadata.extend(metadata if metadata is not None else [{} for _ in chunks])

    def __len__(self) -> int:
        return len(self.texts)

    # ---- persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        """Write the store atomically, so a reader never sees it half-written."""
        encoded = [t.encode("utf-8") for t in self.texts]
        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        if encoded:
            np.cumsum([len(b) for b in encoded], out=offsets[1:])

        payload = {
            "version": np.asarray([STORE_FORMAT_VERSION], dtype=np.int32),
            "offsets": offsets,
            "blob": np.frombuffer(b"".join(encoded), dtype=np.uint8),
            "metadata": np.frombuffer(
                json.dumps(self.metadata, ensure_ascii=False).encode("utf-8"),
                dtype=np.uint8,
            ),
        }

        with atomic_write(f"{path}{STORE_SUFFIX}", "wb") as handle:
            handle.write(_MAGIC)
            np.savez(handle, **payload)

    @classmethod
    def read(cls, path: str) -> "ChunkStore":
        store_path = f"{path}{STORE_SUFFIX}"
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Chunk store not found: {store_path}")

        with open(store_path, "rb") as handle:
            if handle.read(len(_MAGIC)) != _MAGIC:
                raise ValueError(
                    f"'{store_path}' is not a Tensor chunk store, or predates "
                    f"format {STORE_FORMAT_VERSION}. Re-ingest the collection."
                )
            # allow_pickle stays False: this file came from a downloaded ZIM.
            with np.load(handle, allow_pickle=False) as data:
                version = int(data["version"][0])
                if version != STORE_FORMAT_VERSION:
                    raise ValueError(
                        f"Chunk store '{store_path}' has format {version}, "
                        f"expected {STORE_FORMAT_VERSION}. Re-ingest the collection."
                    )
                offsets = data["offsets"]
                blob = data["blob"].tobytes()
                metadata = json.loads(data["metadata"].tobytes().decode("utf-8") or "[]")

        texts = [
            blob[offsets[i] : offsets[i + 1]].decode("utf-8")
            for i in range(len(offsets) - 1)
        ]
        return cls(texts, metadata)


def store_exists(path: str) -> bool:
    return os.path.exists(f"{path}{STORE_SUFFIX}")


_cache: Dict[str, Tuple[tuple, ChunkStore]] = {}


def _stamp(store_path: str) -> tuple:
    stat = os.stat(store_path)
    return (stat.st_mtime_ns, stat.st_size)


def load_shared(path: str) -> ChunkStore:
    """
    Load a chunk store, returning the same object to every caller in this
    process so the vector and keyword indexes share one copy in memory.
    """
    store_path = f"{path}{STORE_SUFFIX}"
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Chunk store not found: {store_path}")

    key = os.path.abspath(store_path)
    stamp = _stamp(store_path)
    cached = _cache.get(key)
    if cached and cached[0] == stamp:
        return cached[1]

    store = ChunkStore.read(path)
    _cache[key] = (stamp, store)
    return store


def clear_cache() -> None:
    """Forget every shared store. For tests, and after deleting indexes."""
    _cache.clear()
