"""
BM25 Okapi keyword search backend.
Scored from an inverted index (see api.bm25) rather than a full corpus scan.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from api.bm25 import PostingsBM25
from api.durability import atomic_write
from api.chunk_store import load_shared
from api.search_backends.base import KeywordSearchBackend

INDEX_FORMAT_VERSION = 4

_MAGIC = b"TSBM25\x00\x00"


class BM25OkapiBackend(KeywordSearchBackend):
    """
    BM25 Okapi variant keyword search backend.
    Good baseline; scoring cost scales with postings length, not corpus size.
    """

    def __init__(self):
        self._index = PostingsBM25()
        self._texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        """Build BM25 index from texts."""
        self._texts = texts
        self._index.build(texts)

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Return top-k chunk indices ranked by BM25 relevance."""
        return self._index.search_indices(query, top_k)

    def search_scored(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return top-k (index, BM25 score) pairs, best first."""
        return self._index.search_scored(query, top_k)

    def term_evidence(self, query: str) -> Dict[str, float]:
        """IDF of each query term present in the corpus vocabulary."""
        return self._index.term_evidence(query)

    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve text chunks at indices."""
        return [self._texts[i] for i in indices if i < len(self._texts)]

    @property
    def texts(self) -> List[str]:
        return self._texts

    def save(self, path: str) -> None:
        """Save index to disk as {path}.bm25."""
        payload = self._index.to_arrays()
        payload["meta"] = np.frombuffer(
            json.dumps(
                {"version": INDEX_FORMAT_VERSION, "variant": getattr(self, "variant", None)}
            ).encode("utf-8"),
            dtype=np.uint8,
        )
        with atomic_write(f"{path}.bm25", "wb") as f:
            f.write(_MAGIC)
            np.savez(f, **payload)

    def load(self, path: str) -> None:
        """Load the postings, and attach the shared chunk store."""
        bm25_path = f"{path}.bm25"
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"Keyword index not found: {bm25_path}")

        with open(bm25_path, "rb") as handle:
            if handle.read(len(_MAGIC)) != _MAGIC:
                raise ValueError(
                    f"'{bm25_path}' predates index format {INDEX_FORMAT_VERSION}. "
                    "Re-ingest the collection."
                )
            # allow_pickle stays False: this file is derived from a downloaded ZIM.
            with np.load(handle, allow_pickle=False) as data:
                meta = json.loads(data["meta"].tobytes().decode("utf-8"))
                if meta.get("version") != INDEX_FORMAT_VERSION:
                    raise ValueError(
                        f"'{bm25_path}' has index format {meta.get('version')}, "
                        f"expected {INDEX_FORMAT_VERSION}. Re-ingest the collection."
                    )
                self._index = PostingsBM25.from_arrays(data)

        # Chunk text comes from the store the vector index also reads, so the
        # corpus is held once in this process rather than once per index.
        self._texts = load_shared(path).texts
