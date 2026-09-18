"""
BM25 Okapi keyword search backend.
Scored from an inverted index (see api.bm25) rather than a full corpus scan.
"""

import os
import pickle
from typing import Dict, List, Tuple

from api.bm25 import PostingsBM25
from api.chunk_store import ChunkStore, load_shared
from api.search_backends.base import KeywordSearchBackend

INDEX_FORMAT_VERSION = 3


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
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump(
                {
                    "index": self._index.to_dict(),
                    "version": INDEX_FORMAT_VERSION,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load index from {path}.bm25."""
        bm25_path = f"{path}.bm25"
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"BM25 index '{bm25_path}' was built by an older version "
                f"(format {data.get('version', 1)}, expected {INDEX_FORMAT_VERSION}). "
                "Re-ingest the collection."
            )
        self._index = PostingsBM25.from_dict(data["index"])
        # Chunk text comes from the store the vector index also reads, so the
        # corpus is held once in this process rather than once per index.
        self._texts = load_shared(path).texts
