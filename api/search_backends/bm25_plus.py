"""
BM25+ keyword search backend.

BM25+ adds a lower-bound term to BM25, so it is the shared postings index (see
api.bm25) with a non-zero delta rather than a separate scoring implementation.
"""

import os
import pickle
from typing import Dict, List, Tuple

from api.bm25 import PostingsBM25
from api.chunk_store import load_shared
from api.search_backends.base import KeywordSearchBackend

INDEX_FORMAT_VERSION = 3
DELTA = 1.0


class BM25PlusBackend(KeywordSearchBackend):
    """
    BM25+ variant keyword search backend.

    The delta term is identical for every document given a query, so it shifts
    all scores equally and never changes their order; it is kept so scores stay
    comparable with the published formula.
    """

    def __init__(self):
        self._index = PostingsBM25(delta=DELTA)
        self._texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        self._texts = texts
        self._index.build(texts)

    def search_indices(self, query: str, top_k: int) -> List[int]:
        return self._index.search_indices(query, top_k)

    def search_scored(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        return self._index.search_scored(query, top_k)

    def term_evidence(self, query: str) -> Dict[str, float]:
        return self._index.term_evidence(query)

    def get_texts(self, indices: List[int]) -> List[str]:
        return [self._texts[i] for i in indices if i < len(self._texts)]

    @property
    def texts(self) -> List[str]:
        return self._texts

    def save(self, path: str) -> None:
        """Persist the postings to {path}.bm25plus; text lives in {path}.chunks."""
        with open(f"{path}.bm25plus", "wb") as f:
            pickle.dump(
                {"index": self._index.to_dict(), "version": INDEX_FORMAT_VERSION}, f
            )

    def load(self, path: str) -> None:
        bm25plus_path = f"{path}.bm25plus"
        if not os.path.exists(bm25plus_path):
            raise FileNotFoundError(f"BM25+ index not found: {bm25plus_path}")
        with open(bm25plus_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"BM25+ index '{bm25plus_path}' was built by an older version "
                f"(format {data.get('version', 1)}, expected {INDEX_FORMAT_VERSION}). "
                "Re-ingest the collection."
            )
        self._index = PostingsBM25.from_dict(data["index"])
        self._texts = load_shared(path).texts
