"""
BM25 Okapi keyword search backend.
Uses rank-bm25 library. Standard baseline for keyword search.
"""

import os
import pickle
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from api.search_backends.base import KeywordSearchBackend


class BM25OkapiBackend(KeywordSearchBackend):
    """
    BM25 Okapi variant keyword search backend.
    Good baseline; efficient for up to 1M documents.
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        """Build BM25 index from texts."""
        self._texts = texts
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Return top-k chunk indices ranked by BM25 relevance."""
        if self._bm25 is None or not self._texts:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top = int(min(top_k, len(self._texts)))
        return [int(i) for i in np.argsort(scores)[::-1][:top]]

    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve text chunks at indices."""
        return [self._texts[i] for i in indices if i < len(self._texts)]

    def save(self, path: str) -> None:
        """Save index to disk as {path}.bm25."""
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump({"bm25": self._bm25, "texts": self._texts}, f)

    def load(self, path: str) -> None:
        """Load index from {path}.bm25."""
        bm25_path = f"{path}.bm25"
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._texts = data["texts"]
