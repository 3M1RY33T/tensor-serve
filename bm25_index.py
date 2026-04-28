"""
BM25 keyword index — built alongside the FAISS vector index during ingestion
and used by the hybrid search pipeline.
"""

import os
import pickle
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self.texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        """Tokenise texts and build the BM25 index."""
        self.texts = texts
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def save(self, path: str) -> None:
        """Persist the index to {path}.bm25."""
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump({"bm25": self._bm25, "texts": self.texts}, f)

    def load(self, path: str) -> None:
        """Load a previously saved index from {path}.bm25."""
        bm25_path = f"{path}.bm25"
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self.texts = data["texts"]

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """
        Return top_k chunk indices ranked by BM25 relevance (best first).
        Returns an empty list if the index has not been built or loaded.
        """
        if self._bm25 is None or not self.texts:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top = int(min(top_k, len(self.texts)))
        return [int(i) for i in np.argsort(scores)[::-1][:top]]

    def get_texts(self, indices: List[int]) -> List[str]:
        """Return the text chunks at the given indices."""
        return [self.texts[i] for i in indices if i < len(self.texts)]
