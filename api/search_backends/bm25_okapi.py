"""
BM25 Okapi keyword search backend.
Uses rank-bm25 library. Standard baseline for keyword search.
"""

import os
import pickle
from typing import List

from rank_bm25 import BM25Okapi

from api.lexical import tokenize, top_k_indices
from api.search_backends.base import KeywordSearchBackend

INDEX_FORMAT_VERSION = 2


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
        tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Return top-k chunk indices ranked by BM25 relevance."""
        if self._bm25 is None or not self._texts:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        return top_k_indices(scores, min(top_k, len(self._texts)))


    def search_scored(self, query: str, top_k: int):
        """Top-k (index, BM25 score) pairs, best first."""
        if self._bm25 is None or not self._texts:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        return [
            (idx, float(scores[idx]))
            for idx in top_k_indices(scores, min(top_k, len(self._texts)))
        ]

    def term_evidence(self, query: str) -> dict:
        """
        IDF of each query term that exists in the corpus at all.

        What separates a real question from nonsense is whether its words are
        in the corpus, not how the scores are spread — margin and ratio tests
        invert on real data. Terms absent from the vocabulary are simply
        missing from this mapping, so summed evidence is zero for gibberish.
        """
        if self._bm25 is None:
            return {}
        idf = getattr(self._bm25, "idf", {}) or {}
        return {t: float(idf[t]) for t in set(tokenize(query)) if t in idf and idf[t] > 0}

    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve text chunks at indices."""
        return [self._texts[i] for i in indices if i < len(self._texts)]

    def save(self, path: str) -> None:
        """Save index to disk as {path}.bm25."""
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "texts": self._texts,
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
                f"(format {data.get('version', 1)}, expected {INDEX_FORMAT_VERSION}) "
                "and uses a different tokeniser. Re-ingest the collection."
            )
        self._bm25 = data["bm25"]
        self._texts = data["texts"]
