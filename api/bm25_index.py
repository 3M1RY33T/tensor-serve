"""
BM25 keyword index — built alongside the FAISS vector index during ingestion
and used by the hybrid search pipeline.
"""

import os
import pickle
from typing import List

from rank_bm25 import BM25Okapi

from api.lexical import tokenize, top_k_indices

# Bumped when the tokeniser changes, so a stale index is rebuilt rather than
# queried with a tokeniser it was not built with.
INDEX_FORMAT_VERSION = 2


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self.texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        """Tokenise texts and build the BM25 index."""
        self.texts = texts
        tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def save(self, path: str) -> None:
        """Persist the index to {path}.bm25."""
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "texts": self.texts,
                    "version": INDEX_FORMAT_VERSION,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load a previously saved index from {path}.bm25."""
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
        self.texts = data["texts"]

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """
        Return top_k chunk indices ranked by BM25 relevance (best first).
        Returns an empty list if the index has not been built or loaded.
        """
        if self._bm25 is None or not self.texts:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        return top_k_indices(scores, min(top_k, len(self.texts)))

    def get_texts(self, indices: List[int]) -> List[str]:
        """Return the text chunks at the given indices."""
        return [self.texts[i] for i in indices if i < len(self.texts)]
