"""
BM25+ keyword search backend.
Enhanced BM25 variant with improved term saturation and precision.
"""

import os
import pickle
from typing import List

import numpy as np

from api.search_backends.base import KeywordSearchBackend


class BM25PlusBackend(KeywordSearchBackend):
    """
    BM25+ variant keyword search backend.
    Improved precision over standard BM25 with better term saturation.
    Better for production use and large collections.
    """

    def __init__(self):
        self._tokenized_docs: List[List[str]] = []
        self._texts: List[str] = []
        self._idf: dict = {}
        self._doc_lengths: List[int] = []
        self._avgdl: float = 0.0
        self._k1 = 1.5
        self._b = 0.75
        self._delta = 1.0

    def build(self, texts: List[str]) -> None:
        """Build BM25+ index from texts."""
        self._texts = texts
        self._tokenized_docs = [t.lower().split() for t in texts]
        self._doc_lengths = [len(doc) for doc in self._tokenized_docs]

        if not self._doc_lengths:
            self._avgdl = 0.0
            return

        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths)

        # Calculate IDF for each term
        num_docs = len(self._tokenized_docs)
        doc_freqs = {}

        for doc in self._tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        for term, freq in doc_freqs.items():
            self._idf[term] = np.log(
                (num_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Return top-k chunk indices ranked by BM25+ relevance."""
        if not self._tokenized_docs or not self._texts:
            return []

        query_terms = query.lower().split()
        scores = [self._score_doc(i, query_terms) for i in range(len(self._texts))]

        top = int(min(top_k, len(self._texts)))
        return [int(i) for i in np.argsort(scores)[::-1][:top]]

    def _score_doc(self, doc_idx: int, query_terms: List[str]) -> float:
        """Calculate BM25+ score for a document."""
        score = 0.0
        doc = self._tokenized_docs[doc_idx]
        doc_len = self._doc_lengths[doc_idx]

        for term in query_terms:
            if term not in self._idf:
                continue

            term_freq = doc.count(term)
            idf = self._idf[term]

            # BM25+ formula with delta term
            numerator = term_freq * (self._k1 + 1)
            denominator = (
                term_freq
                + self._k1
                * (1 - self._b + self._b * (doc_len / self._avgdl))
            )
            score += idf * (numerator / denominator + self._delta)

        return score

    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve text chunks at indices."""
        return [self._texts[i] for i in indices if i < len(self._texts)]

    def save(self, path: str) -> None:
        """Save index to disk as {path}.bm25plus."""
        with open(f"{path}.bm25plus", "wb") as f:
            pickle.dump(
                {
                    "tokenized_docs": self._tokenized_docs,
                    "texts": self._texts,
                    "idf": self._idf,
                    "doc_lengths": self._doc_lengths,
                    "avgdl": self._avgdl,
                    "k1": self._k1,
                    "b": self._b,
                    "delta": self._delta,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load index from {path}.bm25plus."""
        bm25plus_path = f"{path}.bm25plus"
        if not os.path.exists(bm25plus_path):
            raise FileNotFoundError(f"BM25+ index not found: {bm25plus_path}")
        with open(bm25plus_path, "rb") as f:
            data = pickle.load(f)
        self._tokenized_docs = data["tokenized_docs"]
        self._texts = data["texts"]
        self._idf = data["idf"]
        self._doc_lengths = data["doc_lengths"]
        self._avgdl = data["avgdl"]
        self._k1 = data.get("k1", 1.5)
        self._b = data.get("b", 0.75)
        self._delta = data.get("delta", 1.0)
