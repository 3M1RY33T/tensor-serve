"""
BM25+ keyword search backend.
Enhanced BM25 variant with improved term saturation and precision.
"""

import math
import os
import pickle
from collections import Counter
from typing import Dict, List

import numpy as np

from api.lexical import tokenize, top_k_indices
from api.search_backends.base import KeywordSearchBackend

INDEX_FORMAT_VERSION = 2


class BM25PlusBackend(KeywordSearchBackend):
    """
    BM25+ variant keyword search backend.
    Improved precision over standard BM25 with better term saturation.
    Better for production use and large collections.

    Scoring walks an inverted index, touching only the documents that contain a
    query term. The previous implementation scored every document by calling
    ``list.count(term)`` per term, which is O(corpus x doc_length x query_terms)
    and measured 192ms per query at only 8,000 chunks.
    """

    def __init__(self):
        self._postings: Dict[str, Dict[int, int]] = {}
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
        self._postings = {}
        self._doc_lengths = []

        for doc_id, text in enumerate(texts):
            tokens = tokenize(text)
            self._doc_lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self._postings.setdefault(term, {})[doc_id] = freq

        if not self._doc_lengths:
            self._avgdl = 0.0
            self._idf = {}
            return

        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths)

        num_docs = len(self._doc_lengths)
        self._idf = {
            term: math.log((num_docs - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for term, postings in self._postings.items()
        }

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Return top-k chunk indices ranked by BM25+ relevance."""
        scores = self._scores(query)
        if scores is None:
            return []
        return top_k_indices(scores, min(top_k, len(self._texts)))

    def _scores(self, query: str):
        """BM25+ score for every document, or None when the query has no terms."""
        if not self._texts or not self._postings:
            return None

        query_terms = [t for t in tokenize(query) if t in self._idf]
        if not query_terms:
            return None

        scores = np.zeros(len(self._texts), dtype="float64")
        doc_lengths = np.asarray(self._doc_lengths, dtype="float64")

        for term in query_terms:
            idf = self._idf[term]
            postings = self._postings[term]

            doc_ids = np.fromiter(postings.keys(), dtype=np.int64, count=len(postings))
            tfs = np.fromiter(postings.values(), dtype="float64", count=len(postings))

            denominator = tfs + self._k1 * (
                1 - self._b + self._b * (doc_lengths[doc_ids] / self._avgdl)
            )
            scores[doc_ids] += idf * (tfs * (self._k1 + 1) / denominator)

            # BM25+'s delta lower-bound is identical for every document, so it
            # shifts all scores equally and never changes their order. Added in
            # full so scores stay numerically comparable to the direct formula.
            scores += idf * self._delta

        return scores


    def search_scored(self, query: str, top_k: int):
        """Top-k (index, BM25+ score) pairs, best first."""
        scores = self._scores(query)
        if scores is None:
            return []
        return [
            (idx, float(scores[idx]))
            for idx in top_k_indices(scores, min(top_k, len(self._texts)))
        ]

    def term_evidence(self, query: str) -> dict:
        """IDF of each query term present in the corpus vocabulary."""
        return {
            t: float(self._idf[t])
            for t in set(tokenize(query))
            if t in self._idf and self._idf[t] > 0
        }

    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve text chunks at indices."""
        return [self._texts[i] for i in indices if i < len(self._texts)]

    def save(self, path: str) -> None:
        """Save index to disk as {path}.bm25plus."""
        with open(f"{path}.bm25plus", "wb") as f:
            pickle.dump(
                {
                    "postings": self._postings,
                    "texts": self._texts,
                    "idf": self._idf,
                    "doc_lengths": self._doc_lengths,
                    "avgdl": self._avgdl,
                    "k1": self._k1,
                    "b": self._b,
                    "delta": self._delta,
                    "version": INDEX_FORMAT_VERSION,
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
        if data.get("version") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"BM25+ index '{bm25plus_path}' was built by an older version "
                f"(format {data.get('version', 1)}, expected {INDEX_FORMAT_VERSION}) "
                "and uses a different tokeniser. Re-ingest the collection."
            )
        self._postings = data["postings"]
        self._texts = data["texts"]
        self._idf = data["idf"]
        self._doc_lengths = data["doc_lengths"]
        self._avgdl = data["avgdl"]
        self._k1 = data.get("k1", 1.5)
        self._b = data.get("b", 0.75)
        self._delta = data.get("delta", 1.0)
