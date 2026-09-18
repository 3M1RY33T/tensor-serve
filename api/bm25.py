"""
BM25 over an inverted index.

`rank_bm25` scores every document in the corpus for every query, whether or not
it contains a single query term, so latency grows linearly with the collection:
measured at 19.5ms per query over 10,399 chunks against 0.22ms for the FAISS
search beside it, and ~105ms over 80,000. Walking postings instead touches only
the documents that can score above zero.

The scoring formula is the Lucene/Robertson form with the ``+1`` smoothing,

    idf(q) = ln(1 + (N - df + 0.5) / (df + 0.5))

which is non-negative for every term, so it needs none of the negative-IDF
epsilon correction rank_bm25 carries.
"""

import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from api.lexical import tokenize, top_k_indices


class PostingsBM25:
    """BM25 scored from an inverted index, with an optional BM25+ lower bound."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._doc_ids: Dict[str, np.ndarray] = {}
        self._term_freqs: Dict[str, np.ndarray] = {}
        self._idf: Dict[str, float] = {}
        self._doc_lengths = np.zeros(0, dtype="float32")
        self._avgdl = 0.0
        self._n_docs = 0

    # ---- build ----------------------------------------------------------

    def build(self, texts: List[str]) -> None:
        """Tokenise the corpus and build the inverted index."""
        postings: Dict[str, List[Tuple[int, int]]] = {}
        lengths = np.zeros(len(texts), dtype="float32")

        for doc_id, text in enumerate(texts):
            tokens = tokenize(text)
            lengths[doc_id] = len(tokens)
            for term, freq in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_id, freq))

        self._n_docs = len(texts)
        self._doc_lengths = lengths
        self._avgdl = float(lengths.mean()) if self._n_docs else 0.0

        self._doc_ids = {}
        self._term_freqs = {}
        self._idf = {}
        for term, entries in postings.items():
            ids = np.fromiter((d for d, _ in entries), dtype=np.int64, count=len(entries))
            tfs = np.fromiter((f for _, f in entries), dtype="float32", count=len(entries))
            self._doc_ids[term] = ids
            self._term_freqs[term] = tfs
            df = len(entries)
            self._idf[term] = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))

    # ---- query ----------------------------------------------------------

    def scores(self, query: str) -> Optional[np.ndarray]:
        """
        BM25 score for every document, or None when no query term is indexed.

        None and an all-zero array mean different things: the first says the
        question has no footing in this corpus at all, which is what the
        abstention gate needs to know.
        """
        if not self._n_docs:
            return None

        terms = [t for t in tokenize(query) if t in self._idf]
        if not terms:
            return None

        scores = np.zeros(self._n_docs, dtype="float32")
        # A repeated query term contributes once per occurrence, as in rank_bm25.
        for term, count in Counter(terms).items():
            ids = self._doc_ids[term]
            tfs = self._term_freqs[term]
            norm = self.k1 * (
                1.0 - self.b + self.b * (self._doc_lengths[ids] / self._avgdl)
            )
            contribution = self._idf[term] * (tfs * (self.k1 + 1.0) / (tfs + norm))
            if self.delta:
                contribution = contribution + self._idf[term] * self.delta
            scores[ids] += count * contribution

        return scores

    def search_indices(self, query: str, top_k: int) -> List[int]:
        scores = self.scores(query)
        if scores is None:
            return []
        return top_k_indices(scores, min(top_k, self._n_docs))

    def search_scored(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        scores = self.scores(query)
        if scores is None:
            return []
        return [
            (idx, float(scores[idx]))
            for idx in top_k_indices(scores, min(top_k, self._n_docs))
        ]

    def term_evidence(self, query: str) -> Dict[str, float]:
        """IDF of each query term that exists in the corpus vocabulary."""
        return {
            term: self._idf[term]
            for term in set(tokenize(query))
            if term in self._idf and self._idf[term] > 0
        }

    # ---- persistence ----------------------------------------------------
    #
    # Stored as flat arrays plus a JSON term list rather than a dict of arrays,
    # so the file needs no pickle to load: an index is derived from a ZIM
    # someone downloaded, and unpickling such a file runs whatever is in it.
    # Flat arrays also load far faster than tens of thousands of small ones.

    def to_arrays(self) -> dict:
        terms = list(self._doc_ids)
        term_offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        if terms:
            np.cumsum([len(self._doc_ids[t]) for t in terms], out=term_offsets[1:])

        doc_ids = (
            np.concatenate([self._doc_ids[t] for t in terms])
            if terms
            else np.zeros(0, dtype=np.int64)
        )
        term_freqs = (
            np.concatenate([self._term_freqs[t] for t in terms])
            if terms
            else np.zeros(0, dtype="float32")
        )
        idf = np.asarray([self._idf[t] for t in terms], dtype="float32")

        return {
            "terms": np.frombuffer(
                json.dumps(terms, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
            ),
            "term_offsets": term_offsets,
            "doc_ids": doc_ids,
            "term_freqs": term_freqs,
            "idf": idf,
            "doc_lengths": self._doc_lengths,
            "params": np.asarray(
                [self._avgdl, self._n_docs, self.k1, self.b, self.delta], dtype="float64"
            ),
        }

    @classmethod
    def from_arrays(cls, data) -> "PostingsBM25":
        params = data["params"]
        index = cls(k1=float(params[2]), b=float(params[3]), delta=float(params[4]))
        index._avgdl = float(params[0])
        index._n_docs = int(params[1])
        index._doc_lengths = np.asarray(data["doc_lengths"], dtype="float32")

        terms = json.loads(data["terms"].tobytes().decode("utf-8") or "[]")
        offsets = data["term_offsets"]
        doc_ids = data["doc_ids"]
        term_freqs = data["term_freqs"]
        idf = data["idf"]

        index._doc_ids = {
            term: doc_ids[offsets[i] : offsets[i + 1]] for i, term in enumerate(terms)
        }
        index._term_freqs = {
            term: term_freqs[offsets[i] : offsets[i + 1]] for i, term in enumerate(terms)
        }
        index._idf = {term: float(idf[i]) for i, term in enumerate(terms)}
        return index

    @property
    def vocabulary_size(self) -> int:
        return len(self._idf)
