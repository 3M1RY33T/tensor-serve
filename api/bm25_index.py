"""
BM25 keyword index — built alongside the FAISS vector index during ingestion
and used by the hybrid search pipeline.

Scoring walks an inverted index (see api.bm25), so a query touches only the
chunks containing one of its terms rather than the whole collection.
"""

import os
import pickle
from typing import Dict, List, Tuple

from api.bm25 import PostingsBM25
from api.chunk_store import ChunkStore, load_shared

# 2 introduced the shared tokeniser; 3 replaced rank_bm25 with postings.
INDEX_FORMAT_VERSION = 3

# BM25+ differs from Okapi only by a lower-bound term, so both are the same
# postings index with a different delta.
_VARIANT_DELTA = {"bm25_okapi": 0.0, "bm25_plus": 1.0}


def _configured_variant() -> str:
    """The keyword backend named in config, defaulting to Okapi."""
    try:
        from api.config import get_config_value

        return get_config_value("keyword_backend") or "bm25_okapi"
    except Exception:
        return "bm25_okapi"


class BM25Index:
    def __init__(self, variant: str = None):
        """
        Args:
            variant: 'bm25_okapi' or 'bm25_plus'. Defaults to the configured
                     keyword_backend — which every call site used to ignore,
                     leaving bm25_plus unreachable however the profile was set.
        """
        self.variant = variant or _configured_variant()
        if self.variant not in _VARIANT_DELTA:
            raise ValueError(
                f"Unknown keyword backend '{self.variant}'. "
                f"Choose one of: {', '.join(sorted(_VARIANT_DELTA))}"
            )
        self._index = PostingsBM25(delta=_VARIANT_DELTA[self.variant])
        self.texts: List[str] = []

    def build(self, texts: List[str]) -> None:
        """Tokenise texts and build the BM25 index."""
        self.texts = texts
        self._index.build(texts)

    def save(self, path: str) -> None:
        """
        Persist the postings to {path}.bm25.

        Chunk text is not written here: it lives in {path}.chunks, which the
        vector index writes and both indexes read.
        """
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump(
                {
                    "index": self._index.to_dict(),
                    "variant": self.variant,
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
                f"(format {data.get('version', 1)}, expected {INDEX_FORMAT_VERSION}). "
                "Re-ingest the collection."
            )
        self._index = PostingsBM25.from_dict(data["index"])
        self.variant = data.get("variant", self.variant)
        # Chunk text comes from the store the vector index also reads, so the
        # corpus is held once in this process rather than once per index.
        self.texts = load_shared(path).texts

    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Top-k chunk indices ranked by BM25 relevance (best first)."""
        return self._index.search_indices(query, top_k)

    def search_scored(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Top-k (index, BM25 score) pairs, best first."""
        return self._index.search_scored(query, top_k)

    def term_evidence(self, query: str) -> Dict[str, float]:
        """
        IDF of each query term that exists in the corpus at all.

        What separates a real question from nonsense is whether its words are in
        the corpus, not how the scores are spread — margin and ratio tests invert
        on real data. Terms absent from the vocabulary are simply missing here,
        so summed evidence is exactly zero for gibberish.
        """
        return self._index.term_evidence(query)

    def get_texts(self, indices: List[int]) -> List[str]:
        """Return the text chunks at the given indices."""
        return [self.texts[i] for i in indices if i < len(self.texts)]
