"""
BM25 keyword index — built alongside the FAISS vector index during ingestion
and used by the hybrid search pipeline.

Scoring walks an inverted index (see api.bm25), so a query touches only the
chunks containing one of its terms rather than the whole collection.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from api.bm25 import PostingsBM25
from api.durability import atomic_write
from api.chunk_store import load_shared

# 2 introduced the shared tokeniser; 3 replaced rank_bm25 with postings.
INDEX_FORMAT_VERSION = 4

_MAGIC = b"TSBM25\x00\x00"

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
        payload = self._index.to_arrays()
        payload["meta"] = np.frombuffer(
            json.dumps(
                {"version": INDEX_FORMAT_VERSION, "variant": getattr(self, "variant", None)}
            ).encode("utf-8"),
            dtype=np.uint8,
        )
        with atomic_write(f"{path}.bm25", "wb") as f:
            f.write(_MAGIC)
            np.savez(f, **payload)

    def load(self, path: str) -> None:
        """Load the postings, and attach the shared chunk store."""
        bm25_path = f"{path}.bm25"
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"Keyword index not found: {bm25_path}")

        with open(bm25_path, "rb") as handle:
            if handle.read(len(_MAGIC)) != _MAGIC:
                raise ValueError(
                    f"'{bm25_path}' predates index format {INDEX_FORMAT_VERSION}. "
                    "Re-ingest the collection."
                )
            # allow_pickle stays False: this file is derived from a downloaded ZIM.
            with np.load(handle, allow_pickle=False) as data:
                meta = json.loads(data["meta"].tobytes().decode("utf-8"))
                if meta.get("version") != INDEX_FORMAT_VERSION:
                    raise ValueError(
                        f"'{bm25_path}' has index format {meta.get('version')}, "
                        f"expected {INDEX_FORMAT_VERSION}. Re-ingest the collection."
                    )
                self._index = PostingsBM25.from_arrays(data)

        self.variant = meta.get("variant") or self.variant
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
