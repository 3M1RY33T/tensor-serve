"""
FAISS Flat keyword search backend.
Standard exact L2 distance semantic search using FAISS IndexFlatL2.
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from api.chunk_store import ChunkStore, load_shared
from api.search_backends.base import SemanticSearchBackend


def _unit(query_embedding):
    """
    Return the query as a unit vector, and whether it had a direction at all.

    The cos = 1 - d/2 identity holds only when both vectors are unit length.
    The index always stores normalised vectors, but a caller can pass anything:
    an unnormalised query silently yields a wrong similarity, and a zero vector
    yields 0.5 — squarely inside the band that passes a semantic gate.
    """
    vector = np.asarray([query_embedding], dtype="float32")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector, False
    return vector / norm, True



class FAISSFlatBackend(SemanticSearchBackend):
    """
    FAISS IndexFlatL2 semantic search backend.
    Exact search with O(n) complexity. Good for up to 500K vectors.
    Lightweight and suitable for local deployments.
    """

    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self._store = ChunkStore()
        self.dim = dim

    def add(
        self, embeddings: List[List[float]], chunks: List[str], metadata: List = None
    ) -> None:
        """Add embeddings and chunks to index."""
        self.index.add(np.array(embeddings).astype("float32"))
        self._store.extend(chunks, metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Search and return top-k text chunks."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._store.texts):
                results.append(self._store.texts[idx])
        return results

    def search_indices(self, query_embedding: List[float], top_k: int = 5) -> List[int]:
        """Search and return top-k chunk indices."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        return [int(idx) for idx in indices[0] if 0 <= idx < len(self._store.texts)]


    def search_scored(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Top-k (index, cosine similarity) pairs.

        The index stores unit-length vectors (the embedding model ends in a
        Normalize layer), and IndexFlatL2 reports *squared* L2 distance, so
        ||a-b||^2 = 2 - 2cos gives cos = 1 - d/2 exactly.
        """
        query, has_direction = _unit(query_embedding)
        if not has_direction:
            return []

        distances, indices = self.index.search(query, top_k)
        return [
            (int(idx), float(1.0 - dist / 2.0))
            for dist, idx in zip(distances[0], indices[0])
            if 0 <= idx < len(self._store.texts)
        ]

    def save(self, path: str) -> None:
        """
        Save vectors to {path}.faiss_flat.index and chunk text to {path}.chunks.

        Chunk text lives in the shared store rather than beside the vectors, so
        the keyword index does not have to keep a second copy of it.
        """
        faiss.write_index(self.index, f"{path}.faiss_flat.index")
        self._store.save(path)

    def load(self, path: str) -> None:
        """Load vectors, and attach the shared chunk store."""
        index_path = f"{path}.faiss_flat.index"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS Flat index not found: {index_path}")

        self.index = faiss.read_index(index_path)
        self._store = load_shared(path)

    @property
    def texts(self) -> List[str]:
        """Get all indexed texts."""
        return self._store.texts

    @property
    def metadata(self) -> List[dict]:
        """Get metadata for all chunks."""
        return self._store.metadata

    @property
    def store(self) -> ChunkStore:
        """The shared chunk store backing this index."""
        return self._store

    def attach(self, store: ChunkStore) -> None:
        """Point this index at an already-loaded chunk store."""
        self._store = store
