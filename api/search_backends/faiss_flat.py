"""
FAISS Flat keyword search backend.
Standard exact L2 distance semantic search using FAISS IndexFlatL2.
"""

import os
import pickle
from typing import List

import faiss
import numpy as np

from api.search_backends.base import SemanticSearchBackend


class FAISSFlatBackend(SemanticSearchBackend):
    """
    FAISS IndexFlatL2 semantic search backend.
    Exact search with O(n) complexity. Good for up to 500K vectors.
    Lightweight and suitable for local deployments.
    """

    def __init__(self, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self._texts: List[str] = []
        self._metadata: List[dict] = []
        self.dim = dim

    def add(
        self, embeddings: List[List[float]], chunks: List[str], metadata: List = None
    ) -> None:
        """Add embeddings and chunks to index."""
        self.index.add(np.array(embeddings).astype("float32"))
        self._texts.extend(chunks)
        if metadata is None:
            metadata = [{} for _ in chunks]
        self._metadata.extend(metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Search and return top-k text chunks."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._texts):
                results.append(self._texts[idx])
        return results

    def search_indices(self, query_embedding: List[float], top_k: int = 5) -> List[int]:
        """Search and return top-k chunk indices."""
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        return [int(idx) for idx in indices[0] if 0 <= idx < len(self._texts)]

    def save(self, path: str) -> None:
        """Save index to disk as {path}.faiss_flat."""
        faiss.write_index(self.index, f"{path}.faiss_flat.index")
        with open(f"{path}.faiss_flat.pkl", "wb") as f:
            pickle.dump({"texts": self._texts, "metadata": self._metadata}, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        index_path = f"{path}.faiss_flat.index"
        pkl_path = f"{path}.faiss_flat.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS Flat index not found: {index_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"FAISS Flat metadata not found: {pkl_path}")

        self.index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._texts = data.get("texts", [])
        self._metadata = data.get("metadata", [{} for _ in self._texts])

    @property
    def texts(self) -> List[str]:
        """Get all indexed texts."""
        return self._texts

    @property
    def metadata(self) -> List[dict]:
        """Get metadata for all chunks."""
        return self._metadata
