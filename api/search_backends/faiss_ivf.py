"""
FAISS IVF (Inverted File) semantic search backend.
Approximate nearest neighbor search with O(n/k) complexity.
Scales to 500K+ vectors with 50% speed improvement and 20% memory savings.
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

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


# FAISS wants roughly this many training points per centroid before its k-means
# is meaningful; below it, the index is fitted to noise.
_POINTS_PER_CENTROID = 39

# Vectors buffered before training. Training on the first 100-chunk ingest batch
# fitted centroids to whichever articles happened to be read first.
_MIN_TRAINING_VECTORS = 4096

# Cells probed per query. FAISS defaults to 1, which reads a single cell of the
# index and drops recall for no measurable latency saving.
DEFAULT_NPROBE = 8


class FAISSIVFBackend(SemanticSearchBackend):
    """
    FAISS IndexIVFFlat semantic search backend.
    Approximate search using inverted index clustering.
    Ideal for large collections (500K+ vectors) on servers.
    """

    def __init__(self, dim: int = 384, n_clusters: int = None, nprobe: int = DEFAULT_NPROBE):
        self.dim = dim
        self.n_clusters = n_clusters  # None = derive from the corpus at training time
        self.nprobe = nprobe
        self.quantizer = faiss.IndexFlatL2(dim)
        self.index = None
        self._texts: List[str] = []
        self._metadata: List[dict] = []
        self._is_trained = False
        self._pending: List[np.ndarray] = []

    # ---- training -------------------------------------------------------

    def _clusters_for(self, n_vectors: int) -> int:
        """
        Cluster count sized to the corpus actually ingested.

        Previously hardcoded to sqrt(100000) = 316 regardless of corpus size, so
        a small collection was split into more cells than it had vectors to fill.
        """
        if self.n_clusters:
            return max(1, min(int(self.n_clusters), max(1, n_vectors)))
        by_size = int(np.sqrt(max(1, n_vectors)))
        by_training = max(1, n_vectors // _POINTS_PER_CENTROID)
        return max(1, min(by_size, by_training))

    def _train_and_flush(self) -> None:
        """Train on everything buffered so far, then add it all."""
        if not self._pending:
            return

        vectors = np.vstack(self._pending)
        self._pending = []

        if not self._is_trained:
            self.n_clusters = self._clusters_for(len(vectors))
            self.index = faiss.IndexIVFFlat(self.quantizer, self.dim, self.n_clusters)
            self.index.train(vectors)
            self.index.nprobe = self.nprobe
            self._is_trained = True

        self.index.add(vectors)

    def _flush(self) -> None:
        """Ensure every buffered vector has reached the index."""
        if self._pending:
            self._train_and_flush()

    # ---- writes ---------------------------------------------------------

    def add(
        self, embeddings: List[List[float]], chunks: List[str], metadata: List = None
    ) -> None:
        """Add embeddings and chunks to index."""
        embeddings_array = np.asarray(embeddings, dtype="float32")
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        self._texts.extend(chunks)
        if metadata is None:
            metadata = [{} for _ in chunks]
        self._metadata.extend(metadata)

        if self._is_trained:
            self.index.add(embeddings_array)
            return

        # Buffer until there is enough material to fit centroids on.
        self._pending.append(embeddings_array)
        if sum(len(p) for p in self._pending) >= _MIN_TRAINING_VECTORS:
            self._train_and_flush()

    # ---- reads ----------------------------------------------------------

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Search and return top-k text chunks."""
        indices = self.search_indices(query_embedding, top_k)
        return [self._texts[idx] for idx in indices]

    def search_indices(self, query_embedding: List[float], top_k: int = 5) -> List[int]:
        """Search and return top-k chunk indices."""
        self._flush()
        if not self._is_trained or self.index is None or self.index.ntotal == 0:
            return []

        self.index.nprobe = self.nprobe
        query_array = np.asarray([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_array, top_k)
        return [int(idx) for idx in indices[0] if 0 <= idx < len(self._texts)]

    # ---- persistence ----------------------------------------------------


    def search_scored(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Top-k (index, cosine similarity) pairs.

        The index stores unit-length vectors (the embedding model ends in a
        Normalize layer), and IndexFlatL2 reports *squared* L2 distance, so
        ||a-b||^2 = 2 - 2cos gives cos = 1 - d/2 exactly.
        """
        self._flush()
        if not self._is_trained or self.index is None or self.index.ntotal == 0:
            return []

        self.index.nprobe = self.nprobe
        query, has_direction = _unit(query_embedding)
        if not has_direction:
            return []

        distances, indices = self.index.search(query, top_k)
        return [
            (int(idx), float(1.0 - dist / 2.0))
            for dist, idx in zip(distances[0], indices[0])
            if 0 <= idx < len(self._texts)
        ]

    def save(self, path: str) -> None:
        """Save index to disk as {path}.faiss_ivf."""
        self._flush()
        if self.index is None:
            raise ValueError("Cannot save an IVF index with no vectors.")
        faiss.write_index(self.index, f"{path}.faiss_ivf.index")
        with open(f"{path}.faiss_ivf.pkl", "wb") as f:
            pickle.dump(
                {
                    "texts": self._texts,
                    "metadata": self._metadata,
                    "n_clusters": self.n_clusters,
                    "is_trained": self._is_trained,
                    "nprobe": self.nprobe,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load index from disk."""
        index_path = f"{path}.faiss_ivf.index"
        pkl_path = f"{path}.faiss_ivf.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS IVF index not found: {index_path}")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"FAISS IVF metadata not found: {pkl_path}")

        self.index = faiss.read_index(index_path)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._texts = data.get("texts", [])
        self._metadata = data.get("metadata", [{} for _ in self._texts])
        self.n_clusters = data.get("n_clusters", self.n_clusters)
        self._is_trained = data.get("is_trained", True)
        self.nprobe = data.get("nprobe", self.nprobe)
        self._pending = []
        self.index.nprobe = self.nprobe

    @property
    def texts(self) -> List[str]:
        """Get all indexed texts."""
        return self._texts

    @property
    def metadata(self) -> List[dict]:
        """Get metadata for all chunks."""
        return self._metadata
