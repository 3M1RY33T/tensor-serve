import os
import pickle

import faiss
import numpy as np

from api.search_backends import get_semantic_backend


class VectorDB:
    """Vector database with support for multiple FAISS backend variants."""

    def __init__(self, dim: int = 384, variant: str = "faiss_flat"):
        """
        Initialize vector database with specified backend variant.
        
        Args:
            dim: Embedding dimension
            variant: 'faiss_flat' (exact) or 'faiss_ivf' (approximate)
        """
        self.dim = dim
        self.variant = variant
        self.backend = get_semantic_backend(variant)(dim)

    def add(self, embeddings, chunks, metadata=None):
        """Add embeddings and chunks to index."""
        self.backend.add(embeddings, chunks, metadata)

    def save(self, path="db"):
        """Save index to disk."""
        self.backend.save(path)

    def load(self, path="db"):
        """Load index from disk."""
        self.backend.load(path)

    def search(self, query_embedding, top_k=5):
        """Search and return top-k text chunks."""
        return self.backend.search(query_embedding, top_k)

    def search_indices(self, query_embedding, top_k=5):
        """Return top-k chunk indices."""
        return self.backend.search_indices(query_embedding, top_k)

    @property
    def texts(self):
        """Get all indexed texts."""
        return self.backend.texts

    @property
    def metadata(self):
        """Get all metadata."""
        return self.backend.metadata
