"""
Abstract base classes for keyword and semantic search backends.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class KeywordSearchBackend(ABC):
    """Abstract base for keyword/BM25-style search implementations."""

    @abstractmethod
    def build(self, texts: List[str]) -> None:
        """Build the keyword search index from a list of texts."""
        pass

    @abstractmethod
    def search_indices(self, query: str, top_k: int) -> List[int]:
        """Search and return top-k chunk indices ranked by relevance."""
        pass

    def search_scored(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Search and return top-k (index, score) pairs, best first.

        Scores are the backend's own, unnormalised. Returning indices alone
        discarded the only relevance signal the pipeline had, which is why no
        threshold and no abstention were possible. Backends that do not
        override this degrade to rank-derived scores.
        """
        indices = self.search_indices(query, top_k)
        return [(idx, 1.0 / rank) for rank, idx in enumerate(indices, start=1)]

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        pass

    @abstractmethod
    def get_texts(self, indices: List[int]) -> List[str]:
        """Retrieve text chunks at the given indices."""
        pass


class SemanticSearchBackend(ABC):
    """Abstract base for semantic/vector search implementations."""

    @abstractmethod
    def add(
        self, embeddings: List[List[float]], chunks: List[str], metadata: List = None
    ) -> None:
        """Add embeddings and associated chunks to the index."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Search and return top-k text chunks."""
        pass

    @abstractmethod
    def search_indices(self, query_embedding: List[float], top_k: int = 5) -> List[int]:
        """Search and return top-k chunk indices."""
        pass

    def search_scored(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search and return top-k (index, cosine similarity) pairs, best first.

        Cosine is an absolute quantity, comparable across corpora, so it is
        what an abstention gate can threshold on.
        """
        indices = self.search_indices(query_embedding, top_k)
        return [(idx, 1.0 / rank) for rank, idx in enumerate(indices, start=1)]

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        pass

    @property
    @abstractmethod
    def texts(self) -> List[str]:
        """Get all indexed text chunks."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> List[dict]:
        """Get metadata for all indexed chunks."""
        pass
