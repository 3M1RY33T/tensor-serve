"""
Abstract base classes for keyword and semantic search backends.
"""

from abc import ABC, abstractmethod
from typing import List


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
