"""
Query and embedding caching to reduce redundant computation.
"""

import hashlib
import time
from collections import OrderedDict
from typing import List, Optional, Tuple


class QueryCache:
    """LRU cache for query embeddings and hybrid search results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached queries (LRU eviction after this)
            ttl_seconds: Time-to-live for cached results (3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.embedding_cache = OrderedDict()
        self.search_cache = OrderedDict()

    def cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding for a query string."""
        key = self._hash_text(text)
        self.embedding_cache[key] = (embedding, time.time())
        # Keep LRU
        if len(self.embedding_cache) > self.max_size:
            self.embedding_cache.popitem(last=False)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding if available and not expired."""
        key = self._hash_text(text)
        if key not in self.embedding_cache:
            return None

        embedding, cached_time = self.embedding_cache[key]
        if time.time() - cached_time > self.ttl_seconds:
            del self.embedding_cache[key]
            return None

        # Move to end (most recently used)
        self.embedding_cache.move_to_end(key)
        return embedding

    def cache_search_result(
        self,
        query: str,
        search_mode: str,
        top_k: int,
        results: List[str],
    ) -> None:
        """Cache a search result."""
        key = self._hash_search_key(query, search_mode, top_k)
        self.search_cache[key] = (results, time.time())
        # Keep LRU
        if len(self.search_cache) > self.max_size:
            self.search_cache.popitem(last=False)

    def get_search_result(
        self,
        query: str,
        search_mode: str,
        top_k: int,
    ) -> Optional[List[str]]:
        """Retrieve cached search result if available and not expired."""
        key = self._hash_search_key(query, search_mode, top_k)
        if key not in self.search_cache:
            return None

        results, cached_time = self.search_cache[key]
        if time.time() - cached_time > self.ttl_seconds:
            del self.search_cache[key]
            return None

        # Move to end (most recently used)
        self.search_cache.move_to_end(key)
        return results

    def clear(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.search_cache.clear()

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "search_cache_size": len(self.search_cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }

    @staticmethod
    def _hash_text(text: str) -> str:
        """Hash a text query for cache key."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    @staticmethod
    def _hash_search_key(query: str, search_mode: str, top_k: int) -> str:
        """Hash a search key combining query, mode, and top_k."""
        key_str = f"{query.lower().strip()}:{search_mode}:{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()


# Global cache instance (will be initialized in main.py)
query_cache = QueryCache(max_size=100, ttl_seconds=3600)
