"""
Search backends factory for pluggable keyword and semantic search implementations.
Enables scalable search from lightweight local deployments to production servers.
"""

from api.search_backends.base import KeywordSearchBackend, SemanticSearchBackend
from api.search_backends.bm25_okapi import BM25OkapiBackend
from api.search_backends.bm25_plus import BM25PlusBackend
from api.search_backends.faiss_flat import FAISSFlatBackend
from api.search_backends.faiss_ivf import FAISSIVFBackend

# Supported keyword search backends
KEYWORD_BACKENDS = {
    "bm25_okapi": BM25OkapiBackend,
    "bm25_plus": BM25PlusBackend,
}

# Supported semantic search backends
SEMANTIC_BACKENDS = {
    "faiss_flat": FAISSFlatBackend,
    "faiss_ivf": FAISSIVFBackend,
}


def get_keyword_backend(variant: str) -> type:
    """
    Get keyword search backend class by variant name.
    
    Args:
        variant: Backend identifier (e.g., 'bm25_okapi', 'bm25_plus')
    
    Returns:
        Backend class (not instantiated)
    
    Raises:
        ValueError: If variant not found. Falls back to BM25OkapiBackend.
    """
    if variant not in KEYWORD_BACKENDS:
        print(f"Warning: Keyword backend '{variant}' not found. Using 'bm25_okapi'.")
        return KEYWORD_BACKENDS["bm25_okapi"]
    return KEYWORD_BACKENDS[variant]


def get_semantic_backend(variant: str) -> type:
    """
    Get semantic search backend class by variant name.
    
    Args:
        variant: Backend identifier (e.g., 'faiss_flat', 'faiss_ivf')
    
    Returns:
        Backend class (not instantiated)
    
    Raises:
        ValueError: If variant not found. Falls back to FAISSFlatBackend.
    """
    if variant not in SEMANTIC_BACKENDS:
        print(f"Warning: Semantic backend '{variant}' not found. Using 'faiss_flat'.")
        return SEMANTIC_BACKENDS["faiss_flat"]
    return SEMANTIC_BACKENDS[variant]


__all__ = [
    "KeywordSearchBackend",
    "SemanticSearchBackend",
    "BM25OkapiBackend",
    "BM25PlusBackend",
    "FAISSFlatBackend",
    "FAISSIVFBackend",
    "get_keyword_backend",
    "get_semantic_backend",
    "KEYWORD_BACKENDS",
    "SEMANTIC_BACKENDS",
]
