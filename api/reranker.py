"""
Lightweight re-ranker using cross-encoder for second-stage filtering.
Re-ranks hybrid search results to ensure highest quality chunks are prioritized.
Supports multiple model variants for different deployment scenarios.
"""

from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# Supported reranker models by complexity level
RERANKER_MODELS = {
    "lightweight": "ms-marco-MiniLM-L-6-v2",  # 22M params, ~50ms per batch
    "balanced": "ms-marco-MiniLM-L-12-v2",  # 71M params, ~100ms per batch
    "production": "ms-marco-MiniLM-L-12-v2",  # Use balanced for production (best tradeoff)
}


class CrossEncoderReranker:
    """Re-rank search results using a cross-encoder model."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model identifier for cross-encoder.
                       ms-marco-MiniLM-L-6-v2 is lightweight and fast.
        """
        self.model_name = model_name
        self.model = None

        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
            except Exception as e:
                print(f"Warning: Could not load cross-encoder model: {e}")
                self.model = None

    def is_available(self) -> bool:
        """Check if reranker is available and loaded."""
        return self.model is not None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents by relevance to the query.

        Args:
            query: Search query
            documents: List of document chunks to rerank
            top_k: If provided, return only top_k results

        Returns:
            List of (document, score) tuples, sorted by score descending.
            If reranker unavailable, returns original order with score=1.0.
        """
        if not documents:
            return []

        if not self.is_available():
            # Fallback: return originals with neutral scores
            return [(doc, 1.0) for doc in documents]

        try:
            # Prepare (query, document) pairs for cross-encoder
            pairs = [[query, doc] for doc in documents]

            # Score all pairs (returns scores between 0 and 1)
            scores = self.model.predict(pairs)

            # Zip documents with scores and sort by score descending
            scored_docs = list(zip(documents, scores.tolist()))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            if top_k:
                scored_docs = scored_docs[:top_k]

            return scored_docs

        except Exception as e:
            print(f"Warning: Reranking failed: {e}")
            # Fallback: return originals
            return [(doc, 1.0) for doc in documents]


# Global reranker instance (lazy-loaded on first use)
_reranker_instance: Optional[CrossEncoderReranker] = None
_current_model: Optional[str] = None


def get_reranker(
    enabled: bool = True, model_variant: str = "lightweight"
) -> Optional[CrossEncoderReranker]:
    """
    Get or initialize the global reranker instance.

    Args:
        enabled: Whether to initialize reranker (if available)
        model_variant: 'lightweight', 'balanced', or specific model name

    Returns:
        Reranker instance or None if disabled or unavailable.
    """
    global _reranker_instance, _current_model

    if not enabled:
        return None

    # Resolve model name
    model_name = RERANKER_MODELS.get(model_variant, model_variant)

    # If instance already loaded and model matches, return it
    if _reranker_instance is not None and _current_model == model_name:
        return _reranker_instance

    # Otherwise, create new instance with requested model
    try:
        _reranker_instance = CrossEncoderReranker(model_name)
        _current_model = model_name
    except Exception as e:
        print(f"Warning: Could not initialize reranker with model {model_name}: {e}")
        _reranker_instance = None
        _current_model = None

    return _reranker_instance


def rerank_results(
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
    reranker_enabled: bool = True,
    reranker_model: str = "lightweight",
) -> List[str]:
    """
    Convenience function to rerank documents.

    Args:
        query: Search query
        documents: List of document chunks
        top_k: Return only top_k results
        reranker_enabled: Whether to use reranker
        reranker_model: Model variant ('lightweight', 'balanced', or model name)

    Returns:
        Reranked list of documents (or original if reranker unavailable)
    """
    if not reranker_enabled or not documents:
        return documents

    reranker = get_reranker(enabled=True, model_variant=reranker_model)
    if reranker is None or not reranker.is_available():
        return documents

    scored_docs = reranker.rerank(query, documents, top_k)
    return [doc for doc, score in scored_docs]
