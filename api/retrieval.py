"""
The retrieval pipeline, independent of the web layer.

Phase 1 collapsed /search and the chat proxy onto one function so they could
not drift apart. The evaluation harness needs to measure that same function
rather than a reimplementation of it, so it lives here instead of in the
FastAPI module, and api.main binds it to the server's application state.
"""

from typing import List, Optional, Tuple

from api.config import get_config_value


def retrieval_settings() -> dict:
    """
    Every setting the retrieval pipeline reads, resolved once.

    Reads with an explicit None check: `or` would turn a deliberate 0 or False
    back into the fallback, which is how a configured relevance_threshold of
    0.0 became 0.05 and emptied every result set.
    """

    def value(key, fallback):
        configured = get_config_value(key)
        return fallback if configured is None else configured

    return {
        "relevance_threshold": value("relevance_threshold", 0.0),
        "reranker_enabled": value("reranker_enabled", False),
        "reranker_model": value("reranker_model", "lightweight"),
        "keyword_search_mode": value("keyword_search_mode", "auto"),
        "semantic_search_mode": value("semantic_search_mode", "auto"),
        "query_expansion_enabled": value("query_expansion_enabled", False),
        "query_expansion_type": value("query_expansion_type", "none"),
        "max_search_candidates": get_config_value("max_search_candidates"),
        "web_search_enabled": value("web_search_enabled", False),
        "web_search_results": value("web_search_results", 3),
        "abstention_enabled": value("abstention_enabled", True),
        "lexical_evidence_floor": value("lexical_evidence_floor", 1.0),
        "semantic_confidence_floor": value("semantic_confidence_floor", 0.45),
    }


def retrieve(
    query: str,
    top_k: int,
    *,
    db,
    bm25,
    embedder,
    cache=None,
    settings: Optional[dict] = None,
    allow_web: bool = True,
    detailed: bool = False,
):
    """
    Run the retrieval pipeline for one query.

    Args:
        query:     The user's question.
        top_k:     Number of chunks to return.
        db:        VectorDB instance, or None.
        bm25:      BM25Index instance, or None.
        embedder:  Embedder used for the query vector.
        cache:     Optional QueryCache. Pass None to bypass caching, which the
                   evaluation harness does so it measures retrieval rather than
                   cache hits.
        settings:  Pre-resolved settings, to avoid re-reading config per query.
        allow_web: Set False to keep an offline evaluation offline.

        detailed:  Return the full RetrievalOutcome instead of chunk text, so
                   the caller can see scores and why the pipeline abstained.

    Returns:
        (chunks, search_mode), or (RetrievalOutcome, search_mode) when detailed.
    """
    from api.hybrid_search import search as hybrid_search
    from api.query_analyzer import QueryAnalyzer

    if settings is None:
        settings = retrieval_settings()

    search_mode = QueryAnalyzer.select_search_mode(
        query, settings["keyword_search_mode"], settings["semantic_search_mode"]
    )

    # The cache holds the whole outcome, not just the chunk text, so the chat
    # proxy — which needs candidate indices for source attribution — can be
    # served from it too. Caching text alone silently excluded it.
    if cache is not None:
        cached = cache.get_search_result(query, search_mode, top_k)
        if cached is not None:
            return (cached if detailed else cached.texts), search_mode

    query_embedding = cache.get_embedding(query) if cache is not None else None
    if query_embedding is None:
        query_embedding = embedder.encode([query])[0]
        if cache is not None:
            cache.cache_embedding(query, query_embedding)

    web_results = None
    if allow_web and settings["web_search_enabled"] and QueryAnalyzer.is_time_sensitive(query):
        from api.web_search import web_search_manager

        web_results = web_search_manager.search(
            query, num_results=settings["web_search_results"]
        )

    max_candidates = settings["max_search_candidates"]
    if max_candidates is None:
        max_candidates = top_k * 3

    outcome = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        vectordb=db,
        bm25_index=bm25,
        top_k=top_k,
        candidate_k=max(max_candidates, 10),
        relevance_threshold=settings["relevance_threshold"],
        search_mode=search_mode,
        web_results=web_results,
        query_expansion_enabled=settings["query_expansion_enabled"],
        query_expansion_type=settings["query_expansion_type"],
        abstention_enabled=settings["abstention_enabled"],
        lexical_evidence_floor=settings["lexical_evidence_floor"],
        semantic_confidence_floor=settings["semantic_confidence_floor"],
    )

    results = outcome.texts

    if settings["reranker_enabled"] and results:
        from api.reranker import rerank_results

        results = rerank_results(
            query=query,
            documents=results,
            top_k=top_k,
            reranker_enabled=True,
            reranker_model=settings["reranker_model"],
        )

        # Reranking reorders text, so keep the candidates in step with it.
        order = {text: rank for rank, text in enumerate(results)}
        outcome.candidates = sorted(
            (c for c in outcome.candidates if c.text in order),
            key=lambda c: order[c.text],
        )

    if cache is not None:
        cache.cache_search_result(query, search_mode, top_k, outcome)

    if detailed:
        return outcome, search_mode
    return results, search_mode
