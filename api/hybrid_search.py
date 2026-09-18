"""
Reciprocal Rank Fusion (RRF) and hybrid search combining FAISS + BM25 + Web Search.
Supports pluggable backends and query expansion strategies.
"""

from typing import Dict, List, Optional, Tuple

from api.query_expansion import get_expander


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked lists of chunk indices using Reciprocal Rank Fusion.

    Formula:  score(d) = Σ  1 / (k + rank(d))   for each list containing d
    Rank is 1-based. k=60 is the standard default from the original RRF paper.

    Args:
        ranked_lists: Each inner list is a sequence of chunk indices, best first.
        k:            RRF constant — higher values dampen the impact of top ranks.

    Returns:
        List of (chunk_index, rrf_score) sorted by score descending.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def max_rrf_score(num_ranked_lists: int, rrf_k: int = 60) -> float:
    """
    Highest score RRF can produce, for a chunk ranked first in every list.

    RRF scores are rank-derived and bounded, not relevance values: with the
    default rrf_k of 60 even a unanimous first place scores only
    ``num_lists / 61``. A threshold set above this silently empties every
    result set, so callers validating a configured threshold check it here.
    """
    return num_ranked_lists / (rrf_k + 1.0)


def _expand_query(
    query: str,
    expansion_type: str,
    vectordb,
    bm25_index,
) -> str:
    """
    Expand a query before keyword retrieval.

    Pseudo-relevance feedback needs a first retrieval pass to feed back from;
    without one it returns the query unchanged, which is what the 'production'
    profile was silently doing. Run that pass here and hand the expander its
    top-1 result.
    """
    expander = get_expander(expansion_type)

    if expansion_type != "prf":
        return expander.expand(query)

    top_result = None
    if bm25_index is not None:
        hits = bm25_index.search_indices(query, 1)
        if hits:
            texts = bm25_index.texts
            if hits[0] < len(texts):
                top_result = texts[hits[0]]

    return expander.expand(query, top_result=top_result)


def hybrid_search(
    query: str,
    query_embedding,
    vectordb,
    bm25_index,
    top_k: int = 5,
    candidate_k: Optional[int] = None,
    rrf_k: int = 60,
    relevance_threshold: float = 0.0,
    search_mode: str = "hybrid",
    web_results: Optional[List] = None,
    query_expansion_enabled: bool = False,
    query_expansion_type: str = "none",
) -> List[str]:
    """
    Hybrid retrieval: FAISS semantic search + BM25 keyword search + optional Web Search via RRF.

    Both indexes retrieve candidate_k results (default: top_k × 3) so RRF has
    enough candidates to re-rank. The final list is trimmed to top_k.

    Gracefully degrades:
    - No BM25 index → pure semantic search
    - No FAISS index → pure keyword search
    - Neither        → web search only (if available)
    - No web search  → local search only
    - search_mode=None → no search, return empty list

    Args:
        query:                    Raw text query (tokenised for BM25).
        query_embedding:          Pre-computed embedding vector (used for FAISS).
        vectordb:                 VectorDB instance, or None.
        bm25_index:               BM25Index instance, or None.
        top_k:                    Number of final results to return.
        candidate_k:              Candidates fetched from each index (default top_k × 3).
        rrf_k:                    RRF constant (default 60).
        relevance_threshold:      Minimum RRF score to include chunk (0.0 = no filtering).
                                  RRF scores are bounded by max_rrf_score(); a higher
                                  threshold returns nothing.
        search_mode:              'hybrid', 'faiss', 'bm25', 'web', 'hybrid_web', or None (disabled).
        web_results:              Optional list of web search results to merge.
        query_expansion_enabled:  Whether to expand query before retrieval.
        query_expansion_type:     'none', 'prf', or 'entity' expansion strategy.

    Returns:
        List of up to top_k text chunks, best match first.
    """
    # If search is disabled, return empty list
    if search_mode is None:
        return []

    if candidate_k is None:
        candidate_k = top_k * 3

    # --- Optional query expansion ---
    expanded_query = query
    if query_expansion_enabled and query_expansion_type != "none":
        expanded_query = _expand_query(
            query, query_expansion_type, vectordb, bm25_index
        )

    ranked_lists: List[List[int]] = []

    # Chunk texts belonging to the loaded index. Never mutated: this is the
    # backend's own list, and appending web results to it corrupted the index
    # by growing it out of alignment with index.ntotal and metadata.
    base_texts: List[str] = []

    # --- FAISS semantic search (if mode allows) ---
    if search_mode in ("hybrid", "faiss", "hybrid_web") and vectordb is not None:
        faiss_indices = vectordb.search_indices(query_embedding, candidate_k)
        if faiss_indices:
            ranked_lists.append(faiss_indices)
            base_texts = vectordb.texts

    # --- BM25 keyword search (if mode allows) ---
    if search_mode in ("hybrid", "bm25") and bm25_index is not None:
        bm25_indices = bm25_index.search_indices(expanded_query, candidate_k)
        if bm25_indices:
            ranked_lists.append(bm25_indices)
            if not base_texts:
                base_texts = bm25_index.texts

    # --- Web search results (optional, merged whenever they are available) ---
    # Web chunks occupy index space immediately after the indexed chunks, so
    # RRF can rank them alongside without either list being rewritten.
    web_texts: List[str] = []
    if web_results:
        web_texts = [result.to_chunk_text() for result in web_results]
        offset = len(base_texts)
        web_indices = list(range(offset, offset + len(web_texts)))
        if web_indices:
            ranked_lists.append(web_indices)

    if not ranked_lists or (not base_texts and not web_texts):
        return []

    # --- Reciprocal Rank Fusion (only if hybrid mode with both indexes) ---
    if len(ranked_lists) == 1:
        fused_ranked = [(idx, 1.0 / (rrf_k + rank)) for rank, idx in enumerate(ranked_lists[0], start=1)]
    else:
        fused_ranked = reciprocal_rank_fusion(ranked_lists, k=rrf_k)

    # Deduplicate, apply threshold, preserve order, trim to top_k
    total = len(base_texts) + len(web_texts)
    seen: set = set()
    result: List[str] = []
    for idx, score in fused_ranked:
        if idx in seen or idx >= total:
            continue
        if score < relevance_threshold:
            break
        seen.add(idx)
        result.append(
            base_texts[idx] if idx < len(base_texts) else web_texts[idx - len(base_texts)]
        )
        if len(result) >= top_k:
            break

    return result
