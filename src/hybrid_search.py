"""
Reciprocal Rank Fusion (RRF) and hybrid search combining FAISS + BM25 + Web Search.
"""

from typing import Dict, List, Optional, Tuple


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

    Args:
        query:                Raw text query (tokenised for BM25).
        query_embedding:      Pre-computed embedding vector (used for FAISS).
        vectordb:             VectorDB instance, or None.
        bm25_index:           BM25Index instance, or None.
        top_k:                Number of final results to return.
        candidate_k:          Candidates fetched from each index (default top_k × 3).
        rrf_k:                RRF constant (default 60).
        relevance_threshold:  Minimum RRF score to include chunk (0.0 = no filtering).
        search_mode:          'hybrid' (both), 'faiss' (semantic only), or 'bm25' (keyword only).
        web_results:          Optional list of web search results to merge.

    Returns:
        List of up to top_k text chunks, best match first.
    """
    if candidate_k is None:
        candidate_k = top_k * 3

    ranked_lists: List[List[int]] = []
    texts: List[str] = []

    # --- FAISS semantic search (if mode allows) ---
    if search_mode in ("hybrid", "faiss") and vectordb is not None:
        faiss_indices = vectordb.search_indices(query_embedding, candidate_k)
        if faiss_indices:
            ranked_lists.append(faiss_indices)
            texts = vectordb.texts

    # --- BM25 keyword search (if mode allows) ---
    if search_mode in ("hybrid", "bm25") and bm25_index is not None:
        bm25_indices = bm25_index.search_indices(query, candidate_k)
        if bm25_indices:
            ranked_lists.append(bm25_indices)
            if not texts:
                texts = bm25_index.texts

    # --- Web search results (optional, always in hybrid mode if available) ---
    if web_results:
        # Convert web results to text chunks and add to ranking
        web_texts = [result.to_chunk_text() for result in web_results]
        web_indices = list(range(len(texts), len(texts) + len(web_texts)))
        if web_indices:
            ranked_lists.append(web_indices)
            texts.extend(web_texts)

    if not ranked_lists or not texts:
        return []

    # --- Reciprocal Rank Fusion (only if hybrid mode with both indexes) ---
    if len(ranked_lists) == 1:
        fused_ranked = [(idx, 1.0 / (rrf_k + rank)) for rank, idx in enumerate(ranked_lists[0], start=1)]
    else:
        fused_ranked = reciprocal_rank_fusion(ranked_lists, k=rrf_k)

    # Deduplicate, apply threshold, preserve order, trim to top_k
    seen: set = set()
    result: List[str] = []
    for idx, score in fused_ranked:
        if idx in seen or idx >= len(texts):
            continue
        if score < relevance_threshold:
            break
        seen.add(idx)
        result.append(texts[idx])
        if len(result) >= top_k:
            break

    return result
