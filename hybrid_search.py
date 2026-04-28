"""
Reciprocal Rank Fusion (RRF) and hybrid search combining FAISS + BM25.
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
) -> List[str]:
    """
    Hybrid retrieval: FAISS semantic search + BM25 keyword search via RRF.

    Both indexes retrieve candidate_k results (default: top_k × 3) so RRF has
    enough candidates to re-rank. The final list is trimmed to top_k.

    Gracefully degrades:
    - No BM25 index → pure semantic search
    - No FAISS index → pure keyword search
    - Neither        → empty list

    Args:
        query:           Raw text query (tokenised for BM25).
        query_embedding: Pre-computed embedding vector (used for FAISS).
        vectordb:        VectorDB instance, or None.
        bm25_index:      BM25Index instance, or None.
        top_k:           Number of final results to return.
        candidate_k:     Candidates fetched from each index (default top_k × 3).
        rrf_k:           RRF constant (default 60).

    Returns:
        List of up to top_k text chunks, best match first.
    """
    if candidate_k is None:
        candidate_k = top_k * 3

    ranked_lists: List[List[int]] = []
    texts: List[str] = []

    # --- FAISS semantic search --------------------------------------------
    if vectordb is not None:
        faiss_indices = vectordb.search_indices(query_embedding, candidate_k)
        if faiss_indices:
            ranked_lists.append(faiss_indices)
            texts = vectordb.texts

    # --- BM25 keyword search ----------------------------------------------
    if bm25_index is not None:
        bm25_indices = bm25_index.search_indices(query, candidate_k)
        if bm25_indices:
            ranked_lists.append(bm25_indices)
            if not texts:
                texts = bm25_index.texts

    if not ranked_lists or not texts:
        return []

    # --- Reciprocal Rank Fusion -------------------------------------------
    if len(ranked_lists) == 1:
        fused_indices = [idx for idx in ranked_lists[0]]
    else:
        fused_indices = [
            idx for idx, _ in reciprocal_rank_fusion(ranked_lists, k=rrf_k)
        ]

    # Deduplicate, preserve order, trim to top_k
    seen: set = set()
    result: List[str] = []
    for idx in fused_indices:
        if idx in seen or idx >= len(texts):
            continue
        seen.add(idx)
        result.append(texts[idx])
        if len(result) >= top_k:
            break

    return result
