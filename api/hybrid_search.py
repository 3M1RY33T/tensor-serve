"""
Reciprocal Rank Fusion (RRF) and hybrid search combining FAISS + BM25 + Web Search.
Supports pluggable backends and query expansion strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from api.query_expansion import get_expander

# Defaults measured on a 10,399-chunk corpus of Python documentation, over the
# four question families the evaluation harness generates:
#
#   family      top cosine (min / p10 / med)   summed IDF evidence (min / med)
#   passage         0.201 / 0.412 / 0.620            6.71 / 35.65
#   title           0.318 / 0.371 / 0.577            2.27 /  7.11
#   signature       0.382 / 0.444 / 0.627            2.30 / 13.04
#   nonsense        0.308 / 0.313 / 0.331            0.00 /  0.00
#
# Evidence separates the bands cleanly — nonsense scores exactly zero because
# none of its words are in the corpus, while the weakest real question scores
# 2.27. Cosine does not: nonsense reaches 0.381 against a real question at
# 0.201, so an absolute cosine floor alone would reject real questions to
# reject gibberish. Hence evidence first, cosine only as a second chance.
DEFAULT_LEXICAL_EVIDENCE_FLOOR = 1.0
DEFAULT_SEMANTIC_CONFIDENCE_FLOOR = 0.45


@dataclass
class Candidate:
    """One retrieved chunk, with the scores that produced it."""

    index: int
    text: str
    rrf_score: float = 0.0
    cosine: Optional[float] = None
    keyword_score: Optional[float] = None
    sources: tuple = ()

    @property
    def is_web(self) -> bool:
        return "web" in self.sources


@dataclass
class RetrievalOutcome:
    """Candidates plus why the pipeline answered or abstained."""

    candidates: List[Candidate] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""
    evidence: float = 0.0
    best_cosine: float = 0.0

    @property
    def texts(self) -> List[str]:
        return [c.text for c in self.candidates]


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


def _scored(backend, probe, k, fallback_attr):
    """
    Ask a backend for (index, score) pairs, tolerating older backends.

    Test doubles and third-party backends may only implement search_indices;
    those degrade to rank-derived scores rather than failing.
    """
    if backend is None:
        return []
    if hasattr(backend, "search_scored"):
        return list(backend.search_scored(probe, k))
    getter = getattr(backend, fallback_attr, None)
    if getter is None:
        return []
    return [(idx, None) for idx in getter(probe, k)]


def _backend_of(vectordb):
    """The semantic backend, whether handed a VectorDB wrapper or a backend."""
    return getattr(vectordb, "backend", vectordb)


def search(
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
    abstention_enabled: bool = True,
    lexical_evidence_floor: float = DEFAULT_LEXICAL_EVIDENCE_FLOOR,
    semantic_confidence_floor: float = DEFAULT_SEMANTIC_CONFIDENCE_FLOOR,
) -> RetrievalOutcome:
    """
    Hybrid retrieval returning scored candidates, and a reason when it abstains.

    Ordering is still Reciprocal Rank Fusion, but each candidate now carries the
    scores that produced it, so a threshold has something real to threshold on.

    Abstention uses a two-tier gate, in loci's shape: a question is answered if
    it is **lexically grounded** — its words exist in the corpus, weighted by
    IDF — **or** **semantically confident** — some chunk is close enough in
    embedding space. Requiring both would reject exactly the questions
    embeddings were added for; requiring neither is how the pipeline came to
    answer gibberish with confident documentation.
    """
    if search_mode is None:
        return RetrievalOutcome(abstained=True, reason="search_disabled")

    if candidate_k is None:
        candidate_k = top_k * 3

    expanded_query = query
    if query_expansion_enabled and query_expansion_type != "none":
        expanded_query = _expand_query(query, query_expansion_type, vectordb, bm25_index)

    ranked_lists: List[List[int]] = []
    base_texts: List[str] = []
    cosine_by_index: Dict[int, float] = {}
    keyword_by_index: Dict[int, float] = {}

    # --- FAISS semantic search ---
    if search_mode in ("hybrid", "faiss", "hybrid_web") and vectordb is not None:
        hits = _scored(_backend_of(vectordb), query_embedding, candidate_k, "search_indices")
        if hits:
            ranked_lists.append([idx for idx, _ in hits])
            base_texts = vectordb.texts
            cosine_by_index = {idx: score for idx, score in hits if score is not None}

    # --- BM25 keyword search ---
    if search_mode in ("hybrid", "bm25") and bm25_index is not None:
        hits = _scored(bm25_index, expanded_query, candidate_k, "search_indices")
        if hits:
            ranked_lists.append([idx for idx, _ in hits])
            if not base_texts:
                base_texts = bm25_index.texts
            keyword_by_index = {idx: score for idx, score in hits if score is not None}

    # --- Web results occupy index space after the indexed chunks ---
    web_texts: List[str] = []
    if web_results:
        web_texts = [result.to_chunk_text() for result in web_results]
        offset = len(base_texts)
        ranked_lists.append(list(range(offset, offset + len(web_texts))))

    if not ranked_lists or (not base_texts and not web_texts):
        return RetrievalOutcome(abstained=True, reason="no_index")

    if len(ranked_lists) == 1:
        fused = [(idx, 1.0 / (rrf_k + rank)) for rank, idx in enumerate(ranked_lists[0], start=1)]
    else:
        fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)

    # --- Two-tier gate, on the question as a whole ---
    #
    # Only judge with signals actually available. A backend that reports no
    # scores tells us nothing about relevance, and abstaining on the absence of
    # a measurement — rather than on a measurement of absence — would silence
    # every index that has not implemented scoring.
    has_lexical_signal = bm25_index is not None and hasattr(bm25_index, "term_evidence")
    has_semantic_signal = bool(cosine_by_index)

    evidence = (
        sum(bm25_index.term_evidence(query).values()) if has_lexical_signal else 0.0
    )
    best_cosine = max(cosine_by_index.values(), default=0.0)

    can_judge = has_lexical_signal or has_semantic_signal
    if abstention_enabled and can_judge and not web_texts:
        grounded = has_lexical_signal and evidence >= lexical_evidence_floor
        confident = has_semantic_signal and best_cosine >= semantic_confidence_floor
        if not grounded and not confident:
            return RetrievalOutcome(
                abstained=True,
                reason="no_lexical_or_semantic_evidence",
                evidence=evidence,
                best_cosine=best_cosine,
            )

    total = len(base_texts) + len(web_texts)
    seen: set = set()
    candidates: List[Candidate] = []
    for idx, score in fused:
        if idx in seen or idx >= total:
            continue
        if score < relevance_threshold:
            break
        seen.add(idx)
        is_web = idx >= len(base_texts)
        sources = []
        if idx in cosine_by_index:
            sources.append("faiss")
        if idx in keyword_by_index:
            sources.append("bm25")
        if is_web:
            sources.append("web")
        candidates.append(
            Candidate(
                index=idx,
                text=web_texts[idx - len(base_texts)] if is_web else base_texts[idx],
                rrf_score=score,
                cosine=cosine_by_index.get(idx),
                keyword_score=keyword_by_index.get(idx),
                sources=tuple(sources),
            )
        )
        if len(candidates) >= top_k:
            break

    return RetrievalOutcome(
        candidates=candidates,
        abstained=not candidates,
        reason="" if candidates else "no_candidates",
        evidence=evidence,
        best_cosine=best_cosine,
    )


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
    Text-only view of :func:`search`, kept for callers that want chunks alone.

    Abstention is off here: this signature cannot express *why* nothing came
    back, and a caller that cannot tell "no match" from "no index" should not
    be silently given an empty list.
    """
    return search(
        query=query,
        query_embedding=query_embedding,
        vectordb=vectordb,
        bm25_index=bm25_index,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        relevance_threshold=relevance_threshold,
        search_mode=search_mode,
        web_results=web_results,
        query_expansion_enabled=query_expansion_enabled,
        query_expansion_type=query_expansion_type,
        abstention_enabled=False,
    ).texts
