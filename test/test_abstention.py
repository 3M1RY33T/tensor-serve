"""
Phase 3: scores survive fusion, and the pipeline can say it does not know.

Before this, retrieval returned bare chunk text. No score reached the caller,
so no threshold was meaningful and abstention was impossible — the proxy
answered four invented words with confident documentation.
"""

import numpy as np
import pytest

import api.config as cfg
from api.bm25_index import BM25Index
from api.hybrid_search import (
    DEFAULT_LEXICAL_EVIDENCE_FLOOR,
    DEFAULT_SEMANTIC_CONFIDENCE_FLOOR,
    Candidate,
    hybrid_search,
    search,
)
from api.vectordb import VectorDB

CORPUS = [
    "asyncio.gather runs awaitables concurrently and returns their results in order",
    "the drain coroutine blocks until the transport write buffer falls below the low water mark",
    "raised garden beds improve drainage in heavy clay soil and warm earlier in spring",
    "reciprocal rank fusion sums one over sixty plus rank across every ranked list",
]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")
    return tmp_path


@pytest.fixture
def index():
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), list(CORPUS))
    bm25 = BM25Index()
    bm25.build(list(CORPUS))
    return db, bm25


class NoScoreBM25:
    """A keyword index that reports no scores and no evidence."""

    texts = CORPUS

    def search_indices(self, query, top_k):
        return [0, 1]


# ----------------------------------------------------------------- scores kept


def test_candidates_carry_their_index_and_scores(index):
    db, bm25 = index
    outcome = search(
        query="asyncio gather awaitables",
        query_embedding=np.eye(4, dtype="float32")[0],
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert outcome.candidates
    for candidate in outcome.candidates:
        assert isinstance(candidate, Candidate)
        assert 0 <= candidate.index < len(CORPUS)
        assert candidate.text == CORPUS[candidate.index]
        assert candidate.sources


def test_cosine_is_a_real_similarity_not_a_rank(index):
    """cos = 1 - d/2 over unit vectors, so an exact match scores 1.0."""
    db, _ = index
    scored = db.backend.search_scored(np.eye(4, dtype="float32")[0], 4)
    assert scored[0] == (0, pytest.approx(1.0))
    assert all(-1.01 <= s <= 1.01 for _, s in scored)


def test_term_evidence_is_zero_for_words_absent_from_the_corpus(index):
    _, bm25 = index
    assert bm25.term_evidence("asyncio gather")
    assert bm25.term_evidence("zorpquix vempthal glirnwub") == {}


# -------------------------------------------------------------------- the gate


def test_abstains_on_words_absent_from_the_corpus(index):
    db, bm25 = index
    outcome = search(
        query="zorpquix vempthal glirnwub kreshplom",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert outcome.abstained
    assert outcome.candidates == []
    assert outcome.reason == "no_lexical_or_semantic_evidence"
    assert outcome.evidence == 0.0


def test_answers_a_question_whose_words_are_in_the_corpus(index):
    db, bm25 = index
    outcome = search(
        query="asyncio gather awaitables",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert not outcome.abstained
    assert outcome.candidates
    assert outcome.evidence >= DEFAULT_LEXICAL_EVIDENCE_FLOOR


def test_semantic_confidence_rescues_a_question_with_no_shared_words(index):
    """
    A lexical gate alone would reject exactly what embeddings were added for,
    so high cosine must be sufficient on its own.
    """
    db, bm25 = index
    outcome = search(
        query="zorpquix vempthal glirnwub",
        query_embedding=np.eye(4, dtype="float32")[0],  # cosine 1.0 with chunk 0
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert not outcome.abstained
    assert outcome.best_cosine >= DEFAULT_SEMANTIC_CONFIDENCE_FLOOR


def test_gate_is_skipped_when_no_signal_exists_to_judge_with(index):
    """
    Abstaining on the absence of a measurement, rather than on a measurement of
    absence, would silence any backend that does not implement scoring.
    """
    db, _ = index
    outcome = search(
        query="zorpquix vempthal glirnwub",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=None,
        bm25_index=NoScoreBM25(),
        top_k=3,
        search_mode="bm25",
    )
    assert not outcome.abstained
    assert outcome.candidates


def test_abstention_can_be_switched_off(index):
    db, bm25 = index
    outcome = search(
        query="zorpquix vempthal glirnwub kreshplom",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
        abstention_enabled=False,
    )
    assert not outcome.abstained
    assert outcome.candidates


def test_floors_are_configurable(index):
    db, bm25 = index
    outcome = search(
        query="asyncio gather awaitables",
        query_embedding=np.eye(4, dtype="float32")[0],
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
        lexical_evidence_floor=10_000.0,
        semantic_confidence_floor=2.0,
    )
    assert outcome.abstained, "an unreachable floor should abstain"


def test_web_results_bypass_the_gate(index):
    """Web search answers questions the local corpus cannot; do not gate it."""

    class W:
        def to_chunk_text(self):
            return "a fresh web result"

    db, bm25 = index
    outcome = search(
        query="zorpquix vempthal glirnwub kreshplom",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=db,
        bm25_index=bm25,
        top_k=10,
        web_results=[W()],
    )
    assert not outcome.abstained
    assert outcome.reason != "no_lexical_or_semantic_evidence"
    assert any(c.is_web for c in outcome.candidates)


# ------------------------------------------------------------- back-compatible


def test_text_only_wrapper_still_returns_strings(index):
    db, bm25 = index
    results = hybrid_search(
        query="asyncio gather",
        query_embedding=np.eye(4, dtype="float32")[0],
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert results and all(isinstance(r, str) for r in results)


def test_text_only_wrapper_does_not_abstain(index):
    """The plain signature cannot say why it is empty, so it must not gate."""
    db, bm25 = index
    results = hybrid_search(
        query="zorpquix vempthal glirnwub kreshplom",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert results


def test_cosine_ignores_query_magnitude(index):
    """
    cos = 1 - d/2 assumes a unit query. An unnormalised one silently skews the
    similarity, and a zero vector scored 0.5 — inside the band that passes the
    semantic gate.
    """
    db, _ = index
    unit = np.eye(4, dtype="float32")[0]

    assert db.backend.search_scored(unit, 2) == db.backend.search_scored(unit * 7, 2)
    assert db.backend.search_scored(np.zeros(4, dtype="float32"), 2) == []


def test_zero_query_vector_does_not_pass_the_semantic_gate(index):
    db, bm25 = index
    outcome = search(
        query="zorpquix vempthal glirnwub kreshplom",
        query_embedding=np.zeros(4, dtype="float32"),
        vectordb=db,
        bm25_index=bm25,
        top_k=3,
    )
    assert outcome.abstained
    assert outcome.best_cosine == 0.0
