"""
Tests for the retrieval evaluation harness.

The harness is a measuring instrument, so these check that it measures what it
claims: that generated gold answers really are correct by construction, that a
perfect retriever scores 1.0 and a silent one scores 0.0, and that abstention
is reported rather than assumed.
"""

import numpy as np
import pytest

import api.config as cfg
from api.bm25_index import BM25Index
from api.evaluation import (
    FAMILIES,
    build_questions,
    render,
    run_eval,
    save_baseline,
)
from api.lexical import tokenize

TEXTS = [
    "backpressure in asyncio streams uses drain and pause_reading to slow a fast producer",
    "the drain coroutine blocks until the transport write buffer drops below the low water mark",
    "raised garden beds improve drainage in heavy clay soil and warm earlier in spring",
    "companion planting pairs marigolds with tomatoes to deter nematodes in the bed",
    "reciprocal rank fusion sums one over sixty plus rank across every ranked list",
    "fusion dampens the influence of any single retriever by scoring purely on rank position",
]
META = [
    {"article_title": "asyncio backpressure"},
    {"article_title": "asyncio backpressure"},
    {"article_title": "garden beds"},
    {"article_title": "garden beds"},
    {"article_title": "rank fusion"},
    {"article_title": "rank fusion"},
]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")
    return tmp_path


class StubEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


class OracleDB:
    """A vector index that returns whatever the test tells it to."""

    def __init__(self, texts, meta, order=None):
        self._texts = list(texts)
        self.metadata = list(meta)
        self.order = order

    @property
    def texts(self):
        return self._texts

    def search_indices(self, query_embedding, top_k):
        order = self.order if self.order is not None else range(len(self._texts))
        return list(order)[:top_k]


class SilentDB(OracleDB):
    def search_indices(self, query_embedding, top_k):
        return []


# ------------------------------------------------------------ question quality


def test_generates_every_family_when_metadata_is_present():
    questions = build_questions(TEXTS, META, sample=6, seed=0)
    produced = {q.family for q in questions}
    assert produced == set(FAMILIES), f"missing families: {set(FAMILIES) - produced}"


def test_passage_gold_is_the_chunk_the_span_came_from():
    """The defining property: gold is known before retrieval runs."""
    for question in build_questions(TEXTS, META, sample=6, seed=0):
        if question.family != "passage":
            continue
        assert len(question.gold) == 1
        source = TEXTS[next(iter(question.gold))]
        assert question.query in source


def test_signature_and_title_gold_point_at_their_own_article():
    by_title = {}
    for idx, meta in enumerate(META):
        by_title.setdefault(meta["article_title"], set()).add(idx)

    for question in build_questions(TEXTS, META, sample=6, seed=0):
        if question.family in ("title", "signature"):
            assert question.gold == by_title[question.note]


def test_nonsense_questions_use_no_corpus_vocabulary():
    """If a nonsense probe shares a word with the corpus it is not unanswerable."""
    vocabulary = set()
    for text in TEXTS:
        vocabulary.update(tokenize(text))

    probes = [q for q in build_questions(TEXTS, META, sample=6, seed=0) if q.family == "nonsense"]
    assert probes
    for question in probes:
        assert not (set(tokenize(question.query)) & vocabulary)
        assert question.gold == set()


def test_question_generation_is_reproducible():
    a = build_questions(TEXTS, META, sample=6, seed=7)
    b = build_questions(TEXTS, META, sample=6, seed=7)
    assert [(q.family, q.query) for q in a] == [(q.family, q.query) for q in b]


def test_works_without_article_metadata():
    questions = build_questions(TEXTS, [{} for _ in TEXTS], sample=6, seed=0)
    produced = {q.family for q in questions}
    assert "passage" in produced and "nonsense" in produced
    assert "title" not in produced, "title questions need article metadata"


# ------------------------------------------------------------------- scoring


def test_a_silent_retriever_scores_zero_and_abstains_fully(isolated_config):
    bm25 = BM25Index()
    bm25.build(list(TEXTS))
    report = run_eval(
        db=SilentDB(TEXTS, META),
        bm25=None,
        embedder=StubEmbedder(),
        top_k=3,
        sample=6,
        seed=0,
    )
    assert report["families"]["passage"]["recall_at_k"] == 0.0
    assert report["families"]["nonsense"]["abstention"] == 1.0


def test_a_perfect_retriever_scores_one(isolated_config):
    """Rank the gold chunk first for every question and the metrics must agree."""
    questions = build_questions(TEXTS, META, sample=6, seed=0)
    passages = [q for q in questions if q.family == "passage"]

    for question in passages:
        gold = next(iter(question.gold))
        report = run_eval(
            db=OracleDB(TEXTS, META, order=[gold]),
            bm25=None,
            embedder=StubEmbedder(),
            top_k=3,
            questions=[question],
        )
        stats = report["families"]["passage"]
        assert stats["recall_at_k"] == 1.0
        assert stats["precision_at_1"] == 1.0
        assert stats["mrr"] == 1.0


def test_mrr_reflects_the_rank_of_the_first_hit(isolated_config):
    question = next(
        q for q in build_questions(TEXTS, META, sample=6, seed=0) if q.family == "passage"
    )
    gold = next(iter(question.gold))
    distractors = [i for i in range(len(TEXTS)) if i != gold]

    report = run_eval(
        db=OracleDB(TEXTS, META, order=[distractors[0], distractors[1], gold]),
        bm25=None,
        embedder=StubEmbedder(),
        top_k=3,
        questions=[question],
    )
    # the report rounds to 4 decimals
    assert report["families"]["passage"]["mrr"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["families"]["passage"]["precision_at_1"] == 0.0


def test_report_carries_the_settings_it_measured(isolated_config):
    report = run_eval(
        db=OracleDB(TEXTS, META), bm25=None, embedder=StubEmbedder(), top_k=3, sample=6
    )
    assert "relevance_threshold" in report["settings"]
    assert report["corpus_chunks"] == len(TEXTS)
    assert "cold_start" in report["latency_ms"]


def test_empty_corpus_is_refused(isolated_config):
    with pytest.raises(ValueError):
        run_eval(db=OracleDB([], []), bm25=None, embedder=StubEmbedder())


def test_baseline_round_trips(tmp_path, isolated_config):
    import json

    report = run_eval(
        db=OracleDB(TEXTS, META), bm25=None, embedder=StubEmbedder(), top_k=3, sample=6
    )
    path = save_baseline(report, str(tmp_path / "baseline.json"))
    assert json.load(open(path))["corpus_chunks"] == len(TEXTS)
    assert isinstance(render(report), str)
