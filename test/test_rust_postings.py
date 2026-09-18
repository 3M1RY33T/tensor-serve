"""
The optional Rust accelerator must produce exactly the index Python produces.

It is an accelerator, not an alternative implementation: if the two ever
disagree, retrieval quality depends on whether a compiled extension happened to
be installed, which is the worst kind of bug to chase.
"""

import random

import numpy as np
import pytest

import api.bm25 as bm25_module
from api.bm25 import PostingsBM25, accelerated
from api.lexical import tokenize

CASES = [
    "asyncio.gather, and asyncio.gather.",
    "Use the ms-marco-MiniLM model",
    "POST /v1/chat/completions",
    "snake_case_name",
    "café — ünïcodé ✓ 42",
    "",
    "   ",
    "a.b.c.d",
    "x1_y2-z3/w4",
    "Mixed CASE Words",
    "trailing- dash",
    "1234 5678",
]

needs_rust = pytest.mark.skipif(not accelerated(), reason="Rust extension not built")


@pytest.fixture
def corpus():
    rng = random.Random(0)
    vocab = [f"term{i}" for i in range(200)] + ["asyncio.gather", "ms-marco", "api/v1"]
    return [" ".join(rng.choice(vocab) for _ in range(60)) for _ in range(300)]


@needs_rust
@pytest.mark.parametrize("text", CASES)
def test_rust_tokeniser_matches_python(text):
    import tensor_postings

    assert tensor_postings.tokenize(text) == tokenize(text)


@needs_rust
def test_rust_tokeniser_matches_python_on_a_corpus(corpus):
    import tensor_postings

    for text in corpus:
        assert tensor_postings.tokenize(text) == tokenize(text)


def _build_with(texts, use_rust, monkeypatch):
    if not use_rust:
        monkeypatch.setattr(bm25_module, "_rust", None)
    index = PostingsBM25()
    index.build(list(texts))
    return index


@needs_rust
def test_both_builders_produce_the_same_index(corpus, monkeypatch):
    rust = _build_with(corpus, True, monkeypatch)

    with pytest.MonkeyPatch.context() as mp:
        python = _build_with(corpus, False, mp)

    assert rust.vocabulary_size == python.vocabulary_size
    assert np.allclose(rust._doc_lengths, python._doc_lengths)
    assert rust._avgdl == pytest.approx(python._avgdl)
    assert set(rust._idf) == set(python._idf)
    for term in python._idf:
        assert rust._idf[term] == pytest.approx(python._idf[term], abs=1e-9)


@needs_rust
def test_both_builders_score_queries_identically(corpus, monkeypatch):
    rust = _build_with(corpus, True, monkeypatch)
    with pytest.MonkeyPatch.context() as mp:
        python = _build_with(corpus, False, mp)

    for query in ("term5 term9", "asyncio.gather", "ms-marco api/v1", "term1"):
        a, b = rust.scores(query), python.scores(query)
        if a is None or b is None:
            assert a is None and b is None, query
            continue
        assert np.allclose(a, b, atol=1e-5), query
        assert rust.search_indices(query, 10) == python.search_indices(query, 10), query


@needs_rust
def test_unindexed_query_is_none_in_both(corpus, monkeypatch):
    rust = _build_with(corpus, True, monkeypatch)
    with pytest.MonkeyPatch.context() as mp:
        python = _build_with(corpus, False, mp)
    assert rust.scores("zorpquix vempthal") is None
    assert python.scores("zorpquix vempthal") is None


def test_the_python_path_works_without_the_extension(corpus, monkeypatch):
    """The extension is optional; absence must change nothing but speed."""
    monkeypatch.setattr(bm25_module, "_rust", None)
    index = PostingsBM25()
    index.build(list(corpus))
    assert index.vocabulary_size > 0
    assert index.search_indices("term5", 3)


def test_empty_corpus_is_safe_in_both(monkeypatch):
    index = PostingsBM25()
    index.build([])
    assert index.scores("anything") is None

    monkeypatch.setattr(bm25_module, "_rust", None)
    index = PostingsBM25()
    index.build([])
    assert index.scores("anything") is None
