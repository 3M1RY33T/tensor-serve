"""
Phase 4: take the linear scan out of the hot path, and stop storing the corpus
twice.

rank_bm25 scored every chunk in the collection for every query — 104.86ms at
80,000 chunks against 0.22ms for the FAISS search beside it. Scoring now walks
postings, so cost follows the query's terms rather than the corpus size.
"""

import math
import os

import numpy as np
import pytest

import api.config as cfg
import api.main as main
from api.bm25 import PostingsBM25
from api.bm25_index import BM25Index
from api.chunk_store import ChunkStore, clear_cache, load_shared
from api.lexical import tokenize
from api.vectordb import VectorDB, database_files, index_exists, list_databases

CORPUS = [
    "asyncio.gather runs awaitables concurrently and returns their results in order",
    "the drain coroutine blocks until the transport write buffer falls below the mark",
    "raised garden beds improve drainage in heavy clay soil and warm earlier in spring",
    "reciprocal rank fusion sums one over sixty plus rank across every ranked list",
    "gather gather gather repeated terms saturate under the k1 parameter",
]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")
    return tmp_path


# ------------------------------------------------------------ scoring is right


def _reference_bm25(texts, query, k1=1.5, b=0.75):
    """Textbook BM25 scored directly, as an independent check on the postings."""
    docs = [tokenize(t) for t in texts]
    lengths = [len(d) for d in docs]
    avgdl = sum(lengths) / len(lengths)
    n = len(docs)

    df = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    scores = []
    for i, doc in enumerate(docs):
        total = 0.0
        for term in tokenize(query):
            if term not in df:
                continue
            tf = doc.count(term)
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * lengths[i] / avgdl))
        scores.append(total)
    return scores


def test_postings_scores_match_a_direct_implementation():
    index = PostingsBM25()
    index.build(list(CORPUS))

    for query in ("asyncio gather", "drain transport buffer", "garden soil", "rank fusion"):
        got = index.scores(query)
        expected = _reference_bm25(CORPUS, query)
        assert got is not None
        assert np.allclose(got, expected, atol=1e-5), query


def test_ranking_prefers_the_document_about_the_query():
    index = PostingsBM25()
    index.build(list(CORPUS))
    assert CORPUS[index.search_indices("garden clay soil drainage", 1)[0]] == CORPUS[2]
    assert CORPUS[index.search_indices("reciprocal rank fusion", 1)[0]] == CORPUS[3]


def test_unindexed_query_is_none_not_an_all_zero_array():
    """
    None and zeros mean different things: the first says the question has no
    footing in this corpus, which is what the abstention gate needs to know.
    """
    index = PostingsBM25()
    index.build(list(CORPUS))
    assert index.scores("zorpquix vempthal glirnwub") is None
    assert index.scores("asyncio") is not None


def test_scoring_touches_only_documents_holding_a_query_term():
    """The property the whole phase is about."""
    index = PostingsBM25()
    index.build(list(CORPUS))
    scores = index.scores("garden")
    nonzero = [i for i, s in enumerate(scores) if s > 0]
    assert nonzero == [2]


def test_empty_corpus_is_safe():
    index = PostingsBM25()
    index.build([])
    assert index.scores("anything") is None
    assert index.search_indices("anything", 5) == []


# -------------------------------------------------------------- shared store


def test_chunk_store_is_shared_by_object_not_just_by_file(tmp_path, isolated_config):
    clear_cache()
    path = str(tmp_path / "db")

    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), CORPUS[:4])
    db.save(path)

    bm25 = BM25Index()
    bm25.build(list(db.texts))
    bm25.save(path)

    fresh_db = VectorDB(dim=4)
    fresh_db.load(path)
    fresh_bm25 = BM25Index()
    fresh_bm25.load(path)

    assert fresh_db.texts is fresh_bm25.texts, "the corpus is held twice in memory"
    assert fresh_db.texts == CORPUS[:4]


def test_keyword_index_no_longer_persists_the_corpus(tmp_path, isolated_config):
    clear_cache()
    path = str(tmp_path / "db")
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), CORPUS[:4])
    db.save(path)
    bm25 = BM25Index()
    bm25.build(list(db.texts))
    bm25.save(path)

    # The postings file should not contain the chunk text at all.
    raw = open(f"{path}.bm25", "rb").read()
    for chunk in CORPUS[:4]:
        assert chunk.encode("utf-8") not in raw, "chunk text is duplicated into the BM25 index"


def test_store_cache_notices_a_rebuild(tmp_path, isolated_config):
    clear_cache()
    path = str(tmp_path / "db")
    ChunkStore(["one"], [{}]).save(path)
    first = load_shared(path)

    os.utime(f"{path}.chunks", (0, 0))  # force a different mtime
    ChunkStore(["one", "two"], [{}, {}]).save(path)
    second = load_shared(path)

    assert second.texts == ["one", "two"], "a rebuilt store was served stale"


# ------------------------------------------------------------ db discovery


def test_discovery_reports_the_database_name_and_its_keyword_index(tmp_path, isolated_config, monkeypatch):
    clear_cache()
    monkeypatch.chdir(tmp_path)

    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), CORPUS[:4])
    db.save("python_docs")
    bm25 = BM25Index()
    bm25.build(list(db.texts))
    bm25.save("python_docs")

    found = list_databases(".")
    assert len(found) == 1
    entry = found[0]
    assert entry["name"] == "python_docs", "the suffix leaked into the database name"
    assert entry["complete"]
    assert "bm25" in entry["files"], "the keyword index was not found"
    assert index_exists("python_docs")


def test_database_files_lists_every_component():
    files = database_files("somedb", "faiss_flat")
    assert files["vectors"].endswith(".faiss_flat.index")
    assert files["chunks"].endswith(".chunks")
    assert files["bm25"].endswith(".bm25")


# --------------------------------------------------- the two endpoints agree


class StubEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


def test_search_endpoint_and_chat_proxy_return_the_same_chunks(isolated_config, monkeypatch):
    """
    The second half of the Phase 4 gate: one pipeline, so the API and the proxy
    cannot answer the same question differently.
    """
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), CORPUS[:4])
    bm25 = BM25Index()
    bm25.build(list(db.texts))

    monkeypatch.setattr(main.app_state, "embedder", StubEmbedder())
    monkeypatch.setattr(main.app_state, "db", db)
    monkeypatch.setattr(main.app_state, "bm25", bm25)
    monkeypatch.setattr(main.app_state, "db_loaded", True)
    cfg.set_config_value("context_size", 3)

    query = "explain how asyncio gather runs awaitables"

    main.query_cache.clear()
    from_search, _mode = main._retrieve(query, 3)

    main.query_cache.clear()
    from_proxy = [c.text for c in main._context_for_query(query)]

    assert from_search == from_proxy
    assert from_search


# ------------------------------------------------- keyword_backend is honoured


def test_keyword_backend_config_is_not_inert(isolated_config):
    """
    Every call site constructed BM25Index() with no argument, so the configured
    keyword_backend was never consulted and bm25_plus was unreachable — the
    same bug the semantic_backend setting had.
    """
    cfg.set_config_value("keyword_backend", "bm25_plus")
    assert BM25Index().variant == "bm25_plus"

    cfg.set_config_value("keyword_backend", "bm25_okapi")
    assert BM25Index().variant == "bm25_okapi"


def test_unknown_keyword_backend_fails_loudly(isolated_config):
    with pytest.raises(ValueError):
        BM25Index(variant="not_a_backend")


def test_bm25_plus_delta_does_not_change_the_ranking():
    """
    The lower bound is identical for every document given a query, so it shifts
    all scores equally. Both variants must rank a corpus the same way.
    """
    okapi = BM25Index(variant="bm25_okapi")
    okapi.build(list(CORPUS))
    plus = BM25Index(variant="bm25_plus")
    plus.build(list(CORPUS))

    for query in ("asyncio gather", "garden clay soil", "rank fusion"):
        assert okapi.search_indices(query, 5) == plus.search_indices(query, 5), query


def test_saved_index_records_which_variant_built_it(tmp_path, isolated_config):
    clear_cache()
    path = str(tmp_path / "db")
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), CORPUS[:4])
    db.save(path)

    built = BM25Index(variant="bm25_plus")
    built.build(list(db.texts))
    built.save(path)

    loaded = BM25Index(variant="bm25_okapi")
    loaded.load(path)
    assert loaded.variant == "bm25_plus", "the index did not record its own variant"


def test_the_chat_proxy_is_served_from_the_query_cache(isolated_config, monkeypatch):
    """
    The cache stored chunk text only, and the proxy needs candidate indices, so
    it wrote to the cache and never read from it — paying full retrieval on
    every repeat question while /search got hits.
    """
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), CORPUS[:4])
    bm25 = BM25Index()
    bm25.build(list(db.texts))

    monkeypatch.setattr(main.app_state, "embedder", StubEmbedder())
    monkeypatch.setattr(main.app_state, "db", db)
    monkeypatch.setattr(main.app_state, "bm25", bm25)
    monkeypatch.setattr(main.app_state, "db_loaded", True)
    main.query_cache.clear()

    query = "asyncio gather awaitables"

    first = main._context_for_query(query)
    assert first, "nothing retrieved"

    searches = []
    original = bm25.search_scored

    def spy(q, k):
        searches.append(q)
        return original(q, k)

    bm25.search_scored = spy
    second = main._context_for_query(query)

    assert searches == [], "the proxy re-ran retrieval instead of using the cache"
    assert [c.index for c in second] == [c.index for c in first]
    assert [c.text for c in second] == [c.text for c in first]
