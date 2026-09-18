"""
Regressions for the Phase 1 retrieval fixes.

Every test here fails against the code as it stood at 437b031. The suite was
green at that commit because the one test exercising hybrid_search passed
relevance_threshold=0.0 — a value api/main.py could never produce, since
`0.0 or 0.05` evaluates to 0.05. These tests exercise the shipped defaults.
"""

import numpy as np
import pytest

import api.config as cfg
import api.main as main
from api.bm25_index import BM25Index
from api.chunker import chunk_text
from api.config import DEFAULT_CONFIG
from api.hybrid_search import hybrid_search, max_rrf_score
from api.lexical import tokenize, top_k_indices
from api.query_expansion import get_expander
from api.vectordb import VectorDB, index_exists


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point config at a scratch file so tests never touch the real one."""
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")
    return tmp_path


CORPUS = [
    "backpressure in asyncio streams uses drain and pause_reading semantics",
    "gardening tools for clay soil and raised beds",
    "asyncio.gather runs awaitables concurrently and returns their results",
    "the reciprocal rank fusion constant dampens the impact of top ranks",
]


class StubEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


class StubVectorDB:
    """A vector index that ranks by position, so results are deterministic."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.metadata = [{"article_title": f"a{i}"} for i in range(len(texts))]

    @property
    def texts(self):
        return self._texts

    def search_indices(self, query_embedding, top_k):
        return list(range(min(top_k, len(self._texts))))


class StubWebResult:
    def __init__(self, text):
        self._text = text

    def to_chunk_text(self):
        return self._text


# --------------------------------------------------------------------------
# Finding 01 — the default threshold emptied every result set
# --------------------------------------------------------------------------


def test_default_relevance_threshold_is_attainable():
    """A threshold no RRF score can reach silently disables retrieval."""
    assert DEFAULT_CONFIG["relevance_threshold"] < max_rrf_score(1)


def test_search_returns_results_under_default_config(isolated_config, monkeypatch):
    """End to end with the shipped defaults — this returned [] at 437b031."""
    monkeypatch.setattr(main.app_state, "embedder", StubEmbedder())
    monkeypatch.setattr(main.app_state, "db", StubVectorDB(CORPUS))
    bm25 = BM25Index()
    bm25.build(CORPUS)
    monkeypatch.setattr(main.app_state, "bm25", bm25)
    main.query_cache.clear()

    results, mode = main._retrieve("asyncio backpressure drain", top_k=3)

    assert results, f"default config returned no chunks (mode={mode})"
    assert all(chunk in CORPUS for chunk in results)


def test_configured_zero_threshold_survives(isolated_config):
    """`or` fallbacks turned a deliberate 0.0 back into the broken default."""
    cfg.set_config_value("relevance_threshold", 0.0)
    assert main._retrieval_settings()["relevance_threshold"] == 0.0


def test_configured_false_reranker_survives(isolated_config):
    """Same `or` bug class, for the boolean settings."""
    cfg.set_config_value("reranker_enabled", False)
    assert main._retrieval_settings()["reranker_enabled"] is False


def test_stored_broken_threshold_is_migrated(isolated_config):
    """Existing installs carry 0.05 in config.json; loading must repair it."""
    cfg.save_config({**DEFAULT_CONFIG, "relevance_threshold": 0.05})
    assert cfg.load_config()["relevance_threshold"] < max_rrf_score(1)


# --------------------------------------------------------------------------
# Finding 02 — chunks were truncated before reaching their own vector
# --------------------------------------------------------------------------


class StubTokenizer:
    """Whitespace tokenizer exposing the offset mapping chunking needs."""

    def __call__(self, text, **kwargs):
        offsets, pos = [], 0
        for word in text.split(" "):
            if word:
                offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        return {"offset_mapping": offsets}


def test_token_aware_chunks_respect_the_model_limit():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, tokenizer=StubTokenizer(), max_tokens=50, overlap=10)

    assert chunks
    assert all(len(c.split()) <= 50 for c in chunks), "a chunk exceeds the token budget"


def test_token_aware_chunks_are_verbatim_slices():
    text = " ".join(f"w{i}" for i in range(200))
    for chunk in chunk_text(text, tokenizer=StubTokenizer(), max_tokens=30, overlap=5):
        assert chunk in text


def test_word_chunking_still_works_without_a_tokenizer():
    chunks = chunk_text(" ".join(str(i) for i in range(30)), chunk_size=10, overlap=2)
    assert len(chunks) == 4

def test_chunks_fit_the_real_tokenizer_not_just_the_offset_count():
    """
    Packing by offset count is close but not exact: re-tokenising a character
    slice can yield more tokens than the span had in place, because a token at
    the cut splits differently once its neighbours are gone. Ingesting real
    articles produced a 257-token chunk against a 256-token model this way.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )
    except Exception:
        pytest.skip("all-MiniLM-L6-v2 tokenizer not cached locally")

    limit, budget = 256, 254
    text = (
        "The asyncio.gather() coroutine runs awaitables concurrently; "
        "see also loop.run_in_executor(), asyncio.wait_for() and "
        "concurrent.futures.ThreadPoolExecutor for related APIs. "
    ) * 60

    chunks = chunk_text(text, overlap=30, tokenizer=tok, max_tokens=budget)

    assert chunks
    worst = max(len(tok.encode(c, add_special_tokens=True)) for c in chunks)
    assert worst <= limit, f"a chunk reached {worst} tokens against a {limit}-token model"



# --------------------------------------------------------------------------
# Finding 06 — web results were appended to the live index
# --------------------------------------------------------------------------


def test_web_results_do_not_mutate_the_vector_index():
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), list(CORPUS))
    bm25 = BM25Index()
    bm25.build(db.texts)
    before = len(db.texts)

    for _ in range(3):
        hybrid_search(
            query="asyncio",
            query_embedding=np.eye(4, dtype="float32")[0],
            vectordb=db,
            bm25_index=bm25,
            top_k=3,
            search_mode="hybrid",
            web_results=[StubWebResult("web a"), StubWebResult("web b")],
        )

    assert len(db.texts) == before, "web results leaked into the vector index"
    assert len(db.texts) == db.backend.index.ntotal
    assert len(db.metadata) == len(db.texts)


def test_web_results_are_still_retrievable():
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32"), list(CORPUS))
    results = hybrid_search(
        query="nothing matches this query at all",
        query_embedding=np.eye(4, dtype="float32")[0],
        vectordb=db,
        bm25_index=None,
        top_k=10,
        search_mode="hybrid",
        web_results=[StubWebResult("a web chunk")],
    )
    assert "a web chunk" in results


# --------------------------------------------------------------------------
# Finding 07 — pseudo-relevance feedback never expanded anything
# --------------------------------------------------------------------------


def test_prf_expands_when_given_a_first_pass_result():
    expanded = get_expander("prf").expand("asyncio", top_result=CORPUS[2])
    assert expanded != "asyncio"
    assert "gather" in expanded


def test_prf_runs_its_own_first_pass_through_hybrid_search():
    """hybrid_search must supply top_result; PRF is inert without it."""
    bm25 = BM25Index()
    bm25.build(CORPUS)
    calls = []

    # The feedback pass fetches indices; the main pass asks for scores.
    for method in ("search_indices", "search_scored"):
        original = getattr(bm25, method)

        def spy(query, top_k, _original=original, _name=method):
            calls.append((_name, query, top_k))
            return _original(query, top_k)

        setattr(bm25, method, spy)

    hybrid_search(
        query="asyncio",
        query_embedding=None,
        vectordb=None,
        bm25_index=bm25,
        top_k=3,
        search_mode="bm25",
        query_expansion_enabled=True,
        query_expansion_type="prf",
    )

    assert len(calls) == 2, f"PRF did not run a feedback pass: {calls}"
    feedback, main_pass = calls
    assert feedback[2] == 1, "the feedback pass should fetch the top-1 result"
    assert main_pass[1] != "asyncio", "the second pass used the unexpanded query"


# --------------------------------------------------------------------------
# Finding 14 — punctuation forked identifiers into distinct terms
# --------------------------------------------------------------------------


def test_trailing_punctuation_does_not_change_a_term():
    assert tokenize("asyncio.gather,")[0] == "asyncio.gather"
    assert tokenize("asyncio.gather.")[0] == tokenize("asyncio.gather")[0]


def test_compound_identifiers_are_findable_by_their_parts():
    assert "gather" in tokenize("asyncio.gather")


def test_bm25_finds_a_punctuated_identifier():
    bm25 = BM25Index()
    bm25.build(CORPUS)
    hits = bm25.search_indices("asyncio.gather,", 1)
    assert hits and CORPUS[hits[0]] == CORPUS[2]


def test_top_k_indices_matches_a_full_sort():
    scores = np.array([0.1, 5.0, 3.0, 9.0, 2.0])
    assert top_k_indices(scores, 3) == [3, 1, 2]
    assert top_k_indices(scores, 99) == [3, 1, 2, 4, 0]
    assert top_k_indices(np.array([]), 3) == []


# --------------------------------------------------------------------------
# Findings 15 / 16 — auto-load probed the wrong files, backend config was inert
# --------------------------------------------------------------------------


def test_index_exists_matches_what_the_backend_writes(tmp_path, isolated_config):
    path = str(tmp_path / "zim_db")
    assert not index_exists(path, "faiss_flat")

    db = VectorDB(dim=4, variant="faiss_flat")
    db.add(np.eye(4, dtype="float32"), list(CORPUS))
    db.save(path)

    assert index_exists(path, "faiss_flat"), "auto-load would not find a saved index"


def test_vectordb_honours_the_configured_semantic_backend(isolated_config, monkeypatch):
    """
    Every call site hardcoded VectorDB(dim=384), so semantic_backend was inert
    and faiss_ivf was unreachable however the profile was set. Constructing the
    backend does not train it, so this is safe regardless of the OpenMP probe.
    """
    import api.vectordb as vectordb

    monkeypatch.setattr(vectordb, "_ivf_is_usable", lambda: True)

    cfg.set_config_value("semantic_backend", "faiss_ivf")
    assert VectorDB(dim=4).variant == "faiss_ivf"

    cfg.set_config_value("semantic_backend", "faiss_flat")
    assert VectorDB(dim=4).variant == "faiss_flat"


# --------------------------------------------------------------------------
# Finding 11 — IVF trained on the first batch, with nprobe left at 1
# --------------------------------------------------------------------------


def test_ivf_sizes_clusters_to_the_corpus_and_sets_nprobe():
    from api.search_backends.faiss_ivf import FAISSIVFBackend
    from api.search_backends.ivf_support import ivf_training_is_safe

    if not ivf_training_is_safe():
        pytest.skip("FAISS IVF training aborts alongside torch in this environment")

    backend = FAISSIVFBackend(dim=8, nprobe=4)
    rng = np.random.default_rng(0)
    vectors = rng.random((300, 8), dtype="float32")
    for start in range(0, 300, 100):
        backend.add(vectors[start:start + 100], [f"t{i}" for i in range(start, start + 100)])

    hits = backend.search_indices(vectors[0], 5)

    assert hits, "IVF returned nothing"
    assert backend.n_clusters < 316, "cluster count still hardcoded for 100k vectors"
    assert backend.index.nprobe == 4, "nprobe left at the FAISS default of 1"


def test_unusable_ivf_falls_back_instead_of_killing_the_process(isolated_config):
    """
    faiss and torch can ship conflicting OpenMP runtimes; training then aborts
    the process. Selecting faiss_ivf must degrade to exact search, not crash.
    """
    from api.search_backends.ivf_support import ivf_training_is_safe

    cfg.set_config_value("semantic_backend", "faiss_ivf")
    db = VectorDB(dim=4)

    if ivf_training_is_safe():
        assert db.variant == "faiss_ivf"
    else:
        assert db.variant == "faiss_flat", "unusable IVF backend was not replaced"

    db.add(np.eye(4, dtype="float32"), list(CORPUS))
    assert db.search_indices(np.eye(4, dtype="float32")[0], 2)


# --------------------------------------------------------------------------
# Finding 08 — the chat proxy ran a weaker pipeline than /search
# --------------------------------------------------------------------------


def test_chat_proxy_and_search_share_one_pipeline(isolated_config, monkeypatch):
    monkeypatch.setattr(main.app_state, "embedder", StubEmbedder())
    monkeypatch.setattr(main.app_state, "db", StubVectorDB(CORPUS))
    monkeypatch.setattr(main.app_state, "db_loaded", True)
    monkeypatch.setattr(main.app_state, "bm25", None)
    main.query_cache.clear()

    seen = []
    original = main._retrieve

    def spy(query, top_k, *args, **kwargs):
        seen.append((query, top_k))
        return original(query, top_k, *args, **kwargs)

    monkeypatch.setattr(main, "_retrieve", spy)
    main._context_for_query("explain how backpressure works in asyncio")

    assert seen, "the chat proxy does not go through the shared retrieval path"


def test_retrieval_settings_cover_every_pipeline_knob(isolated_config):
    settings = main._retrieval_settings()
    for key in (
        "relevance_threshold",
        "reranker_enabled",
        "reranker_model",
        "query_expansion_enabled",
        "query_expansion_type",
        "max_search_candidates",
    ):
        assert key in settings
