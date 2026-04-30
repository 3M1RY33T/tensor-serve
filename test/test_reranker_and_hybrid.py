from hybrid_search import hybrid_search, reciprocal_rank_fusion
from reranker import get_reranker, rerank_results


class FakeVDB:
    def __init__(self, texts, indices):
        self.texts = texts
        self._indices = indices

    def search_indices(self, query_embedding, top_k):
        return self._indices


class FakeBM25:
    def __init__(self, texts, indices):
        self.texts = texts
        self._indices = indices

    def search_indices(self, query, top_k):
        return self._indices


def test_rerank_disabled_returns_original():
    docs = ["a", "b"]

    assert rerank_results("q", docs, reranker_enabled=False) == docs


def test_rerank_no_model_returns_original_or_list():
    r = get_reranker(enabled=True)
    if r is None or not r.is_available():
        assert rerank_results("q", ["a", "b"], reranker_enabled=True) == ["a", "b"]
    else:
        assert isinstance(rerank_results("q", ["a", "b"], reranker_enabled=True), list)


def test_rrf_and_hybrid_combination():
    ranked_lists = [[2, 1, 0], [1, 3, 2]]
    fused = reciprocal_rank_fusion(ranked_lists, k=60)

    assert isinstance(fused, list)
    assert all(isinstance(t, tuple) for t in fused)

    texts = ["t0", "t1", "t2", "t3"]
    v = FakeVDB(texts, [2, 0])
    b = FakeBM25(texts, [1, 2])
    res = hybrid_search(
        "q",
        [0.1, 0.1],
        v,
        b,
        top_k=3,
        candidate_k=3,
        rrf_k=60,
        relevance_threshold=0.0,
    )

    assert all(r in texts for r in res)
