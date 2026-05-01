import time

from src.cache import QueryCache


def test_cache_embedding_and_ttl():
    qc = QueryCache(max_size=2, ttl_seconds=1)
    qc.cache_embedding("a", [1.0])

    assert qc.get_embedding("a") == [1.0]
    time.sleep(1.1)
    assert qc.get_embedding("a") is None


def test_lru_eviction():
    qc = QueryCache(max_size=2, ttl_seconds=100)
    qc.cache_embedding("a", [1])
    qc.cache_embedding("b", [2])
    qc.cache_embedding("c", [3])

    assert qc.get_embedding("a") is None
    assert qc.get_embedding("b") == [2]
    assert qc.get_embedding("c") == [3]


def test_search_result_cache_uses_query_mode_and_top_k():
    qc = QueryCache(max_size=2, ttl_seconds=100)
    qc.cache_search_result("asyncio.gather", "bm25", 3, ["chunk-a"])

    assert qc.get_search_result("asyncio.gather", "bm25", 3) == ["chunk-a"]
    assert qc.get_search_result("asyncio.gather", "hybrid", 3) is None
    assert qc.get_search_result("asyncio.gather", "bm25", 5) is None
