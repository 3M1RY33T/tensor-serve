from src.query_analyzer import QueryAnalyzer


def test_needs_rag_skips_simple_queries():
    needs_rag, reason = QueryAnalyzer.needs_rag("what is 2+2?")

    assert needs_rag is False
    assert reason == "matches_simple_pattern"


def test_needs_rag_uses_context_for_domain_queries():
    needs_rag, reason = QueryAnalyzer.needs_rag("How do I optimize FastAPI performance?")

    assert needs_rag is True
    assert reason == "contains_domain_indicator"


def test_select_search_mode_for_keyword_heavy_query():
    assert QueryAnalyzer.select_search_mode("asyncio.gather") == "bm25"


def test_select_search_mode_for_conceptual_query():
    query = "Explain the architecture and design pattern behind event loops"

    assert QueryAnalyzer.select_search_mode(query) == "faiss"
