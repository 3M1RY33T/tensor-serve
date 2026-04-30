from bm25_index import BM25Index


def test_bm25_search_and_get_texts():
    idx = BM25Index()
    texts = ["hello world", "foo bar", "world of foo"]
    idx.build(texts)

    res = idx.search_indices("world", top_k=2)

    assert isinstance(res, list)
    assert all(isinstance(i, int) for i in res)
    assert all(isinstance(t, str) for t in idx.get_texts(res))
