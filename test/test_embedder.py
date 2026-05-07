import pytest

import api.embedder as embedder_module
from api.embedder import Embedder


def test_embedder_prefers_local_model_cache(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def encode(self, texts, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    embedder = Embedder()

    assert calls == [("all-MiniLM-L6-v2", {"local_files_only": True})]
    assert embedder.encode(["hello"]) == [[1.0]]


def test_embedder_falls_back_to_download_when_cache_is_missing(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))
            if kwargs.get("local_files_only"):
                raise OSError("missing local model")

        def encode(self, texts, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    Embedder()

    assert calls == [
        ("all-MiniLM-L6-v2", {"local_files_only": True}),
        ("all-MiniLM-L6-v2", {}),
    ]


def test_embedder_raises_clear_error_when_model_cannot_load(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            raise OSError("no model")

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    with pytest.raises(RuntimeError, match="Could not load embedding model"):
        Embedder()
