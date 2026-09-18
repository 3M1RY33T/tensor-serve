import pytest

import api.embedder as embedder_module
from api.embedder import Embedder


def test_embedder_prefers_local_model_cache(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def encode(self, texts, batch_size=32, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    embedder = Embedder(backend="torch")

    assert calls == [("all-MiniLM-L6-v2", {"local_files_only": True})]
    assert embedder.encode(["hello"]) == [[1.0]]


def test_embedder_falls_back_to_download_when_cache_is_missing(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))
            if kwargs.get("local_files_only"):
                raise OSError("missing local model")

        def encode(self, texts, batch_size=32, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    Embedder(backend="torch")

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
        Embedder(backend="torch")


def test_encode_is_safe_under_concurrent_use():
    """
    The model and its tokenizer are shared by every request, and retrieval runs
    in a thread pool, so two chat requests that both miss the cache call encode
    at once. HuggingFace's fast tokenizer is a Rust object behind a runtime
    borrow check and raises "RuntimeError: Already borrowed" when that happens —
    measured at 4 failures in 200 encodes across 8 threads before the lock.
    """
    import threading

    import torch

    from api.embedder import Embedder

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # the serving configuration that exposes the race
    try:
        embedder = Embedder()
    except RuntimeError:
        pytest.skip("embedding model not available locally")

    failures = []
    succeeded = []

    def hammer(worker):
        for i in range(20):
            try:
                embedder.encode([f"asyncio gather variant {worker}-{i} awaited"])
                succeeded.append(1)
            except Exception as exc:  # noqa: BLE001 - the failure mode is the point
                failures.append(f"{type(exc).__name__}: {exc}")

    try:
        threads = [threading.Thread(target=hammer, args=(w,)) for w in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        torch.set_num_threads(original_threads)

    assert not failures, f"{len(failures)} of {len(succeeded) + len(failures)}: {failures[:3]}"


def test_unknown_backend_fails_loudly():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        Embedder(backend="not_a_backend")


def test_a_missing_onnx_runtime_falls_back_to_torch(monkeypatch):
    """
    The ONNX runtime is an optional extra. Selecting it without installing it
    must degrade to torch with a warning, not stop the server from starting.
    """
    attempts = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            attempts.append(kwargs)
            if kwargs.get("backend") == "onnx":
                raise ImportError("onnxruntime is not installed")

        def encode(self, texts, batch_size=32, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    with pytest.warns(RuntimeWarning, match="unavailable"):
        embedder = Embedder(backend="onnx")

    assert embedder.backend == "torch"
    assert any(a.get("backend") == "onnx" for a in attempts)


def test_auto_prefers_onnx_when_the_runtime_is_present(monkeypatch):
    monkeypatch.setattr(embedder_module, "_resolve_auto", lambda workload="query": "onnx")

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            self.kwargs = kwargs

        def encode(self, texts, batch_size=32, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)
    assert Embedder(backend="auto").backend == "onnx"


def test_bulk_workload_stays_on_torch(monkeypatch):
    """
    ONNX is 3.6x faster for single queries and 2.5x slower for long chunks in
    large batches, so ingestion must not inherit the query backend.
    """
    assert embedder_module._resolve_auto("bulk") == "torch"


def test_query_workload_prefers_onnx_when_available(monkeypatch):
    import sys

    if "onnxruntime" not in sys.modules:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime not installed")
    assert embedder_module._resolve_auto("query") == "onnx"


def test_for_ingest_selects_the_bulk_backend(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            pass

        def encode(self, texts, batch_size=32, show_progress_bar=False):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder.for_ingest()
    assert embedder.workload == "bulk"
    assert embedder.backend == "torch"
