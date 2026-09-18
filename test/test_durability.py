"""
Phase 5: an index that is being rebuilt must never be readable half-written,
and two builds must not interleave.

Measured before the change: writing a chunk store in place produced 34,704 torn
reads out of 34,709 attempts during continuous rewriting. Afterwards, 0.
"""

import json
import os
import threading
import time

import numpy as np
import pytest

import api.config as cfg
from api.bm25_index import BM25Index
from api.chunk_store import ChunkStore, clear_cache, load_shared
from api.durability import BuildLock, BuildLockError, atomic_write
from api.vectordb import VectorDB, index_exists, save_collection

CORPUS = ["alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda mu"]


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")
    return tmp_path


# ------------------------------------------------------------- atomic writes


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path):
    path = str(tmp_path / "f.bin")
    with atomic_write(path) as handle:
        handle.write(b"first")

    with pytest.raises(RuntimeError):
        with atomic_write(path) as handle:
            handle.write(b"partial")
            raise RuntimeError("interrupted")

    assert open(path, "rb").read() == b"first"


def test_a_failed_write_leaves_no_temporary_files(tmp_path):
    path = str(tmp_path / "f.bin")
    with pytest.raises(RuntimeError):
        with atomic_write(path) as handle:
            handle.write(b"partial")
            raise RuntimeError("interrupted")

    assert [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")] == []


def test_readers_never_see_a_half_written_store(tmp_path):
    """The property the phase exists for."""
    clear_cache()
    path = str(tmp_path / "db")
    texts = [f"chunk {i} " + "x" * 400 for i in range(2000)]
    ChunkStore(texts, [{} for _ in texts]).save(path)

    stop = threading.Event()
    torn, reads = [0], [0]

    def writer():
        store = ChunkStore(texts, [{} for _ in texts])
        while not stop.is_set():
            store.save(path)

    def reader():
        while not stop.is_set():
            clear_cache()
            reads[0] += 1
            try:
                if len(ChunkStore.read(path).texts) != len(texts):
                    torn[0] += 1
            except Exception:
                torn[0] += 1

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(2)
    ]
    for t in threads:
        t.start()
    time.sleep(1.5)
    stop.set()
    for t in threads:
        t.join()

    assert reads[0] > 0, "the reader never ran"
    assert torn[0] == 0, f"{torn[0]} of {reads[0]} reads saw a partial store"


# --------------------------------------------------------------- build locks


def test_a_second_build_is_refused(tmp_path):
    path = str(tmp_path / "db")
    with BuildLock(path):
        with pytest.raises(BuildLockError):
            BuildLock(path).acquire()


def test_a_second_build_is_refused_across_threads(tmp_path):
    path = str(tmp_path / "db")
    outcome = []

    def attempt():
        try:
            BuildLock(path).acquire()
            outcome.append("acquired")
        except BuildLockError:
            outcome.append("refused")

    with BuildLock(path):
        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join()

    assert outcome == ["refused"]


def test_the_lock_is_released_on_failure(tmp_path):
    path = str(tmp_path / "db")
    with pytest.raises(RuntimeError):
        with BuildLock(path):
            raise RuntimeError("ingest failed")

    with BuildLock(path):  # must not raise
        pass


def test_a_lock_left_by_a_dead_process_is_broken(tmp_path):
    """A crash must not require manual cleanup before the next build."""
    path = str(tmp_path / "db")
    lock_path = f"{path}.lock"
    with open(lock_path, "w") as handle:
        json.dump({"pid": 999_999_999, "purpose": "ingest", "started": "then"}, handle)

    with BuildLock(path):  # must not raise
        pass


# ---------------------------------------------------------- no pickles left


def test_the_chunk_store_holds_no_pickle(tmp_path):
    path = str(tmp_path / "db")
    ChunkStore(list(CORPUS), [{"a": 1} for _ in CORPUS]).save(path)
    raw = open(f"{path}.chunks", "rb").read()
    assert not raw.startswith(b"\x80"), "looks like a pickle stream"
    assert ChunkStore.read(path).texts == CORPUS


def test_the_chunk_store_rejects_a_foreign_file(tmp_path):
    path = str(tmp_path / "db")
    with open(f"{path}.chunks", "wb") as handle:
        handle.write(b"not a tensor store at all")
    with pytest.raises(ValueError, match="not a Tensor chunk store"):
        ChunkStore.read(path)


def test_unicode_survives_the_round_trip(tmp_path):
    path = str(tmp_path / "db")
    texts = ["café — ünïcodé ✓", "日本語のテキスト", "", "emoji 🎉 tail"]
    ChunkStore(texts, [{} for _ in texts]).save(path)
    assert ChunkStore.read(path).texts == texts


def test_the_keyword_index_holds_no_pickle(tmp_path, isolated_config):
    clear_cache()
    path = str(tmp_path / "db")
    store = ChunkStore(list(CORPUS), [{} for _ in CORPUS])
    store.save(path)

    bm25 = BM25Index()
    bm25.build(list(CORPUS))
    bm25.save(path)

    raw = open(f"{path}.bm25", "rb").read()
    assert not raw.startswith(b"\x80")

    loaded = BM25Index()
    loaded.load(path)
    assert loaded.search_indices("alpha beta", 1) == [0]


# ------------------------------------------------------------- write ordering


def test_the_vector_index_is_the_last_file_written(tmp_path, isolated_config):
    """
    index_exists gates on the vector index, so it must appear last: a crash
    part-way then reads as "not built" rather than as an index promising chunks
    that were never stored.
    """
    clear_cache()
    path = str(tmp_path / "db")
    written = []

    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32")[:3], list(CORPUS))
    bm25 = BM25Index()
    bm25.build(list(db.texts))

    real_save_vectors = db.save_vectors_only
    real_store_save = db.backend.store.save

    def track_store(p):
        written.append("chunks")
        return real_store_save(p)

    def track_vectors(p):
        written.append("vectors")
        return real_save_vectors(p)

    db.backend.store.save = track_store
    db.save_vectors_only = track_vectors

    save_collection(path, db, bm25)

    assert written == ["chunks", "vectors"], written
    assert index_exists(path)


def test_a_collection_written_by_save_collection_loads_back(tmp_path, isolated_config):
    clear_cache()
    path = str(tmp_path / "db")
    db = VectorDB(dim=4)
    db.add(np.eye(4, dtype="float32")[:3], list(CORPUS))
    bm25 = BM25Index()
    bm25.build(list(db.texts))
    save_collection(path, db, bm25)

    fresh_db = VectorDB(dim=4)
    fresh_db.load(path)
    fresh_bm = BM25Index()
    fresh_bm.load(path)

    assert fresh_db.texts == CORPUS
    assert fresh_db.texts is fresh_bm.texts
    assert fresh_bm.search_indices("iota kappa", 1) == [2]
