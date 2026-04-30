import numpy as np

from vectordb import VectorDB


def test_vectordb_save_load_and_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    db = VectorDB(dim=2)
    db.add(
        np.array([[0.0, 0.0], [10.0, 10.0]], dtype="float32"),
        ["near origin", "far away"],
    )
    db.save("sample")

    loaded = VectorDB(dim=2)
    loaded.load("sample")

    assert loaded.texts == ["near origin", "far away"]
    assert loaded.search([0.0, 0.0], top_k=1) == ["near origin"]
    assert loaded.search_indices([0.0, 0.0], top_k=1) == [0]
