import numpy as np

from api.vectordb import VectorDB


def test_vectordb_save_load_and_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    db = VectorDB(dim=2)
    db.add(
        np.array([[0.0, 0.0], [10.0, 10.0]], dtype="float32"),
        ["near origin", "far away"],
        [{"zim_title": "Near Docs"}, {"zim_title": "Far Docs"}],
    )
    db.save("sample")

    loaded = VectorDB(dim=2)
    loaded.load("sample")

    assert loaded.texts == ["near origin", "far away"]
    assert loaded.metadata == [{"zim_title": "Near Docs"}, {"zim_title": "Far Docs"}]
    assert loaded.search([0.0, 0.0], top_k=1) == ["near origin"]
    assert loaded.search_indices([0.0, 0.0], top_k=1) == [0]


def test_vectordb_loads_legacy_text_pickle(tmp_path, monkeypatch):
    import pickle

    monkeypatch.chdir(tmp_path)

    db = VectorDB(dim=2)
    db.add(np.array([[0.0, 0.0]], dtype="float32"), ["legacy chunk"])
    db.save("legacy")

    with open("legacy.pkl", "wb") as f:
        pickle.dump(["legacy chunk"], f)

    loaded = VectorDB(dim=2)
    loaded.load("legacy")

    assert loaded.texts == ["legacy chunk"]
    assert loaded.metadata == [{}]
