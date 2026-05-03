import src.collections as collections
import src.zim_downloader as zim_downloader
from fastapi.testclient import TestClient

import main


def test_custom_collection_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    zim_file = tmp_path / "docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")

    collections.init_collections()
    created = collections.create_custom_collection(
        "local_docs",
        "Local Docs",
        "Temporary local docs",
        [str(zim_file)],
    )

    assert created["category"] == "collection"
    assert created["path"] == str(tmp_path / zim_downloader.ZIM_FOLDER / "local_docs")
    source_path = tmp_path / zim_downloader.ZIM_FOLDER / "docs.zim"
    linked_path = tmp_path / zim_downloader.ZIM_FOLDER / "local_docs" / "docs.zim"
    assert created["zim_files"][0]["path"] == str(linked_path)
    assert linked_path.read_text() == zim_file.read_text()
    assert source_path.read_text() == zim_file.read_text()
    assert linked_path.samefile(source_path)
    assert zim_file.exists()
    assert collections.get_collection("local_docs")["name"] == "Local Docs"
    assert collections.set_active_collection("local_docs") is True
    assert collections.get_active_collection()["id"] == "local_docs"
    assert collections.delete_custom_collection("local_docs") is True
    assert collections.get_active_collection() is None
    assert source_path.exists()


def test_scan_zim_folder_registers_untracked_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    zim_dir = tmp_path / zim_downloader.ZIM_FOLDER
    zim_dir.mkdir()
    zim_file = zim_dir / "sample_docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")

    installed = zim_downloader.scan_zim_folder()

    assert "sample_docs" in installed
    assert installed["sample_docs"]["untracked"] is True
    assert installed["sample_docs"]["path"] == str(zim_file)


def test_custom_zim_source_folder_skips_default_zim_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "external_zims"
    source_dir.mkdir()
    zim_file = source_dir / "external_docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")

    zim_downloader.set_zim_source_folder(str(source_dir))
    installed = zim_downloader.scan_zim_folder()

    assert not (tmp_path / zim_downloader.ZIM_FOLDER).exists()
    assert "external_docs" in installed
    assert installed["external_docs"]["path"] == str(zim_file)


def test_collection_folder_is_discovered_from_source_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "external_zims"
    collection_dir = source_dir / "local_docs"
    collection_dir.mkdir(parents=True)
    zim_file = collection_dir / "docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")

    zim_downloader.set_zim_source_folder(str(source_dir))

    all_collections = collections.get_all_collections()

    assert "local_docs" in all_collections
    assert all_collections["local_docs"]["path"] == str(collection_dir)
    assert all_collections["local_docs"]["zim_files"][0]["path"] == str(zim_file)


def test_register_existing_zim_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    zim_file = tmp_path / "custom_docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")

    info = zim_downloader.register_zim_file(str(zim_file), title="Custom Docs")

    assert info["title"] == "Custom Docs"
    assert info["path"] == str(zim_file)
    assert not (tmp_path / zim_downloader.ZIM_FOLDER).exists()


def test_collections_api_custom_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "source_zims"
    source_dir.mkdir()
    zim_file = source_dir / "docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")
    zim_downloader.set_zim_source_folder(str(source_dir))
    main.app_state.active_collection = None
    client = TestClient(main.app)

    response = client.post(
        "/collections",
        json={
            "collection_id": "local_docs",
            "name": "Local Docs",
            "description": "Temporary local docs",
            "zim_paths": [str(zim_file)],
        },
    )
    assert response.status_code == 200
    assert response.json()["collection_id"] == "local_docs"
    assert response.json()["file_count"] == 1
    linked_file = source_dir / "local_docs" / "docs.zim"
    assert linked_file.exists()
    assert linked_file.samefile(zim_file)
    assert zim_file.exists()

    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json()["collections"]["local_docs"]["name"] == "Local Docs"
    assert response.json()["collections"]["local_docs"]["file_count"] == 1

    response = client.delete("/collections/local_docs")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert not (source_dir / "local_docs").exists()
    assert zim_file.exists()


def test_collection_api_empty_rename_add_and_remove_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "source_zims"
    source_dir.mkdir()
    docs_file = tmp_path / "docs.zim"
    other_file = tmp_path / "other.zim"
    docs_file.write_text("docs fixture")
    other_file.write_text("other fixture")
    zim_downloader.set_zim_source_folder(str(source_dir))
    client = TestClient(main.app)

    response = client.post(
        "/collections",
        json={
            "collection_id": "local_docs",
            "name": "Local Docs",
            "description": "Temporary local docs",
            "zim_paths": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["file_count"] == 0

    response = client.patch(
        "/collections/local_docs",
        json={"name": "Renamed Docs"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed Docs"

    response = client.post(
        "/collections/local_docs/files",
        json={"zim_paths": [str(docs_file), str(other_file)]},
    )
    assert response.status_code == 200
    assert response.json()["file_count"] == 2
    assert (source_dir / "local_docs" / "docs.zim").read_text() == "docs fixture"
    assert (source_dir / "local_docs" / "other.zim").read_text() == "other fixture"
    assert (source_dir / "docs.zim").exists()
    assert (source_dir / "other.zim").exists()
    assert (source_dir / "local_docs" / "docs.zim").samefile(source_dir / "docs.zim")
    assert docs_file.exists()
    assert other_file.exists()

    response = client.request(
        "DELETE",
        "/collections/local_docs/files",
        json={"file_names": ["docs.zim"]},
    )
    assert response.status_code == 200
    assert response.json()["file_count"] == 1
    assert not (source_dir / "local_docs" / "docs.zim").exists()
    assert (source_dir / "docs.zim").exists()
    assert docs_file.exists()

    response = client.get("/collections/local_docs/files")
    assert response.status_code == 200
    assert response.json()["files"][0]["name"] == "other.zim"


def test_zim_source_folder_api_uses_custom_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "external_zims"
    source_dir.mkdir()
    zim_file = source_dir / "external_docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")
    client = TestClient(main.app)

    response = client.put("/zim/source-folder", json={"path": str(source_dir)})
    assert response.status_code == 200
    assert response.json()["path"] == str(source_dir)
    assert response.json()["custom"] is True

    response = client.get("/zim/installed")
    assert response.status_code == 200
    assert "external_docs" in response.json()["installed_files"]
    assert not (tmp_path / zim_downloader.ZIM_FOLDER).exists()

    response = client.post("/zim/register", json={"path": str(zim_file)})
    assert response.status_code == 200
    assert response.json()["file_id"] == "external_docs"


def test_resolve_zim_inputs_expands_directories(tmp_path):
    source_dir = tmp_path / "source_zims"
    collection_dir = source_dir / "local_docs"
    collection_dir.mkdir(parents=True)
    root_zim = source_dir / "root.zim"
    nested_zim = collection_dir / "nested.zim"
    ignored = collection_dir / "ignored.txt"
    root_zim.write_text("fixture")
    nested_zim.write_text("fixture")
    ignored.write_text("fixture")

    resolved = zim_downloader.resolve_zim_inputs([str(source_dir), str(root_zim)])

    assert resolved == [str(nested_zim), str(root_zim)]


def test_resolve_zim_inputs_deduplicates_collection_links(tmp_path):
    source_dir = tmp_path / "source_zims"
    collection_dir = source_dir / "local_docs"
    collection_dir.mkdir(parents=True)
    source_zim = source_dir / "docs.zim"
    linked_zim = collection_dir / "docs.zim"
    source_zim.write_text("fixture")
    linked_zim.hardlink_to(source_zim)

    resolved = zim_downloader.resolve_zim_inputs([str(source_dir)])

    assert len(resolved) == 1


def test_scan_zim_folder_deduplicates_collection_links(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "source_zims"
    collection_dir = source_dir / "local_docs"
    collection_dir.mkdir(parents=True)
    source_zim = source_dir / "docs.zim"
    linked_zim = collection_dir / "docs.zim"
    source_zim.write_text("fixture")
    linked_zim.hardlink_to(source_zim)
    zim_downloader.set_zim_source_folder(str(source_dir))

    installed = zim_downloader.scan_zim_folder()

    assert list(installed) == ["docs"]
    assert installed["docs"]["path"] == str(source_zim)


def test_ingest_multiple_api_expands_collection_directory(tmp_path, monkeypatch):
    source_dir = tmp_path / "source_zims"
    collection_dir = source_dir / "local_docs"
    collection_dir.mkdir(parents=True)
    zim_file = collection_dir / "docs.zim"
    zim_file.write_text("not a real zim, just a manifest fixture")
    captured = {}

    def fake_run_multi_ingest(zim_paths, output_name):
        captured["zim_paths"] = zim_paths
        captured["output_name"] = output_name
        return {"status": "ok"}

    monkeypatch.setattr(main, "run_multi_ingest", fake_run_multi_ingest)
    client = TestClient(main.app)

    response = client.post(
        "/ingest-multiple",
        json={"zim_paths": [str(collection_dir)], "output_name": "local_docs_db"},
    )

    assert response.status_code == 200
    assert captured["zim_paths"] == [str(zim_file)]
    assert captured["output_name"] == "local_docs_db"


def test_install_category_accepts_named_categories(monkeypatch):
    monkeypatch.setattr(main, "is_file_installed", lambda file_id: True)
    client = TestClient(main.app)

    response = client.post("/zim/install-category", json={"category_id": "learning"})

    assert response.status_code == 200
    assert response.json()["category"] == "Learning"
    assert response.json()["status"] == "nothing_to_do"
    assert response.json()["already_installed"] == [
        "wikiversity_en_all",
        "wikibooks_en_all",
    ]
