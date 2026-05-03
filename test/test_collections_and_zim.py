import src.collections as collections
import src.zim_downloader as zim_downloader
from fastapi.testclient import TestClient

import main


def test_custom_collection_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    collections.init_collections()
    created = collections.create_custom_collection(
        "local_docs",
        "Local Docs",
        "Temporary local docs",
        ["/tmp/docs.zim"],
    )

    assert created["category"] == "custom"
    assert collections.get_collection("local_docs")["name"] == "Local Docs"
    assert collections.set_active_collection("local_docs") is True
    assert collections.get_active_collection()["id"] == "local_docs"
    assert collections.delete_custom_collection("local_docs") is True
    assert collections.get_active_collection() is None


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
    main.app_state.active_collection = None
    client = TestClient(main.app)

    response = client.post(
        "/collections/custom/create",
        json={
            "collection_id": "local_docs",
            "name": "Local Docs",
            "description": "Temporary local docs",
            "zim_paths": ["/tmp/docs.zim"],
        },
    )
    assert response.status_code == 200
    assert response.json()["collection_id"] == "local_docs"

    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json()["collections"]["local_docs"]["name"] == "Local Docs"

    response = client.delete("/collections/custom/local_docs")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


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
