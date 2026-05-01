import src.presets as presets
import src.zim_downloader as zim_downloader


def test_custom_preset_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    presets.init_presets()
    created = presets.create_custom_preset(
        "local_docs",
        "Local Docs",
        "Temporary local docs",
        ["/tmp/docs.zim"],
    )

    assert created["category"] == "custom"
    assert presets.get_preset("local_docs")["name"] == "Local Docs"
    assert presets.set_active_preset("local_docs") is True
    assert presets.get_active_preset()["id"] == "local_docs"
    assert presets.delete_custom_preset("local_docs") is True
    assert presets.get_active_preset() is None


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
