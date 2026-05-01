import json

import src.config as cfg


def test_config_load_save(tmp_path):
    cfg.CONFIG_FILE = str(tmp_path / "cfg.json")

    c = cfg.load_config()
    assert isinstance(c, dict)

    cfg.set_config_value("ai_endpoint", "http://example.local")
    assert cfg.get_config_value("ai_endpoint") == "http://example.local"

    with open(cfg.CONFIG_FILE, "r") as f:
        data = json.load(f)
    assert data["ai_endpoint"] == "http://example.local"
