import json

import api.config as cfg
import api.main as main
from fastapi.testclient import TestClient


def test_config_load_save(tmp_path):
    cfg.CONFIG_FILE = str(tmp_path / "cfg.json")
    cfg.SECRET_KEY_FILE = ".tensor_config.key"

    c = cfg.load_config()
    assert isinstance(c, dict)

    cfg.set_config_value("ai_endpoint", "http://example.local")
    assert cfg.get_config_value("ai_endpoint") == "http://example.local"

    with open(cfg.CONFIG_FILE, "r") as f:
        data = json.load(f)
    assert data["ai_endpoint"] == "http://example.local"


def test_secret_values_are_encrypted_at_rest(tmp_path):
    cfg.CONFIG_FILE = str(tmp_path / "cfg.json")
    cfg.SECRET_KEY_FILE = ".tensor_config.key"

    cfg.set_config_value("ai_api_key", "sk-test-secret")
    cfg.set_config_value("web_search_api_key", "web-secret")
    cfg.set_config_value("ai_extra_headers", {"X-Api-Key": "header-secret"})

    with open(cfg.CONFIG_FILE, "r") as f:
        data = json.load(f)

    assert data["ai_api_key"]["__tensor_encrypted__"] is True
    assert "sk-test-secret" not in json.dumps(data)
    assert "web-secret" not in json.dumps(data)
    assert "header-secret" not in json.dumps(data)
    assert cfg.get_config_value("ai_api_key") == "sk-test-secret"
    assert cfg.get_config_value("web_search_api_key") == "web-secret"
    assert cfg.get_config_value("ai_extra_headers") == {"X-Api-Key": "header-secret"}


def test_plaintext_secret_values_are_migrated_on_load(tmp_path):
    cfg.CONFIG_FILE = str(tmp_path / "cfg.json")
    cfg.SECRET_KEY_FILE = ".tensor_config.key"

    with open(cfg.CONFIG_FILE, "w") as f:
        json.dump({"ai_api_key": "legacy-secret"}, f)

    assert cfg.get_config_value("ai_api_key") == "legacy-secret"

    with open(cfg.CONFIG_FILE, "r") as f:
        data = json.load(f)

    assert data["ai_api_key"]["__tensor_encrypted__"] is True
    assert "legacy-secret" not in json.dumps(data)


def test_config_reset_endpoint_restores_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")
    cfg.set_config_value("ai_endpoint", "http://example.local")
    cfg.set_config_value("ai_api_key", "sk-test-secret")
    main.app_state.ai_client.update_config("http://example.local", "test-model")

    response = TestClient(main.app).post("/config/reset")

    assert response.status_code == 200
    assert response.json()["status"] == "reset"
    assert cfg.get_config_value("ai_endpoint") is None
    assert cfg.get_config_value("ai_api_key") is None
    assert main.app_state.ai_client.is_configured() is False
