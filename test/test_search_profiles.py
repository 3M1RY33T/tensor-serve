import argparse
import json

import api.config as cfg
import cli.__main__ as cli_main
from api.search_profiles import (
    RERANKER_MODELS,
    get_profile,
    merge_profile_with_overrides,
    validate_manual_config,
)
from fastapi.testclient import TestClient

import api.main as main


def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_FILE", str(tmp_path / "cfg.json"))
    monkeypatch.setattr(cfg, "SECRET_KEY_FILE", ".tensor_config.key")


def _profile_args(profile, **overrides):
    values = {
        "profile": profile,
        "server": None,
        "timeout": 60,
        "overrides": None,
        "keyword_backend": None,
        "semantic_backend": None,
        "max_candidates": None,
        "query_expansion_type": None,
        "enable_query_expansion": False,
        "disable_query_expansion": False,
        "enable_reranker": False,
        "disable_reranker": False,
        "reranker_model": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_profile_definitions_include_public_options():
    lightweight = get_profile("lightweight")
    balanced = get_profile("balanced")
    production = get_profile("production")

    assert lightweight["keyword_backend"] == "bm25_okapi"
    assert lightweight["semantic_backend"] == "faiss_flat"
    assert lightweight["query_expansion_type"] == "none"
    assert lightweight["reranker_enabled"] is False

    assert balanced["query_expansion_type"] == "none"
    assert balanced["reranker_model"] == "lightweight"

    assert production["keyword_backend"] == "bm25_plus"
    assert production["semantic_backend"] == "faiss_ivf"
    assert production["query_expansion_enabled"] is True
    assert production["query_expansion_type"] == "prf"
    assert production["reranker_model"] == "balanced"

    assert RERANKER_MODELS == {
        "lightweight": "ms-marco-MiniLM-L-6-v2",
        "balanced": "ms-marco-MiniLM-L-12-v2",
    }


def test_profile_merge_and_manual_validation():
    merged = merge_profile_with_overrides(
        "balanced",
        {
            "query_expansion_enabled": True,
            "query_expansion_type": "entity",
            "max_search_candidates": 80,
        },
    )

    assert merged["keyword_backend"] == "bm25_okapi"
    assert merged["query_expansion_enabled"] is True
    assert merged["query_expansion_type"] == "entity"
    assert merged["max_search_candidates"] == 80

    assert validate_manual_config(
        {"keyword_backend": "bm25_plus", "semantic_backend": "faiss_ivf"}
    )
    assert not validate_manual_config({"keyword_backend": "bm25_plus"})
    assert not validate_manual_config(
        {"keyword_backend": "unknown", "semantic_backend": "faiss_ivf"}
    )


def test_search_profiles_endpoint_lists_and_applies_profiles(tmp_path, monkeypatch):
    _isolate_config(tmp_path, monkeypatch)
    client = TestClient(main.app)

    response = client.get("/config/search-profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_profile"] == "balanced"
    assert "lightweight" in payload["profiles"]
    assert payload["profiles"]["lightweight"]["query_expansion_type"] == "none"
    assert payload["available_backends"]["keyword"] == ["bm25_okapi", "bm25_plus"]
    assert "balanced" in payload["reranker_models"]

    response = client.post("/config/search-profiles/production")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "production"
    assert cfg.get_config_value("search_profile") == "production"
    assert cfg.get_config_value("keyword_backend") == "bm25_plus"
    assert cfg.get_config_value("semantic_backend") == "faiss_ivf"
    assert cfg.get_config_value("query_expansion_type") == "prf"
    assert cfg.get_config_value("reranker_model") == "balanced"


def test_search_profile_endpoint_applies_overrides(tmp_path, monkeypatch):
    _isolate_config(tmp_path, monkeypatch)
    client = TestClient(main.app)

    response = client.post(
        "/config/search-profiles/balanced",
        json={
            "query_expansion_enabled": True,
            "query_expansion_type": "prf",
            "reranker_model": "balanced",
            "max_search_candidates": 75,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"] == "balanced"
    assert payload["applied_config"]["query_expansion_enabled"] is True
    assert cfg.get_config_value("query_expansion_enabled") is True
    assert cfg.get_config_value("query_expansion_type") == "prf"
    assert cfg.get_config_value("reranker_model") == "balanced"
    assert cfg.get_config_value("max_search_candidates") == 75


def test_search_profile_endpoint_rejects_invalid_profiles(tmp_path, monkeypatch):
    _isolate_config(tmp_path, monkeypatch)
    client = TestClient(main.app)

    response = client.post("/config/search-profiles/does-not-exist")
    assert response.status_code == 400
    assert "Unknown profile" in response.json()["detail"]

    response = client.post("/config/search-profiles/manual")
    assert response.status_code == 400
    assert "Manual profile requires overrides" in response.json()["detail"]

    response = client.post(
        "/config/search-profiles/manual",
        json={"keyword_backend": "not-real", "semantic_backend": "faiss_ivf"},
    )
    assert response.status_code == 400
    assert "Invalid manual configuration" in response.json()["detail"]


def test_cli_search_profiles_reads_local_config(tmp_path, monkeypatch, capsys):
    _isolate_config(tmp_path, monkeypatch)
    cfg.set_config_value("search_profile", "lightweight")

    cli_main.config_search_profiles(
        argparse.Namespace(server=None, timeout=60)
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["current_profile"] == "lightweight"
    assert "production" in payload["profiles"]
    assert payload["available_backends"]["semantic"] == ["faiss_flat", "faiss_ivf"]


def test_cli_set_search_profile_writes_local_config(tmp_path, monkeypatch, capsys):
    _isolate_config(tmp_path, monkeypatch)

    cli_main.config_set_search_profile(
        _profile_args(
            "manual",
            keyword_backend="bm25_plus",
            semantic_backend="faiss_ivf",
            max_candidates=123,
            query_expansion_type="prf",
            enable_reranker=True,
            reranker_model="balanced",
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "manual"
    assert cfg.get_config_value("search_profile") == "manual"
    assert cfg.get_config_value("keyword_backend") == "bm25_plus"
    assert cfg.get_config_value("semantic_backend") == "faiss_ivf"
    assert cfg.get_config_value("max_search_candidates") == 123
    assert cfg.get_config_value("query_expansion_enabled") is True
    assert cfg.get_config_value("query_expansion_type") == "prf"
    assert cfg.get_config_value("reranker_enabled") is True
    assert cfg.get_config_value("reranker_model") == "balanced"


def test_cli_set_search_profile_can_disable_profile_features(tmp_path, monkeypatch):
    _isolate_config(tmp_path, monkeypatch)

    cli_main.config_set_search_profile(
        _profile_args(
            "production",
            query_expansion_type="none",
            disable_reranker=True,
        )
    )

    assert cfg.get_config_value("search_profile") == "production"
    assert cfg.get_config_value("keyword_backend") == "bm25_plus"
    assert cfg.get_config_value("query_expansion_enabled") is False
    assert cfg.get_config_value("query_expansion_type") == "none"
    assert cfg.get_config_value("reranker_enabled") is False


def test_cli_set_search_profile_posts_to_server(monkeypatch, capsys):
    captured = {}

    def fake_post(server, path, params=None, json_body=None, timeout=60):
        captured.update(
            {
                "server": server,
                "path": path,
                "params": params,
                "json_body": json_body,
                "timeout": timeout,
            }
        )
        return {"status": "updated", "profile": "balanced"}

    monkeypatch.setattr(cli_main, "_server_post", fake_post)

    cli_main.config_set_search_profile(
        _profile_args(
            "balanced",
            server="http://localhost:8000",
            timeout=5,
            overrides='{"max_search_candidates": 44}',
            query_expansion_type="prf",
        )
    )

    assert captured == {
        "server": "http://localhost:8000",
        "path": "/config/search-profiles/balanced",
        "params": None,
        "json_body": {
            "max_search_candidates": 44,
            "query_expansion_type": "prf",
            "query_expansion_enabled": True,
        },
        "timeout": 5,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "updated",
        "profile": "balanced",
    }


def test_cli_search_profile_input_validation():
    try:
        cli_main.config_set_search_profile(
            _profile_args("manual", keyword_backend="bm25_plus")
        )
    except SystemExit as exc:
        assert "Invalid manual configuration" in str(exc)
    else:
        raise AssertionError("Expected invalid manual config to exit")

    try:
        cli_main.config_set_search_profile(
            _profile_args("balanced", max_candidates=0)
        )
    except SystemExit as exc:
        assert "--max-candidates must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected invalid max candidates to exit")

    try:
        cli_main.config_set_search_profile(
            _profile_args("balanced", overrides="[1, 2, 3]")
        )
    except SystemExit as exc:
        assert "--overrides must be a JSON object" in str(exc)
    else:
        raise AssertionError("Expected invalid overrides to exit")
