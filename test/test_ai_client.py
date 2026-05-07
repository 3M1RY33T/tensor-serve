import requests

from api.ai_client import AIClient


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_list_models_uses_openai_compatible_endpoint(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert url == "http://local.test/v1/models"
        assert headers == {}
        assert timeout == 5
        return FakeResponse(200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    assert AIClient.list_models("http://local.test") == [
        {"id": "model-a", "source": "openai"},
        {"id": "model-b", "source": "openai"},
    ]


def test_list_models_falls_back_to_ollama(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if url.endswith("/v1/models"):
            return FakeResponse(404, {})
        return FakeResponse(200, {"models": [{"name": "llama-local"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    assert AIClient.list_models("http://local.test/") == [
        {"id": "llama-local", "source": "ollama"}
    ]
    assert calls == [
        "http://local.test/v1/models",
        "http://local.test/api/tags",
    ]


def test_list_models_accepts_endpoint_that_already_includes_v1(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        assert url == "http://local.test/v1/models"
        assert headers == {"Authorization": "Bearer secret"}
        return FakeResponse(200, {"data": [{"id": "cloud-model"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    assert AIClient.list_models("http://local.test/v1", api_key="secret") == [
        {"id": "cloud-model", "source": "openai"}
    ]


def test_detect_local_endpoints_reports_reachable_servers(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if url == "http://localhost:11434/v1/models":
            return FakeResponse(200, {"data": [{"id": "llama3"}]})
        return FakeResponse(404, {})

    monkeypatch.setattr(requests, "get", fake_get)

    detected = AIClient.detect_local_endpoints()

    assert detected == [
        {
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "models": ["llama3"],
            "model_count": 1,
            "source": "openai",
        }
    ]
