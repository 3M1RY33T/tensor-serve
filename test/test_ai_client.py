import requests

from ai_client import AIClient


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_build_prompt_deduplicates_context():
    client = AIClient()

    prompt = client._build_prompt(
        "What is Python?",
        ["Python is a language.", "Python is a language.", "It is widely used."],
    )

    assert prompt.count("Python is a language.") == 1
    assert "Question: What is Python?" in prompt


def test_list_models_uses_openai_compatible_endpoint(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://local.test/v1/models"
        assert timeout == 5
        return FakeResponse(200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    assert AIClient.list_models("http://local.test") == [
        {"id": "model-a", "source": "openai"},
        {"id": "model-b", "source": "openai"},
    ]


def test_list_models_falls_back_to_ollama(monkeypatch):
    calls = []

    def fake_get(url, timeout):
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
