import json

from fastapi.testclient import TestClient

import main as main
from main import app


class FakeResponse:
    def __init__(
        self,
        content=b'{"proxied":true}',
        status_code=200,
        headers=None,
        chunks=None,
    ):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self._chunks = chunks or [content]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"upstream returned {self.status_code}")

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def close(self):
        pass


def test_v1_models_is_proxied_to_upstream(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return FakeResponse(
            content=b'{"object":"list","data":[{"id":"upstream-model"}]}'
        )

    monkeypatch.setattr(main.requests, "request", fake_request)
    main.app_state.ai_client.update_config("http://upstream.local", "ignored")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "upstream-model"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://upstream.local/v1/models"


def test_chat_completions_injects_context_and_returns_raw_upstream_response(monkeypatch):
    captured = {}

    def fake_context(query):
        assert query == "Explain FastAPI routing"
        return ["FastAPI routes map HTTP requests to Python callables."]

    def fake_request(method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return FakeResponse(
            content=b'{"id":"real-upstream-id","choices":[]}',
            status_code=201,
        )

    monkeypatch.setattr(main, "_context_for_query", fake_context)
    monkeypatch.setattr(main.requests, "request", fake_request)
    main.app_state.ai_client.update_config("http://upstream.local", "local-model")

    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Explain FastAPI routing"}],
        "temperature": 0.2,
        "extra_body_field": {"kept": True},
    }

    response = TestClient(app).post("/v1/chat/completions", json=payload)

    assert response.status_code == 201
    assert response.json()["id"] == "real-upstream-id"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://upstream.local/v1/chat/completions"
    assert captured["json"]["temperature"] == 0.2
    assert captured["json"]["extra_body_field"] == {"kept": True}
    assert captured["json"]["messages"][0]["role"] == "system"
    assert "FastAPI routes map" in captured["json"]["messages"][0]["content"]
    assert captured["json"]["messages"][1] == payload["messages"][0]


def test_chat_completions_forwards_without_context_when_no_user_message(monkeypatch):
    captured = {}

    def fake_context(query):
        raise AssertionError("context lookup should not run without a user message")

    def fake_request(method, url, **kwargs):
        captured.update(kwargs)
        return FakeResponse(content=b'{"ok":true}')

    monkeypatch.setattr(main, "_context_for_query", fake_context)
    monkeypatch.setattr(main.requests, "request", fake_request)
    main.app_state.ai_client.update_config("http://upstream.local", "local-model")

    payload = {
        "model": "local-model",
        "messages": [{"role": "system", "content": "Be concise."}],
    }

    response = TestClient(app).post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert captured["json"] == payload


def test_generic_v1_endpoint_is_proxied_with_query_and_body(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return FakeResponse(content=b'{"embedding":[1,2,3]}')

    monkeypatch.setattr(main.requests, "request", fake_request)
    main.app_state.ai_client.update_config("http://upstream.local/", "ignored")

    payload = {"input": "hello", "model": "embedder"}
    response = TestClient(app).post("/v1/embeddings?trace=1", json=payload)

    assert response.status_code == 200
    assert response.json() == {"embedding": [1, 2, 3]}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://upstream.local/v1/embeddings?trace=1"
    assert json.loads(captured["data"].decode("utf-8")) == payload
    assert captured["json"] is None


def test_v1_proxy_requires_configured_ai_endpoint():
    main.app_state.ai_client.update_config(None, None)

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 400
    assert "AI endpoint not configured" in response.json()["detail"]
