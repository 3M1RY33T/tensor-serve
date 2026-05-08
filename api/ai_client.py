from typing import Dict, List, Optional

import requests

from api.config import get_config_value


class AIClient:
    """Client for communicating with a local AI endpoint."""

    def __init__(self):
        self.provider = get_config_value("ai_provider") or "openai-compatible"
        self.endpoint = get_config_value("ai_endpoint")
        self.model = get_config_value("ai_model")
        self.api_key = get_config_value("ai_api_key")
        self.api_key_header = get_config_value("ai_api_key_header") or "Authorization"
        self.api_key_prefix = get_config_value("ai_api_key_prefix")
        self.extra_headers = get_config_value("ai_extra_headers") or {}

    def is_configured(self) -> bool:
        """Check if AI endpoint is configured."""
        return self.endpoint is not None

    def update_config(
        self,
        endpoint: Optional[str],
        model: Optional[str],
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_header: Optional[str] = None,
        api_key_prefix: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """Update AI endpoint configuration."""
        self.provider = provider or "openai-compatible"
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.api_key_header = api_key_header or "Authorization"
        self.api_key_prefix = api_key_prefix
        self.extra_headers = extra_headers or {}

    def auth_headers(self) -> Dict[str, str]:
        """Return configured provider headers for upstream requests."""
        headers = dict(self.extra_headers or {})
        if self.api_key:
            value = self.api_key
            if self.api_key_prefix:
                value = f"{self.api_key_prefix} {self.api_key}"
            headers[self.api_key_header or "Authorization"] = value
        return headers

    @staticmethod
    def endpoint_url(endpoint: str, path: str) -> str:
        """Join endpoint and API path, avoiding duplicate /v1 segments."""
        base = endpoint.rstrip("/")
        path = path.lstrip("/")
        if base.endswith("/v1") and path.startswith("v1/"):
            path = path[3:]
        return f"{base}/{path}"

    @staticmethod
    def _auth_headers(
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_prefix: Optional[str] = "Bearer",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers = dict(extra_headers or {})
        if api_key:
            value = api_key
            if api_key_prefix:
                value = f"{api_key_prefix} {api_key}"
            headers[api_key_header] = value
        return headers

    @staticmethod
    def list_models(
        endpoint: str,
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_prefix: Optional[str] = "Bearer",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Probe an endpoint and return its available models.

        Tries the OpenAI-compatible ``GET /v1/models`` route first (covers
        vLLM, LM Studio, LocalAI, and most other servers), then falls back to
        Ollama's native ``GET /api/tags`` route.

        Returns a list of dicts, each with at minimum an ``id`` key containing
        the model name/identifier as the server reports it.

        Raises ``RuntimeError`` if neither route responds successfully.
        """
        headers = AIClient._auth_headers(
            api_key,
            api_key_header,
            api_key_prefix,
            extra_headers,
        )

        # --- OpenAI-compatible /v1/models -----------------------------------
        try:
            resp = requests.get(
                AIClient.endpoint_url(endpoint, "v1/models"),
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data.get("data"), list):
                    return [
                        {"id": m["id"], "source": "openai"}
                        for m in data["data"]
                        if "id" in m
                    ]
        except requests.exceptions.RequestException:
            pass

        # --- Ollama /api/tags -----------------------------------------------
        try:
            resp = requests.get(
                AIClient.endpoint_url(endpoint, "api/tags"),
                headers=headers,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data.get("models"), list):
                    return [
                        {"id": m["name"], "source": "ollama"}
                        for m in data["models"]
                        if "name" in m
                    ]
        except requests.exceptions.RequestException:
            pass

        raise RuntimeError(
            f"Could not detect models from '{endpoint}'. "
            "Ensure the server is running and the URL is correct."
        )

    @staticmethod
    def detect_local_endpoints() -> List[Dict]:
        """Probe common local AI servers and return reachable endpoints."""
        candidates = [
            {"provider": "ollama", "endpoint": "http://localhost:11434"},
            {"provider": "lm-studio", "endpoint": "http://localhost:1234"},
            {"provider": "lm-studio", "endpoint": "http://127.0.0.1:1234"},
            {"provider": "llama.cpp/localai", "endpoint": "http://localhost:8080"},
        ]
        found = []
        seen = set()
        for candidate in candidates:
            endpoint = candidate["endpoint"]
            if endpoint in seen:
                continue
            seen.add(endpoint)
            try:
                models = AIClient.list_models(endpoint)
            except RuntimeError:
                continue
            found.append(
                {
                    **candidate,
                    "models": [model["id"] for model in models],
                    "model_count": len(models),
                    "source": models[0]["source"] if models else None,
                }
            )
        return found
