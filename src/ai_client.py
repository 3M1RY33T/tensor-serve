from typing import Dict, List, Optional

import requests

from src.config import get_config_value


class AIClient:
    """Client for communicating with a local AI endpoint."""

    def __init__(self):
        self.endpoint = get_config_value("ai_endpoint")
        self.model = get_config_value("ai_model")

    def is_configured(self) -> bool:
        """Check if AI endpoint is configured."""
        return self.endpoint is not None

    def update_config(self, endpoint: Optional[str], model: Optional[str]):
        """Update AI endpoint configuration."""
        self.endpoint = endpoint
        self.model = model

    @staticmethod
    def list_models(endpoint: str) -> List[Dict]:
        """
        Probe an endpoint and return its available models.

        Tries the OpenAI-compatible ``GET /v1/models`` route first (covers
        vLLM, LM Studio, LocalAI, and most other servers), then falls back to
        Ollama's native ``GET /api/tags`` route.

        Returns a list of dicts, each with at minimum an ``id`` key containing
        the model name/identifier as the server reports it.

        Raises ``RuntimeError`` if neither route responds successfully.
        """
        base = endpoint.rstrip("/")

        # --- OpenAI-compatible /v1/models -----------------------------------
        try:
            resp = requests.get(f"{base}/v1/models", timeout=5)
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
            resp = requests.get(f"{base}/api/tags", timeout=5)
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
