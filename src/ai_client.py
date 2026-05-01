from typing import Dict, List, Optional
import difflib

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

    def chat(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Send a chat message to the AI model with optional context.

        Args:
            message: User message
            context: List of relevant context chunks from vector DB

        Returns:
            Response from AI model
        """
        if not self.is_configured():
            raise ValueError("AI endpoint not configured")

        # Build the prompt with context
        prompt = self._build_prompt(message, context)

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"AI endpoint error: {str(e)}")

    def _build_prompt(self, message: str, context: Optional[List[str]] = None) -> str:
        """Build a prompt with deduplicated context."""
        if not context or len(context) == 0:
            return message

        # Deduplicate context chunks
        deduped = self._deduplicate_context(context)
        context_text = "\n\n".join([f"- {chunk}" for chunk in deduped])
        return f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {message}"

    @staticmethod
    def _deduplicate_context(chunks: List[str], similarity_threshold: float = 0.85) -> List[str]:
        """
        Remove duplicate or near-duplicate chunks from context.

        Uses sequence matching to detect similar chunks and keeps only the first
        occurrence. Also merges chunks with significant overlap.

        Args:
            chunks: List of context chunks
            similarity_threshold: Similarity ratio (0-1) above which chunks are considered duplicates

        Returns:
            Deduplicated list of chunks
        """
        if not chunks or len(chunks) <= 1:
            return chunks

        deduped: List[str] = []
        for chunk in chunks:
            is_duplicate = False
            for existing in deduped:
                ratio = difflib.SequenceMatcher(None, chunk.lower(), existing.lower()).ratio()
                if ratio >= similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(chunk)

        return deduped
