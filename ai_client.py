import requests
from typing import Optional, List, Dict
from config import get_config_value


class AIClient:
    """Client for communicating with a local AI endpoint."""

    def __init__(self):
        self.endpoint = get_config_value("ai_endpoint")
        self.model = get_config_value("ai_model")

    def is_configured(self) -> bool:
        """Check if AI endpoint is configured."""
        return self.endpoint is not None

    def update_config(self, endpoint: str, model: str):
        """Update AI endpoint configuration."""
        self.endpoint = endpoint
        self.model = model

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
        """Build a prompt with context."""
        if not context or len(context) == 0:
            return message

        context_text = "\n\n".join([f"- {chunk}" for chunk in context])
        return (
            f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {message}"
        )
