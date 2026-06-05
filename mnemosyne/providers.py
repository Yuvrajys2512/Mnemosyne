from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base interface for any LLM that can complete a system+user prompt."""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """
    Groq inference — uses the OpenAI-compatible API.
    Free tier: ~14,400 requests/day on Llama 3.3 70B.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"groq/{self._model}"

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=2_048,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    """
    Ollama local inference — uses the OpenAI-compatible API at localhost.
    Completely free, no rate limits, requires a running Ollama instance.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key="ollama",  # Ollama ignores the key but openai client requires one
            base_url=base_url,
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=2_048,
        )
        return response.choices[0].message.content or ""


def build_providers(
    groq_api_key: str | None = None,
    groq_model: str = "llama-3.3-70b-versatile",
    ollama_model: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434/v1",
) -> tuple[LLMProvider | None, LLMProvider | None]:
    """
    Return (primary, fallback) based on what's configured.

    Priority:
      1. Groq (if GROQ_API_KEY is set or passed explicitly)
      2. Ollama as primary if no Groq key
      3. Ollama always wired as fallback when Groq is primary
    """
    key = groq_api_key or os.environ.get("GROQ_API_KEY")
    ollama = OllamaProvider(model=ollama_model, base_url=ollama_base_url)

    if key:
        groq = GroqProvider(api_key=key, model=groq_model)
        return groq, ollama

    # No Groq key — Ollama is primary, no fallback
    logger.warning(
        "GROQ_API_KEY not set. Using Ollama as primary provider. "
        "Make sure Ollama is running at %s with model '%s'.",
        ollama_base_url,
        ollama_model,
    )
    return ollama, None
