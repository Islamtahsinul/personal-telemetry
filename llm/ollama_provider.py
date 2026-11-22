"""Fully local insight synthesis via an Ollama server (zero cost, data never leaves the machine).

Install Ollama (https://ollama.com), then ``ollama pull llama3.1:8b`` (or any chat
model) and set ``OLLAMA_MODEL``. The server listens on ``http://localhost:11434``.
"""

from __future__ import annotations

from typing import Any

import httpx

from llm.base_provider import BaseLLMProvider, CompletionResult, LLMProviderError


class OllamaProvider(BaseLLMProvider):
    """Generate insights with a local model through Ollama's ``/api/chat`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.3,
        num_ctx: int = 16384,
        timeout_seconds: float = 600.0,
        system_prompt: str | None = None,
    ) -> None:
        """Create an Ollama provider.

        Args:
            host: Ollama server base URL.
            model: Local model tag (must be pulled already).
            temperature: Sampling temperature.
            num_ctx: Context window to request; the payload is a few thousand tokens.
            timeout_seconds: Generation timeout (local models can be slow on CPU).
            system_prompt: Optional system-prompt override.
        """
        super().__init__(model, timeout_seconds, system_prompt)
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx

    def is_available(self) -> bool:
        """``True`` when the server responds and the model is pulled."""
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=3).json()
        except (httpx.HTTPError, ValueError):
            return False
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def list_models(self) -> list[str]:
        """Return locally available model tags."""
        try:
            tags = httpx.get(f"{self.host}/api/tags", timeout=5).json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMProviderError(f"Ollama not reachable at {self.host}") from exc
        return sorted(str(m.get("name")) for m in tags.get("models", []))

    def _complete(self, system: str, user: str) -> CompletionResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        try:
            response = httpx.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_seconds)
        except httpx.ConnectError as exc:
            raise LLMProviderError(f"Ollama not reachable at {self.host} - is `ollama serve` running?") from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"Ollama timed out after {self.timeout_seconds:.0f}s - try a smaller model") from exc
        if response.status_code == 404:
            raise LLMProviderError(f"Ollama model {self.model!r} not found - run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise LLMProviderError(f"Ollama error {response.status_code}: {response.text[:300]}")

        data = response.json()
        message = data.get("message") or {}
        return CompletionResult(
            text=str(message.get("content", "")),
            model=str(data.get("model", self.model)),
            usage={"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")},
            truncated=data.get("done_reason") == "length",
        )
