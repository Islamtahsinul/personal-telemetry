"""Claude insight synthesis via the official ``anthropic`` SDK (personal API key).

Note: unlike the Ollama provider, this sends the aggregated payload (never raw
rows, notes or transactions) to Anthropic's API.
"""

from __future__ import annotations

import os
from typing import Any

from llm.base_provider import BaseLLMProvider, CompletionResult, LLMProviderError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(BaseLLMProvider):
    """Generate insights with Claude.

    Uses adaptive thinking with a configurable effort level. Server-side refusal
    fallbacks (``fallbacks="default"``) are enabled by default so that a request
    declined by one model's safety classifier is retried on a suitable fallback
    model within the same call; disable with ``ANTHROPIC_ENABLE_FALLBACKS=false``.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        max_tokens: int = 16000,
        enable_fallbacks: bool = True,
        timeout_seconds: float = 600.0,
        system_prompt: str | None = None,
    ) -> None:
        """Create a Claude provider.

        Args:
            api_key: Anthropic API key. When ``None`` the SDK falls back to its own
                credential discovery (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
            model: Claude model id.
            effort: ``output_config.effort`` - low | medium | high | xhigh | max.
            max_tokens: Output ceiling (includes thinking tokens).
            enable_fallbacks: Opt into server-side refusal fallbacks (Claude API only).
            timeout_seconds: HTTP timeout.
            system_prompt: Optional system-prompt override.
        """
        super().__init__(model, timeout_seconds, system_prompt)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMProviderError("anthropic is not installed: pip install anthropic") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=3)
        self.effort = effort
        self.max_tokens = max_tokens
        self.enable_fallbacks = enable_fallbacks
        self._has_key = api_key is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))

    def is_available(self) -> bool:
        """``True`` when an API key is configured (``ant auth login`` profiles are not probed)."""
        return self._has_key

    def _complete(self, system: str, user: str) -> CompletionResult:
        anthropic = self._anthropic
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        try:
            if self.enable_fallbacks:
                response = self._client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                response = self._client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise LLMProviderError("Anthropic rejected the API key (401) - check ANTHROPIC_API_KEY") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMProviderError(f"Anthropic permission denied: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMProviderError(f"Unknown Anthropic model {self.model!r}: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMProviderError("Anthropic rate limit reached - retry later") from exc
        except anthropic.BadRequestError as exc:
            raise LLMProviderError(f"Anthropic rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMProviderError("Could not reach the Anthropic API - check your network") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMProviderError(f"Claude declined to generate insights (category: {category or 'unspecified'})")

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return CompletionResult(
            text=text,
            model=response.model,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
            truncated=response.stop_reason == "max_tokens",
        )
