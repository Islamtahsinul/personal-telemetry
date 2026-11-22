"""Swappable LLM providers for insight synthesis."""

from __future__ import annotations

from core.config import Settings, get_settings
from llm.base_provider import BaseLLMProvider, InsightReport, LLMProviderError
from llm.ollama_provider import OllamaProvider


def get_provider(settings: Settings | None = None, name: str | None = None) -> BaseLLMProvider:
    """Instantiate the configured provider (``LLM_PROVIDER``) or the one named.

    Raises:
        LLMProviderError: If the provider is ``none`` or unknown.
    """
    settings = settings or get_settings()
    choice = (name or settings.llm_provider).lower()
    if choice == "anthropic":
        from llm.anthropic_provider import AnthropicProvider

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicProvider(
            api_key=key,
            model=settings.anthropic_model,
            effort=settings.anthropic_effort,
            max_tokens=settings.anthropic_max_tokens,
            enable_fallbacks=settings.anthropic_enable_fallbacks,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if choice == "ollama":
        return OllamaProvider(
            host=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if choice == "none":
        raise LLMProviderError("LLM synthesis is disabled (LLM_PROVIDER=none)")
    raise LLMProviderError(f"Unknown LLM provider {choice!r}")


__all__ = ["BaseLLMProvider", "InsightReport", "LLMProviderError", "OllamaProvider", "get_provider"]
