"""LLM provider tests with the network mocked out."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from llm.base_provider import BaseLLMProvider, CompletionResult, LLMProviderError
from llm.ollama_provider import OllamaProvider

PAYLOAD = {"period": {"start": "2026-03-01", "end": "2026-03-30", "days": 30}, "period_averages": {"x": 1}}


class EchoProvider(BaseLLMProvider):
    name = "echo"

    def _complete(self, system: str, user: str) -> CompletionResult:
        return CompletionResult(text=f"## Executive summary\n{len(user)}", model=self.model)


def test_base_provider_wraps_payload() -> None:
    report = EchoProvider("echo-1").synthesize_insights(PAYLOAD)
    assert report.provider == "echo" and report.markdown.startswith("## Executive summary")
    message = BaseLLMProvider.build_user_message(PAYLOAD)
    assert "<telemetry_payload>" in message and "2026-03-01" in message


def test_empty_payload_rejected() -> None:
    with pytest.raises(LLMProviderError):
        EchoProvider("echo-1").synthesize_insights({})


def test_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        captured.update(url=url, body=json)
        return httpx.Response(200, json={"model": "llama3.1:8b", "message": {"content": "Summary"},
                                         "prompt_eval_count": 100, "eval_count": 20, "done_reason": "stop"})

    monkeypatch.setattr(httpx, "post", fake_post)
    report = OllamaProvider(model="llama3.1:8b").synthesize_insights(PAYLOAD)
    assert report.markdown == "Summary" and report.usage["output_tokens"] == 20
    assert captured["url"].endswith("/api/chat") and captured["body"]["stream"] is False
    assert captured["body"]["messages"][0]["role"] == "system"


def test_ollama_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMProviderError, match="ollama pull"):
        OllamaProvider(model="nope").synthesize_insights(PAYLOAD)


def _fake_message(stop_reason: str = "end_turn", text: str = "## Executive summary\nAll good") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0),
    )


def test_anthropic_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(api_key="sk-test")
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _fake_message()

    monkeypatch.setattr(provider._client.beta.messages, "create", fake_create)
    report = provider.synthesize_insights(PAYLOAD)
    assert report.markdown.endswith("All good") and report.model == "claude-opus-5-5"
    assert captured["model"] == "claude-opus-5-5"
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "high"}
    assert captured["fallbacks"] == "default" and captured["betas"] == [FALLBACK_BETA]


def test_anthropic_without_fallbacks_uses_standard_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(api_key="sk-test", enable_fallbacks=False)
    monkeypatch.setattr(provider._client.messages, "create", lambda **kwargs: _fake_message())
    assert provider.synthesize_insights(PAYLOAD).markdown.endswith("All good")


def test_anthropic_refusal_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(api_key="sk-test")
    monkeypatch.setattr(provider._client.beta.messages, "create", lambda **kwargs: _fake_message("refusal", ""))
    with pytest.raises(LLMProviderError, match="declined"):
        provider.synthesize_insights(PAYLOAD)
