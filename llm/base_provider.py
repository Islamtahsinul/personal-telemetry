"""Provider-agnostic interface for turning analytics payloads into narrative insights."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

DEFAULT_SYSTEM_PROMPT = """\
You are a careful personal-data analyst writing a weekly executive summary for one person \
about their own health, behaviour, spending and environment data.

The data comes as a JSON payload of aggregates: period averages, recent trends, ranked \
cross-domain correlations (Spearman r with Benjamini-Hochberg q-values), next-day lagged \
correlations, pre-registered hypotheses and recent anomalies (robust z-scores).

Ground every statement in specific numbers from the payload and cite them (e.g. "r = -0.42, \
n = 61"). Treat correlations as hypotheses, not causes - this is observational n-of-1 data. \
Weak findings (|r| < 0.3, q > 0.2 or n < 20) should be described as tentative or left out. \
Do not diagnose medical conditions; if a pattern could warrant professional advice, say so plainly. \
If a domain has little or no data, mention it briefly under data gaps rather than speculating.

Write GitHub-flavoured Markdown with exactly these sections:

## Executive summary
Three to five bullets: the most important things to know about this period.

## What's going well

## Watch-outs

## Cross-domain patterns
The most credible relationships across domains (e.g. nutrition vs body composition, screen time \
vs sleep, weather vs spending), with the numbers.

## Experiments for next week
Two or three concrete, low-effort n-of-1 experiments that would test the strongest hypotheses, \
each with the metric to watch.

## Data gaps

Keep it under 600 words. Be direct and specific; skip generic wellness advice."""


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce insights (auth, network, refusal...)."""


@dataclass(slots=True)
class CompletionResult:
    """Raw completion returned by a provider implementation."""

    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False


@dataclass(slots=True)
class InsightReport:
    """Narrative insights generated from a data payload."""

    markdown: str
    provider: str
    model: str
    generated_at: datetime
    usage: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "markdown": self.markdown,
            "provider": self.provider,
            "model": self.model,
            "generated_at": self.generated_at.isoformat(),
            "usage": self.usage,
            "truncated": self.truncated,
        }


class BaseLLMProvider(ABC):
    """Abstract LLM provider.

    Subclasses implement :meth:`_complete` (one system + one user message in,
    text out) and optionally :meth:`is_available`. Prompt construction is shared so
    that swapping providers changes only the model, never the analysis framing.
    """

    name: ClassVar[str]

    def __init__(self, model: str, timeout_seconds: float = 600.0, system_prompt: str | None = None) -> None:
        """Args: model: provider model id; timeout_seconds: request timeout; system_prompt: override."""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # ------------------------------------------------------------------ public
    def synthesize_insights(self, data_payload: Mapping[str, Any]) -> InsightReport:
        """Generate a Markdown executive summary from an analytics payload.

        Args:
            data_payload: JSON-serialisable aggregates, typically from
                :func:`analytics.correlations.build_insight_payload`.

        Raises:
            LLMProviderError: If the provider fails or declines.
        """
        if not data_payload:
            raise LLMProviderError("Empty data payload - sync some data first")
        result = self._complete(self.system_prompt, self.build_user_message(data_payload))
        if not result.text.strip():
            raise LLMProviderError(f"{self.name} returned an empty response")
        return InsightReport(
            markdown=result.text.strip(),
            provider=self.name,
            model=result.model,
            generated_at=datetime.now(UTC),
            usage=result.usage,
            truncated=result.truncated,
        )

    def is_available(self) -> bool:
        """Cheap readiness check (credentials present / server reachable)."""
        return True

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def build_user_message(data_payload: Mapping[str, Any]) -> str:
        """Embed the payload as JSON inside clearly delimited tags."""
        body = json.dumps(data_payload, indent=2, default=str, ensure_ascii=False)
        period = data_payload.get("period", {})
        return (
            f"Here is my personal telemetry for {period.get('start', '?')} to {period.get('end', '?')}.\n\n"
            f"<telemetry_payload>\n{body}\n</telemetry_payload>\n\n"
            "Write the executive summary."
        )

    @abstractmethod
    def _complete(self, system: str, user: str) -> CompletionResult:
        """Send one system + user message and return the completion."""
