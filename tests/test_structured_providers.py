"""Tests for structured provider output (Change 12)."""

from __future__ import annotations

from unittest.mock import patch

from epistemic_agents.panel import ModelPanel
from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.schema import (
    Belief,
    ConfidenceLevel,
    PanelResponse,
    ProviderPosition,
)
from conftest import FakeProvider


class StructuredFakeProvider(BaseProvider):
    """A provider that supports structured output."""

    def __init__(self, name: str = "structured-fake", beliefs: list[Belief] | None = None):
        self.name = name
        self.model_id = "fake-structured-v1"
        self._last_usage = None
        self._beliefs = beliefs or [
            Belief(
                id="fb1",
                claim="Test claim from structured output",
                confidence=ConfidenceLevel.HIGH,
                justification="Structured justification",
                reasoning_basis="empirical data analysis",
            )
        ]
        self.structured_calls: list[tuple[str, str]] = []
        self.plain_calls: list[tuple[str, str]] = []

    def analyze(self, task: str, system_prompt: str) -> str:
        self.plain_calls.append((task, system_prompt))
        return "Fallback plain analysis"

    @property
    def supports_structured_output(self) -> bool:
        return True

    def structured_analyze(self, task: str, system_prompt: str) -> PanelResponse:
        self.structured_calls.append((task, system_prompt))
        return PanelResponse(
            beliefs=self._beliefs,
            raw_analysis="Structured analysis output",
        )


class FailingStructuredProvider(BaseProvider):
    """A provider that claims structured support but fails."""

    def __init__(self):
        self.name = "failing-structured"
        self.model_id = "fail-v1"
        self._last_usage = None

    def analyze(self, task: str, system_prompt: str) -> str:
        return "Fallback plain analysis after structured failure"

    @property
    def supports_structured_output(self) -> bool:
        return True

    def structured_analyze(self, task: str, system_prompt: str) -> PanelResponse:
        raise RuntimeError("Structured output failed")


def test_structured_path_skips_extraction():
    """When provider supports structured output, extraction step is skipped."""
    provider = StructuredFakeProvider()
    position = ModelPanel._query_provider(
        provider, "test task", "system prompt",
        extract_beliefs=True, extraction_model="haiku",
    )
    assert len(provider.structured_calls) == 1
    assert len(provider.plain_calls) == 0
    assert position.beliefs[0].claim == "Test claim from structured output"
    assert position.raw_analysis == "Structured analysis output"


def test_fallback_when_structured_fails():
    """When structured output fails, falls back to plain analyze + extraction."""
    provider = FailingStructuredProvider()
    with patch("epistemic_agents.belief_extractor.extract_beliefs", return_value=[]):
        position = ModelPanel._query_provider(
            provider, "test task", "system prompt",
            extract_beliefs=True, extraction_model="haiku",
        )
    assert position.raw_analysis == "Fallback plain analysis after structured failure"


def test_plain_path_when_not_supported():
    """Non-structured providers use plain analyze + extraction."""
    provider = FakeProvider()
    assert not provider.supports_structured_output
    with patch("epistemic_agents.belief_extractor.extract_beliefs", return_value=[]):
        position = ModelPanel._query_provider(
            provider, "test task", "system prompt",
            extract_beliefs=True, extraction_model="haiku",
        )
    assert len(provider.calls) == 1
    assert position.raw_analysis == "This is a fake analysis."


def test_plain_path_when_extract_disabled():
    """Structured output not used when extract_beliefs=False."""
    provider = StructuredFakeProvider()
    position = ModelPanel._query_provider(
        provider, "test task", "system prompt",
        extract_beliefs=False,
    )
    assert len(provider.structured_calls) == 0
    assert len(provider.plain_calls) == 1


def test_reasoning_basis_serialization():
    """reasoning_basis field survives Belief serialization roundtrip."""
    belief = Belief(
        id="b1",
        claim="Test claim",
        confidence=ConfidenceLevel.HIGH,
        justification="Test",
        reasoning_basis="empirical observation of latency metrics",
    )
    data = belief.model_dump()
    restored = Belief.model_validate(data)
    assert restored.reasoning_basis == "empirical observation of latency metrics"


def test_reasoning_basis_defaults_none():
    """reasoning_basis defaults to None for backward compat."""
    belief = Belief(
        id="b1",
        claim="Test",
        confidence=ConfidenceLevel.HIGH,
        justification="Test",
    )
    assert belief.reasoning_basis is None


def test_panel_response_validation():
    """PanelResponse model validates correctly."""
    response = PanelResponse(
        beliefs=[
            Belief(
                id="b1", claim="test", confidence=ConfidenceLevel.HIGH,
                justification="j", reasoning_basis="data-driven",
            )
        ],
        raw_analysis="Analysis text",
    )
    assert len(response.beliefs) == 1
    assert response.raw_analysis == "Analysis text"
    # Roundtrip
    data = response.model_dump()
    restored = PanelResponse.model_validate(data)
    assert restored.beliefs[0].reasoning_basis == "data-driven"


def test_structured_provider_preserves_reasoning_basis():
    """Structured output path preserves reasoning_basis on beliefs."""
    beliefs = [
        Belief(
            id="b1", claim="Redis improves latency",
            confidence=ConfidenceLevel.HIGH, justification="Benchmarks",
            reasoning_basis="p99 latency data from production",
        )
    ]
    provider = StructuredFakeProvider(beliefs=beliefs)
    position = ModelPanel._query_provider(
        provider, "test", "system",
        extract_beliefs=True,
    )
    assert position.beliefs[0].reasoning_basis == "p99 latency data from production"
