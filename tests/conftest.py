"""Shared test fixtures — FakeProvider, belief factory, mock helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.schema import (
    Belief,
    ConfidenceLevel,
    StrategicHandoff,
)


class FakeProvider(BaseProvider):
    """A deterministic provider for integration testing.

    Records all calls and returns configurable responses.
    """

    def __init__(
        self,
        name: str = "fake",
        model_id: str = "fake-v1",
        response: str = "This is a fake analysis.",
        is_available: bool = True,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._response = response
        self._is_available = is_available
        self.calls: list[tuple[str, str]] = []

    def analyze(self, task: str, system_prompt: str) -> str:
        self.calls.append((task, system_prompt))
        return self._response

    @property
    def available(self) -> bool:
        return self._is_available


@pytest.fixture
def fake_provider():
    """Create a FakeProvider with default settings."""
    return FakeProvider()


@pytest.fixture
def make_belief():
    """Factory fixture to create Belief objects with sensible defaults."""

    def _make(
        id: str = "b1",
        claim: str = "Test claim",
        confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
        justification: str = "Test justification",
        **kwargs,
    ) -> Belief:
        return Belief(
            id=id,
            claim=claim,
            confidence=confidence,
            justification=justification,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_handoff(make_belief):
    """Factory fixture to create StrategicHandoff objects."""

    def _make(
        intent: str = "Test intent",
        beliefs: list[Belief] | None = None,
        plan_steps: list[str] | None = None,
        **kwargs,
    ) -> StrategicHandoff:
        return StrategicHandoff(
            intent=intent,
            beliefs=beliefs or [make_belief()],
            plan_steps=plan_steps or ["Step 1"],
            **kwargs,
        )

    return _make


@pytest.fixture
def mock_structured_request():
    """Create a mock patcher for client.structured_request."""

    def _mock(return_value):
        patcher = patch(
            "epistemic_agents.client.structured_request",
            return_value=return_value,
        )
        return patcher

    return _mock
