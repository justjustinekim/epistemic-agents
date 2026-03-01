"""Integration tests for the tiered orchestrator."""

from __future__ import annotations

from unittest.mock import patch

from epistemic_agents.orchestrator import Orchestrator, Tier
from epistemic_agents.schema import (
    Belief,
    ConfidenceLevel,
    ExecutorFeedback,
    StrategicHandoff,
    Verdict,
)


def _handoff() -> StrategicHandoff:
    return StrategicHandoff(
        intent="Test intent",
        beliefs=[
            Belief(id="b1", claim="Test", confidence=ConfidenceLevel.HIGH, justification="J"),
        ],
        plan_steps=["Step 1"],
    )


def _feedback() -> ExecutorFeedback:
    return ExecutorFeedback(
        step_completed=0,
        observations=["Done"],
        execution_result="Complete",
    )


def _verdict() -> Verdict:
    return Verdict(
        decision_point="Should we proceed?",
        recommendation="Yes, proceed with plan.",
        confidence=ConfidenceLevel.HIGH,
        key_risk="Minimal risk",
        tier_used="quick",
    )


@patch("epistemic_agents.orchestrator.client.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_run_quick(mock_thinker_sr, mock_orch_sr):
    """Test quick tier end-to-end."""
    mock_thinker_sr.return_value = _handoff()
    mock_orch_sr.return_value = _verdict()

    orch = Orchestrator(verbose=False)
    result = orch.run("Quick question: what is Redis?", Tier.QUICK)

    assert result.tier == Tier.QUICK
    assert result.verdict is not None
    assert result.handoff is not None


@patch("epistemic_agents.orchestrator.client.structured_request")
@patch("epistemic_agents.executor.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_run_standard(mock_thinker_sr, mock_executor_sr, mock_orch_sr):
    """Test standard tier with thinker-executor loop."""
    mock_thinker_sr.return_value = _handoff()
    mock_executor_sr.return_value = _feedback()
    mock_orch_sr.return_value = _verdict()

    orch = Orchestrator(verbose=False)
    result = orch.run("Standard task", Tier.STANDARD)

    assert result.tier == Tier.STANDARD
    assert result.verdict is not None
    assert result.conversation_log is not None


@patch("epistemic_agents.orchestrator.client.structured_request")
@patch("epistemic_agents.executor.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_run_deep_without_panel_fallback(mock_thinker_sr, mock_executor_sr, mock_orch_sr):
    """Test deep tier falls back to standard when no panel configured."""
    mock_thinker_sr.return_value = _handoff()
    mock_executor_sr.return_value = _feedback()
    mock_orch_sr.return_value = _verdict()

    orch = Orchestrator(panel=None, verbose=False)
    result = orch.run("Deep analysis task", Tier.DEEP)

    # Should fall back to standard
    assert result.tier == Tier.STANDARD


def test_auto_route_quick():
    """Test auto-routing classifies simple tasks as quick."""
    orch = Orchestrator(verbose=False)
    tier = orch._classify_task("What is Redis?")
    assert tier == Tier.QUICK


def test_auto_route_deep():
    """Test auto-routing classifies complex tasks as deep."""
    orch = Orchestrator(verbose=False)
    tier = orch._classify_task(
        "Compare the tradeoffs between microservices and monolith architecture "
        "for a high-stakes critical system with long-term investment implications"
    )
    assert tier == Tier.DEEP


def test_auto_route_standard():
    """Test auto-routing defaults to standard."""
    orch = Orchestrator(verbose=False)
    tier = orch._classify_task("Implement a user registration endpoint with validation")
    assert tier == Tier.STANDARD
