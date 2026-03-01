"""Integration tests for the epistemic feedback loop."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from epistemic_agents.ledger import BeliefLedger, BeliefOutcome
from epistemic_agents.loop import EpistemicLoop
from epistemic_agents.schema import (
    AmendmentType,
    Belief,
    ChallengedBelief,
    ConfidenceLevel,
    Escalation,
    EscalationSeverity,
    EscalationType,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)
from epistemic_agents.thinker import Thinker
from epistemic_agents.executor import Executor


def _handoff() -> StrategicHandoff:
    return StrategicHandoff(
        intent="Test intent",
        beliefs=[
            Belief(id="b1", claim="Claim A", confidence=ConfidenceLevel.HIGH, justification="J"),
        ],
        plan_steps=["Step 1", "Step 2"],
    )


def _converge_feedback() -> ExecutorFeedback:
    return ExecutorFeedback(
        step_completed=1,
        observations=["All good"],
        execution_result="Done",
    )


def _escalate_feedback() -> ExecutorFeedback:
    return ExecutorFeedback(
        step_completed=0,
        observations=["Found issue"],
        escalation_type=EscalationType.CONTRADICTION,
        escalations=[
            Escalation(
                type=EscalationType.CONTRADICTION,
                severity=EscalationSeverity.BLOCKING,
                detail="Blocking issue found",
            ),
        ],
        challenged_beliefs=[
            ChallengedBelief(belief_id="b1", evidence="Wrong"),
        ],
        decision_needed=True,
    )


def _revise_amendment() -> ThinkerAmendment:
    return ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        updated_beliefs=[
            Belief(id="b1", claim="Revised claim", confidence=ConfidenceLevel.MODERATE, justification="Updated"),
        ],
        guidance="Revised the strategy",
    )


def _abort_amendment() -> ThinkerAmendment:
    return ThinkerAmendment(
        amendment_type=AmendmentType.ABORT,
        guidance="Abandoning this approach",
    )


@patch("epistemic_agents.executor.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_convergence_path(mock_thinker_sr, mock_executor_sr):
    """Test that loop converges when executor reports no escalation."""
    mock_thinker_sr.return_value = _handoff()
    mock_executor_sr.return_value = _converge_feedback()

    thinker = Thinker(model="opus")
    executor = Executor(model="sonnet")
    loop = EpistemicLoop(thinker=thinker, executor=executor, verbose=False)

    log = loop.run("Test task")
    assert log.converged is True
    assert log.round_trips == 1
    assert len(log.entries) >= 2  # handoff + feedback


@patch("epistemic_agents.executor.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_max_rounds_path(mock_thinker_sr, mock_executor_sr):
    """Test that loop exits after max rounds when it never converges."""
    mock_thinker_sr.side_effect = [
        _handoff(),
        _revise_amendment(),
        _revise_amendment(),
    ]
    mock_executor_sr.side_effect = [
        _escalate_feedback(),
        _escalate_feedback(),
    ]

    thinker = Thinker(model="opus")
    executor = Executor(model="sonnet")
    loop = EpistemicLoop(thinker=thinker, executor=executor, max_rounds=2, verbose=False)

    log = loop.run("Test task")
    assert log.converged is False
    assert log.round_trips == 2


@patch("epistemic_agents.executor.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_abort_path(mock_thinker_sr, mock_executor_sr):
    """Test that loop exits when thinker aborts."""
    mock_thinker_sr.side_effect = [
        _handoff(),
        _abort_amendment(),
    ]
    mock_executor_sr.return_value = _escalate_feedback()

    thinker = Thinker(model="opus")
    executor = Executor(model="sonnet")
    loop = EpistemicLoop(thinker=thinker, executor=executor, verbose=False)

    log = loop.run("Test task")
    assert log.converged is False
    # Should have handoff, feedback, and abort amendment
    assert any(
        e.entry_type == "amendment" for e in log.entries
    )


@patch("epistemic_agents.executor.structured_request")
@patch("epistemic_agents.thinker.structured_request")
def test_ledger_recording(mock_thinker_sr, mock_executor_sr):
    """Test that ledger records outcomes after loop completes."""
    mock_thinker_sr.return_value = _handoff()
    mock_executor_sr.return_value = _converge_feedback()

    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")
        thinker = Thinker(model="opus")
        executor = Executor(model="sonnet")
        loop = EpistemicLoop(
            thinker=thinker, executor=executor, verbose=False, ledger=ledger
        )

        log = loop.run("Test task")
        assert len(ledger.records) == 1
        # WP3: converged beliefs are UNTESTED, not CONFIRMED
        assert ledger.records[0].outcome == BeliefOutcome.UNTESTED
