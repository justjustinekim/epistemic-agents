"""Tests for BeliefLedger — calibration tracking and outcome recording."""

import json
import tempfile
from pathlib import Path

from epistemic_agents.schema import (
    Belief,
    ChallengedBelief,
    ConfidenceLevel,
    ConversationLog,
    EscalationType,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
    AmendmentType,
)
from epistemic_agents.ledger import (
    BeliefLedger,
    BeliefOutcome,
    BeliefRecord,
)


def _make_handoff(*beliefs: Belief) -> StrategicHandoff:
    return StrategicHandoff(
        intent="Test",
        beliefs=list(beliefs),
        plan_steps=["Step 1"],
    )


def _make_belief(id: str, confidence: ConfidenceLevel = ConfidenceLevel.HIGH) -> Belief:
    return Belief(
        id=id,
        claim=f"Claim for {id}",
        confidence=confidence,
        justification="Test justification",
    )


def test_ledger_records_confirmed_beliefs():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        log = ConversationLog(task="Test task")
        handoff = _make_handoff(_make_belief("b1"), _make_belief("b2"))
        log.add(role="thinker", entry_type="handoff", content=handoff)

        feedback = ExecutorFeedback(
            observations=["All good"],
            execution_result="Done",
        )
        log.add(role="executor", entry_type="feedback", content=feedback)
        log.converged = True

        records = ledger.record_outcomes(log)
        assert len(records) == 2
        assert all(r.outcome == BeliefOutcome.CONFIRMED for r in records)


def test_ledger_records_falsified_beliefs():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        log = ConversationLog(task="Test task")
        handoff = _make_handoff(_make_belief("b1"), _make_belief("b2"))
        log.add(role="thinker", entry_type="handoff", content=handoff)

        feedback = ExecutorFeedback(
            step_completed=0,
            observations=["Found issue"],
            escalation_type=EscalationType.CONTRADICTION,
            challenged_beliefs=[
                ChallengedBelief(belief_id="b1", evidence="b1 was wrong because X"),
            ],
            decision_needed=True,
        )
        log.add(role="executor", entry_type="feedback", content=feedback)

        records = ledger.record_outcomes(log)
        b1_record = next(r for r in records if r.belief_id == "b1")
        b2_record = next(r for r in records if r.belief_id == "b2")

        assert b1_record.outcome == BeliefOutcome.FALSIFIED
        assert b1_record.failure_reason == "b1 was wrong because X"
        assert b2_record.outcome == BeliefOutcome.UNTESTED


def test_ledger_records_revised_beliefs():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        log = ConversationLog(task="Test task")
        handoff = _make_handoff(_make_belief("b1"))
        log.add(role="thinker", entry_type="handoff", content=handoff)

        feedback = ExecutorFeedback(
            observations=["Need revision"],
            escalation_type=EscalationType.DISCOVERY,
            decision_needed=True,
        )
        log.add(role="executor", entry_type="feedback", content=feedback)

        amendment = ThinkerAmendment(
            amendment_type=AmendmentType.REVISE,
            updated_beliefs=[_make_belief("b1", ConfidenceLevel.MODERATE)],
            guidance="Updated b1",
        )
        log.add(role="thinker", entry_type="amendment", content=amendment)

        records = ledger.record_outcomes(log)
        assert records[0].outcome == BeliefOutcome.REVISED


def test_ledger_persistence():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.json"

        # Create and save
        ledger1 = BeliefLedger(path=path)
        log = ConversationLog(task="Task 1")
        log.add(
            role="thinker",
            entry_type="handoff",
            content=_make_handoff(_make_belief("b1")),
        )
        log.add(
            role="executor",
            entry_type="feedback",
            content=ExecutorFeedback(
                observations=["Done"], execution_result="OK"
            ),
        )
        log.converged = True
        ledger1.record_outcomes(log)

        # Load from disk
        ledger2 = BeliefLedger(path=path)
        assert len(ledger2.records) == 1
        assert ledger2.records[0].belief_id == "b1"


def test_calibration_report():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        # Simulate 6 beliefs across sessions
        for task_num in range(3):
            log = ConversationLog(task=f"Task {task_num}")
            handoff = _make_handoff(
                _make_belief(f"h{task_num}", ConfidenceLevel.HIGH),
                _make_belief(f"l{task_num}", ConfidenceLevel.LOW),
            )
            log.add(role="thinker", entry_type="handoff", content=handoff)

            if task_num == 2:
                # Third task: challenge the HIGH belief
                feedback = ExecutorFeedback(
                    observations=["Issue"],
                    challenged_beliefs=[
                        ChallengedBelief(
                            belief_id=f"h{task_num}",
                            evidence="Wrong",
                        ),
                    ],
                    escalation_type=EscalationType.CONTRADICTION,
                    decision_needed=True,
                )
            else:
                feedback = ExecutorFeedback(
                    observations=["Done"],
                    execution_result="OK",
                )
                log.converged = True

            log.add(role="executor", entry_type="feedback", content=feedback)
            ledger.record_outcomes(log)

        report = ledger.calibration_report()
        high_stats = next(s for s in report if s.level == ConfidenceLevel.HIGH)
        assert high_stats.total == 3
        assert high_stats.confirmed == 2
        assert high_stats.falsified == 1


def test_calibration_context_insufficient_data():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")
        # No data
        assert ledger.calibration_context() == ""

        # Only 2 beliefs (below threshold of 5)
        log = ConversationLog(task="Task")
        log.add(
            role="thinker",
            entry_type="handoff",
            content=_make_handoff(_make_belief("b1"), _make_belief("b2")),
        )
        log.add(
            role="executor",
            entry_type="feedback",
            content=ExecutorFeedback(observations=["Done"], execution_result="OK"),
        )
        log.converged = True
        ledger.record_outcomes(log)
        assert ledger.calibration_context() == ""


def test_calibration_context_with_data():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        # Add 6 beliefs (above threshold)
        for i in range(3):
            log = ConversationLog(task=f"Task {i}")
            log.add(
                role="thinker",
                entry_type="handoff",
                content=_make_handoff(
                    _make_belief(f"a{i}", ConfidenceLevel.HIGH),
                    _make_belief(f"b{i}", ConfidenceLevel.MODERATE),
                ),
            )
            log.add(
                role="executor",
                entry_type="feedback",
                content=ExecutorFeedback(
                    observations=["Done"], execution_result="OK"
                ),
            )
            log.converged = True
            ledger.record_outcomes(log)

        ctx = ledger.calibration_context()
        assert "CALIBRATION DATA" in ctx
        assert "6 beliefs" in ctx
        assert "HIGH" in ctx
