"""Tests for BeliefLedger — calibration tracking and outcome recording."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
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
    VerificationMethod,
)
from epistemic_agents.ledger import (
    BeliefLedger,
    BeliefOutcome,
    BeliefRecord,
    classify_domain,
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


def test_ledger_records_untested_on_convergence():
    """WP3 fix: converged beliefs should be UNTESTED, not CONFIRMED."""
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
        assert all(r.outcome == BeliefOutcome.UNTESTED for r in records)
        assert all("Converged without positive verification" in r.failure_reason for r in records)


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

        report = ledger.calibration_report(decay_half_life_days=0)
        high_stats = next(s for s in report if s.level == ConfidenceLevel.HIGH)
        assert high_stats.total == 3
        # WP3: converged beliefs are now UNTESTED, not CONFIRMED
        assert high_stats.untested == 2
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

        # Add 6 beliefs (above threshold) — use mark_verified to get CONFIRMED
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
            # Explicitly verify them
            ledger.mark_verified(f"a{i}", VerificationMethod.EXECUTOR_CHALLENGE, BeliefOutcome.CONFIRMED)
            ledger.mark_verified(f"b{i}", VerificationMethod.EXECUTOR_CHALLENGE, BeliefOutcome.CONFIRMED)

        ctx = ledger.calibration_context()
        assert "CALIBRATION DATA" in ctx
        assert "HIGH" in ctx


# ---------------------------------------------------------------------------
# WP3: mark_verified tests
# ---------------------------------------------------------------------------


def test_mark_verified():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        log = ConversationLog(task="Test")
        log.add(
            role="thinker",
            entry_type="handoff",
            content=_make_handoff(_make_belief("b1")),
        )
        log.add(
            role="executor",
            entry_type="feedback",
            content=ExecutorFeedback(observations=["Done"], execution_result="OK"),
        )
        log.converged = True
        ledger.record_outcomes(log)

        # Initially UNTESTED
        assert ledger.records[0].outcome == BeliefOutcome.UNTESTED

        # Mark as verified
        result = ledger.mark_verified(
            "b1", VerificationMethod.CODE_EXECUTION, BeliefOutcome.CONFIRMED
        )
        assert result is not None
        assert result.outcome == BeliefOutcome.CONFIRMED
        assert result.verified_by == VerificationMethod.CODE_EXECUTION

        # Persisted
        ledger2 = BeliefLedger(path=Path(d) / "ledger.json")
        assert ledger2.records[0].outcome == BeliefOutcome.CONFIRMED


def test_mark_verified_not_found():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")
        result = ledger.mark_verified(
            "nonexistent", VerificationMethod.CODE_EXECUTION, BeliefOutcome.CONFIRMED
        )
        assert result is None


# ---------------------------------------------------------------------------
# WP3: Domain classification tests
# ---------------------------------------------------------------------------


def test_classify_domain_infrastructure():
    assert classify_domain("Deploy the docker container to kubernetes") == "infrastructure"


def test_classify_domain_database():
    assert classify_domain("Optimize the SQL query for postgres") == "database"


def test_classify_domain_api():
    assert classify_domain("Build a REST API endpoint for users") == "api"


def test_classify_domain_frontend():
    assert classify_domain("Fix the React component CSS styling") == "frontend"


def test_classify_domain_general():
    assert classify_domain("Do something vague") == "general"


def test_domain_recorded_on_beliefs():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        log = ConversationLog(task="Deploy docker to kubernetes cluster")
        log.add(
            role="thinker",
            entry_type="handoff",
            content=_make_handoff(_make_belief("b1")),
        )
        log.add(
            role="executor",
            entry_type="feedback",
            content=ExecutorFeedback(observations=["Done"], execution_result="OK"),
        )
        log.converged = True
        ledger.record_outcomes(log)

        assert ledger.records[0].domain == "infrastructure"


# ---------------------------------------------------------------------------
# WP3: Temporal decay tests
# ---------------------------------------------------------------------------


def test_temporal_decay_weights_recent_more():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        # Add a confirmed belief from 60 days ago
        old_record = BeliefRecord(
            belief_id="old",
            claim="Old claim",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.CONFIRMED,
            task_summary="Old task",
            timestamp=datetime.now(timezone.utc) - timedelta(days=60),
        )
        # Add a confirmed belief from today
        new_record = BeliefRecord(
            belief_id="new",
            claim="New claim",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.CONFIRMED,
            task_summary="New task",
            timestamp=datetime.now(timezone.utc),
        )
        ledger._records = [old_record, new_record]

        # With decay (half_life=30 days), old record gets weight ~0.25, new gets ~1.0
        report_decay = ledger.calibration_report(decay_half_life_days=30.0)
        high_decay = next(s for s in report_decay if s.level == ConfidenceLevel.HIGH)

        # Without decay, both count equally
        report_no_decay = ledger.calibration_report(decay_half_life_days=0)
        high_no_decay = next(s for s in report_no_decay if s.level == ConfidenceLevel.HIGH)

        # Decayed total should be less than non-decayed
        assert high_decay.total < high_no_decay.total
        assert high_no_decay.total == 2.0


def test_per_domain_filtering():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")

        r1 = BeliefRecord(
            belief_id="b1",
            claim="DB claim",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.CONFIRMED,
            domain="database",
        )
        r2 = BeliefRecord(
            belief_id="b2",
            claim="API claim",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.FALSIFIED,
            domain="api",
        )
        ledger._records = [r1, r2]

        # Filter by database domain
        report = ledger.calibration_report(decay_half_life_days=0, domain="database")
        high = next(s for s in report if s.level == ConfidenceLevel.HIGH)
        assert high.total == 1.0
        assert high.confirmed == 1.0
        assert high.falsified == 0.0
