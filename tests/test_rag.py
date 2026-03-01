"""Tests for RAG — retrieval-augmented context from past sessions."""

import tempfile
from pathlib import Path

from epistemic_agents.feedback import FeedbackLog, SessionFeedback
from epistemic_agents.ledger import BeliefLedger, BeliefOutcome, BeliefRecord
from epistemic_agents.schema import ConfidenceLevel
from epistemic_agents.tracker import UsageTracker, ProviderContribution
from epistemic_agents.rag import (
    _tokenize,
    _jaccard_similarity,
    build_rag_context,
)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_basic():
    tokens = _tokenize("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_removes_stopwords():
    tokens = _tokenize("the quick brown fox and the lazy dog")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens


def test_tokenize_strips_punctuation():
    tokens = _tokenize("hello, world! (test)")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_tokenize_removes_single_chars():
    tokens = _tokenize("a b c hello")
    assert "hello" in tokens
    # Single chars after stopword removal and length filter
    assert "b" not in tokens
    assert "c" not in tokens


def test_tokenize_empty():
    tokens = _tokenize("")
    assert tokens == set()


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------


def test_jaccard_identical():
    a = {"hello", "world"}
    assert _jaccard_similarity(a, a) == 1.0


def test_jaccard_disjoint():
    a = {"hello", "world"}
    b = {"foo", "bar"}
    assert _jaccard_similarity(a, b) == 0.0


def test_jaccard_partial():
    a = {"hello", "world", "test"}
    b = {"hello", "world", "other"}
    sim = _jaccard_similarity(a, b)
    # intersection = 2, union = 4 → 0.5
    assert abs(sim - 0.5) < 0.01


def test_jaccard_empty_sets():
    assert _jaccard_similarity(set(), set()) == 0.0
    assert _jaccard_similarity({"hello"}, set()) == 0.0


# ---------------------------------------------------------------------------
# build_rag_context — empty stores
# ---------------------------------------------------------------------------


def test_build_rag_context_all_empty():
    """With no data, should return empty string."""
    result = build_rag_context("test task")
    assert result == ""


def test_build_rag_context_empty_ledger():
    """Empty ledger should not contribute anything."""
    ledger = BeliefLedger(path=Path(tempfile.mktemp(suffix=".json")))
    result = build_rag_context("test task", ledger=ledger)
    assert result == ""


# ---------------------------------------------------------------------------
# Falsified beliefs
# ---------------------------------------------------------------------------


def _make_ledger_with_records(records: list[BeliefRecord]) -> BeliefLedger:
    ledger = BeliefLedger(path=Path(tempfile.mktemp(suffix=".json")))
    ledger._records = records
    return ledger


def test_falsified_beliefs_included():
    """Falsified beliefs relevant to the task should appear in context."""
    records = [
        BeliefRecord(
            belief_id="b1",
            claim="Python startup performance is fast enough for CLI tools",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.FALSIFIED,
            failure_reason="Measured 500ms cold start",
            task_summary="Build a Python CLI tool for quick queries",
        ),
    ]
    ledger = _make_ledger_with_records(records)
    result = build_rag_context("Build a fast Python CLI application", ledger=ledger)
    assert "FALSIFIED" in result
    assert "Python startup performance" in result


def test_revised_beliefs_included():
    """Revised beliefs relevant to the task should appear in context."""
    records = [
        BeliefRecord(
            belief_id="b2",
            claim="Docker containers always improve deployment reliability",
            confidence=ConfidenceLevel.MODERATE,
            outcome=BeliefOutcome.REVISED,
            failure_reason="Revised during loop",
            task_summary="Deploy microservices with Docker",
        ),
    ]
    ledger = _make_ledger_with_records(records)
    result = build_rag_context("Deploy services using Docker containers", ledger=ledger)
    assert "REVISED" in result
    assert "Docker" in result


def test_irrelevant_beliefs_excluded():
    """Beliefs not relevant to the task should not appear."""
    records = [
        BeliefRecord(
            belief_id="b3",
            claim="Sourdough bread needs 24 hours to proof",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.FALSIFIED,
            failure_reason="Only needs 12 hours",
            task_summary="Baking sourdough bread recipe",
        ),
    ]
    ledger = _make_ledger_with_records(records)
    result = build_rag_context("Deploy a Python web application", ledger=ledger)
    # Should not include sourdough beliefs for a Python web task
    assert "sourdough" not in result.lower()


# ---------------------------------------------------------------------------
# Feedback patterns
# ---------------------------------------------------------------------------


def _make_feedback_log(entries: list[SessionFeedback]) -> FeedbackLog:
    log = FeedbackLog(path=Path(tempfile.mktemp(suffix=".json")))
    log._entries = entries
    return log


def test_feedback_patterns_included():
    """Feedback with relevant comments should appear in context."""
    entries = [
        SessionFeedback(
            task_summary="Analyze Python performance bottlenecks",
            useful="yes",
            comment="",
            tier_used="deep",
        ),
        SessionFeedback(
            task_summary="Python optimization strategies",
            useful="partially",
            comment="Should have included profiling data",
            tier_used="standard",
        ),
    ]
    log = _make_feedback_log(entries)
    result = build_rag_context("Optimize Python application performance", feedback_log=log)
    assert "FEEDBACK" in result
    assert "profiling" in result.lower()


def test_feedback_tier_stats():
    """Tier effectiveness stats should appear."""
    entries = [
        SessionFeedback(task_summary="Task A", useful="yes", tier_used="deep"),
        SessionFeedback(task_summary="Task B", useful="yes", tier_used="deep"),
        SessionFeedback(task_summary="Task C", useful="no", tier_used="quick"),
    ]
    log = _make_feedback_log(entries)
    result = build_rag_context("Any task", feedback_log=log)
    assert "deep:" in result
    assert "quick:" in result


# ---------------------------------------------------------------------------
# Provider track records
# ---------------------------------------------------------------------------


def test_provider_track_records():
    """Provider contribution data should appear in context."""
    path = Path(tempfile.mktemp(suffix=".json"))
    tracker = UsageTracker(path=path)
    tracker.record_usage("claude", "opus", input_tokens=1000, output_tokens=500)
    tracker.finalize_session(
        task="Test task",
        tier="deep",
        model_count=2,
        contributions=[
            ProviderContribution(
                provider_name="claude",
                model_id="opus",
                unique_insights=5,
                tensions_involved=2,
            ),
            ProviderContribution(
                provider_name="gemini",
                model_id="flash",
                unique_insights=3,
                tensions_involved=1,
            ),
        ],
    )

    result = build_rag_context("Any task", tracker=tracker)
    assert "PROVIDER TRACK RECORDS" in result
    assert "claude" in result
    assert "gemini" in result


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation():
    """Context should be truncated to max_length."""
    records = [
        BeliefRecord(
            belief_id=f"b{i}",
            claim=f"Belief claim number {i} about software architecture patterns " * 5,
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.FALSIFIED,
            failure_reason="Detailed failure reason " * 10,
            task_summary="Software architecture design patterns",
        )
        for i in range(20)
    ]
    ledger = _make_ledger_with_records(records)
    result = build_rag_context(
        "Design software architecture",
        ledger=ledger,
        max_length=500,
    )
    assert len(result) <= 500 + 50  # Allow small margin for line break
    assert "[truncated]" in result


# ---------------------------------------------------------------------------
# Calibration section
# ---------------------------------------------------------------------------


def test_calibration_included():
    """When ledger has enough data, calibration section should appear."""
    records = [
        BeliefRecord(
            belief_id=f"b{i}",
            claim=f"Claim {i}",
            confidence=ConfidenceLevel.HIGH,
            outcome=BeliefOutcome.CONFIRMED,
            task_summary="Some task",
        )
        for i in range(6)
    ]
    ledger = _make_ledger_with_records(records)
    result = build_rag_context("Any task", ledger=ledger)
    assert "CALIBRATION" in result
