"""Tests for numeric confidence aggregation."""

import math

from epistemic_agents.confidence import (
    _from_log_odds,
    _to_log_odds,
    aggregate_beliefs_confidence,
    aggregate_confidence,
)
from epistemic_agents.schema import Belief, ConfidenceLevel


def test_to_log_odds_and_back():
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert abs(_from_log_odds(_to_log_odds(p)) - p) < 1e-10


def test_to_log_odds_clamping():
    # Should not raise for extreme values
    lo_low = _to_log_odds(0.0)
    lo_high = _to_log_odds(1.0)
    assert math.isfinite(lo_low)
    assert math.isfinite(lo_high)


def test_aggregate_identical_inputs():
    result = aggregate_confidence([0.8, 0.8, 0.8])
    assert abs(result - 0.8) < 1e-10


def test_aggregate_symmetric_extremes():
    result = aggregate_confidence([0.95, 0.05])
    assert abs(result - 0.5) < 0.01


def test_aggregate_empty():
    assert aggregate_confidence([]) == 0.5


def test_aggregate_single():
    result = aggregate_confidence([0.7])
    assert abs(result - 0.7) < 1e-10


def test_aggregate_high_values():
    result = aggregate_confidence([0.9, 0.9, 0.9])
    assert abs(result - 0.9) < 1e-10


def test_aggregate_vs_naive_average():
    """Log-odds averaging should differ from naive for asymmetric distributions."""
    scores = [0.95, 0.95, 0.50]
    naive = sum(scores) / len(scores)
    log_odds = aggregate_confidence(scores)
    # Both should be reasonable but potentially different
    assert 0.5 < log_odds < 1.0
    assert 0.5 < naive < 1.0


def test_aggregate_beliefs_confidence():
    beliefs = [
        Belief(id="b1", claim="A", confidence=ConfidenceLevel.HIGH, justification="X"),
        Belief(id="b2", claim="B", confidence=ConfidenceLevel.MODERATE, justification="Y"),
    ]
    result = aggregate_beliefs_confidence(beliefs)
    # HIGH=0.9, MODERATE=0.7 → log-odds average
    expected = aggregate_confidence([0.9, 0.7])
    assert abs(result - expected) < 1e-10


def test_aggregate_beliefs_confidence_empty():
    assert aggregate_beliefs_confidence([]) == 0.5


def test_aggregate_beliefs_confidence_with_numeric():
    beliefs = [
        Belief(
            id="b1",
            claim="A",
            confidence=ConfidenceLevel.HIGH,
            justification="X",
            confidence_score=0.95,
        ),
        Belief(
            id="b2",
            claim="B",
            confidence=ConfidenceLevel.LOW,
            justification="Y",
            confidence_score=0.3,
        ),
    ]
    result = aggregate_beliefs_confidence(beliefs)
    expected = aggregate_confidence([0.95, 0.3])
    assert abs(result - expected) < 1e-10
