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


def test_extremize_identity():
    from epistemic_agents.confidence import extremize
    assert abs(extremize(0.7, d=1.0) - 0.7) < 1e-10


def test_extremize_pushes_toward_extremes():
    from epistemic_agents.confidence import extremize
    p = 0.7
    result = extremize(p, d=2.0)
    assert result > p  # Pushed toward 1


def test_extremize_pushes_low_toward_zero():
    from epistemic_agents.confidence import extremize
    p = 0.3
    result = extremize(p, d=2.0)
    assert result < p  # Pushed toward 0


def test_extremize_symmetric():
    from epistemic_agents.confidence import extremize
    high = extremize(0.8, d=2.0)
    low = extremize(0.2, d=2.0)
    assert abs(high + low - 1.0) < 1e-10


def test_aggregate_extremized_without_neff():
    from epistemic_agents.confidence import aggregate_confidence_extremized
    result = aggregate_confidence_extremized([0.8, 0.8], n_eff=None)
    assert abs(result - 0.8) < 1e-10


def test_aggregate_extremized_with_neff():
    from epistemic_agents.confidence import aggregate_confidence_extremized
    base = aggregate_confidence_extremized([0.8, 0.8], n_eff=1.0)
    ext = aggregate_confidence_extremized([0.8, 0.8], n_eff=5.0)
    # With higher n_eff, should be more extreme (pushed further from 0.5)
    assert ext >= base


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
