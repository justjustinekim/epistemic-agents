"""Tests for PairwiseTracker and correlation-aware effective panel size."""

import tempfile
from pathlib import Path

from epistemic_agents.prediction_market import PairwiseTracker, PredictionMarket
from epistemic_agents.schema import Belief, ConfidenceLevel, ProviderPosition


def _pos(name: str, claim: str, score: float) -> ProviderPosition:
    return ProviderPosition(
        provider_name=name,
        model_id=f"{name}-v1",
        beliefs=[
            Belief(
                id=f"{name}-b1",
                claim=claim,
                confidence=ConfidenceLevel.HIGH,
                justification="test",
                confidence_score=score,
            )
        ],
        raw_analysis="test",
    )


def test_fully_correlated_neff_approaches_one():
    """When all providers agree on everything, N_eff should approach 1."""
    tracker = PairwiseTracker()
    # All providers have the same belief with same confidence
    for _ in range(5):
        positions = [
            _pos("claude", "Redis improves latency", 0.9),
            _pos("gemini", "Redis improves latency", 0.85),
            _pos("grok", "Redis improves latency", 0.88),
        ]
        tracker.record_round(positions)
    n_eff = tracker.n_eff(3)
    assert n_eff < 1.5  # Should be close to 1


def test_fully_independent_neff_equals_n():
    """When providers never agree, N_eff should equal N."""
    tracker = PairwiseTracker()
    for _ in range(5):
        positions = [
            _pos("claude", "Redis is best for caching", 0.9),
            _pos("gemini", "Memcached outperforms everything else", 0.9),
            _pos("grok", "Database queries need no caching layer", 0.9),
        ]
        tracker.record_round(positions)
    n_eff = tracker.n_eff(3)
    assert n_eff == 3.0  # No correlation → full independence


def test_partial_correlation():
    """Mixed agreement should give N_eff between 1 and N."""
    tracker = PairwiseTracker()
    # Round 1: claude and gemini agree, grok disagrees
    tracker.record_round([
        _pos("claude", "Redis improves latency", 0.9),
        _pos("gemini", "Redis improves latency", 0.85),
        _pos("grok", "Memcached is better for caching", 0.9),
    ])
    n_eff = tracker.n_eff(3)
    assert 1.0 < n_eff < 3.0


def test_empty_tracker():
    """Empty tracker should return N_eff = N (no data → assume independent)."""
    tracker = PairwiseTracker()
    assert tracker.n_eff(5) == 5.0
    assert tracker.avg_correlation() == 0.0
    assert tracker.pairwise_rates() == {}


def test_single_provider():
    """Single provider should have N_eff = 1."""
    tracker = PairwiseTracker()
    assert tracker.n_eff(1) == 1.0


def test_zero_providers():
    """Zero providers should have N_eff = 0."""
    tracker = PairwiseTracker()
    assert tracker.n_eff(0) == 0.0


def test_pairwise_rates_structure():
    """pairwise_rates should return tuple keys with float values."""
    tracker = PairwiseTracker()
    tracker.record_round([
        _pos("claude", "Redis improves latency", 0.9),
        _pos("gemini", "Redis improves latency", 0.85),
    ])
    rates = tracker.pairwise_rates()
    assert len(rates) == 1
    key = list(rates.keys())[0]
    assert isinstance(key, tuple)
    assert len(key) == 2
    assert 0.0 <= list(rates.values())[0] <= 1.0


def test_avg_correlation_range():
    """avg_correlation should be between 0 and 1."""
    tracker = PairwiseTracker()
    tracker.record_round([
        _pos("claude", "Redis improves latency", 0.9),
        _pos("gemini", "Redis improves latency", 0.85),
        _pos("grok", "Memcached is better", 0.9),
    ])
    corr = tracker.avg_correlation()
    assert 0.0 <= corr <= 1.0


def test_serialization_roundtrip():
    """PairwiseTracker should survive serialization/deserialization."""
    tracker = PairwiseTracker()
    tracker.record_round([
        _pos("claude", "Redis improves latency", 0.9),
        _pos("gemini", "Redis improves latency", 0.85),
    ])
    data = tracker.to_dict()
    restored = PairwiseTracker.from_dict(data)
    assert restored.avg_correlation() == tracker.avg_correlation()
    assert restored.pairwise_rates() == tracker.pairwise_rates()


def test_persistence_via_prediction_market():
    """PairwiseTracker should persist through PredictionMarket save/load."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "predictions.json"
        market = PredictionMarket(path=path)
        market.pairwise_tracker.record_round([
            _pos("claude", "Redis improves latency", 0.9),
            _pos("gemini", "Redis improves latency", 0.85),
        ])
        market._save()

        market2 = PredictionMarket(path=path)
        assert market2.pairwise_tracker.avg_correlation() == market.pairwise_tracker.avg_correlation()


def test_no_beliefs_skipped():
    """Positions without beliefs should be skipped."""
    tracker = PairwiseTracker()
    tracker.record_round([
        ProviderPosition(provider_name="claude", model_id="opus", beliefs=[], raw_analysis="test"),
        ProviderPosition(provider_name="gemini", model_id="flash", beliefs=[], raw_analysis="test"),
    ])
    assert tracker.avg_correlation() == 0.0
    assert tracker.pairwise_rates() == {}
