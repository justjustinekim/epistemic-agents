"""Tests for prediction market and Brier scoring."""

import tempfile
from pathlib import Path

from epistemic_agents.prediction_market import (
    Prediction,
    PredictionMarket,
    ProviderTrackRecord,
    Resolution,
)


def test_provider_track_record_brier_default():
    record = ProviderTrackRecord(provider_name="claude")
    assert record.brier_score == 0.5
    assert record.weight == max(0.1, 2.0 - 2.0 * 0.5)


def test_provider_track_record_perfect_calibration():
    record = ProviderTrackRecord(
        provider_name="claude",
        total_predictions=10,
        brier_score_sum=0.0,
    )
    assert record.brier_score == 0.0
    assert record.weight == 2.0


def test_provider_track_record_terrible_calibration():
    record = ProviderTrackRecord(
        provider_name="bad",
        total_predictions=10,
        brier_score_sum=10.0,  # Every prediction maximally wrong
    )
    assert record.brier_score == 1.0
    assert record.weight == 0.1  # Floor


def test_place_prediction_and_resolve():
    with tempfile.TemporaryDirectory() as d:
        market = PredictionMarket(path=Path(d) / "predictions.json")

        market.place_prediction(Prediction(
            provider_name="claude",
            claim="Redis will improve latency",
            confidence_score=0.9,
        ))
        market.place_prediction(Prediction(
            provider_name="gemini",
            claim="Redis will improve latency",
            confidence_score=0.3,
        ))

        updated = market.resolve(Resolution(
            claim="Redis will improve latency",
            outcome=True,
        ))
        assert set(updated) == {"claude", "gemini"}

        # Claude should have lower (better) Brier score
        assert market.records["claude"].brier_score < market.records["gemini"].brier_score


def test_get_weight_no_record():
    with tempfile.TemporaryDirectory() as d:
        market = PredictionMarket(path=Path(d) / "predictions.json")
        assert market.get_weight("unknown") == 1.0


def test_prediction_market_persistence():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "predictions.json"

        market1 = PredictionMarket(path=path)
        market1.place_prediction(Prediction(
            provider_name="claude",
            claim="Test claim",
            confidence_score=0.8,
        ))
        market1.resolve(Resolution(claim="Test claim", outcome=True))

        # Load from disk
        market2 = PredictionMarket(path=path)
        assert len(market2._predictions) == 1
        assert "claude" in market2.records


def test_domain_record_defaults():
    from epistemic_agents.prediction_market import DomainRecord
    dr = DomainRecord(domain="ml")
    assert dr.count == 2
    assert dr.brier_score == 0.25  # 0.5 / 2
    assert dr.weight == max(0.1, 2.0 - 2.0 * 0.25)


def test_domain_record_perfect():
    from epistemic_agents.prediction_market import DomainRecord
    dr = DomainRecord(domain="ml", brier_score_sum=0.0, count=10)
    assert dr.brier_score == 0.0
    assert dr.weight == 2.0


def test_domain_record_empty_count():
    from epistemic_agents.prediction_market import DomainRecord
    dr = DomainRecord(domain="ml", brier_score_sum=0.0, count=0)
    assert dr.brier_score == 0.25


def test_get_weight_with_domain():
    from epistemic_agents.prediction_market import DomainRecord
    with tempfile.TemporaryDirectory() as d:
        market = PredictionMarket(path=Path(d) / "predictions.json")
        market._records["claude"] = ProviderTrackRecord(
            provider_name="claude",
            total_predictions=10,
            brier_score_sum=2.5,
            domain_scores={"ml": DomainRecord(domain="ml", brier_score_sum=0.0, count=5)},
        )
        # Domain-specific weight should differ from global
        global_w = market.get_weight("claude")
        domain_w = market.get_weight("claude", domain="ml")
        assert domain_w == 2.0  # Perfect domain score
        assert global_w != domain_w


def test_get_weight_domain_fallback():
    with tempfile.TemporaryDirectory() as d:
        market = PredictionMarket(path=Path(d) / "predictions.json")
        market._records["claude"] = ProviderTrackRecord(
            provider_name="claude",
            total_predictions=10,
            brier_score_sum=2.5,
        )
        # No domain scores, should fall back to global
        w = market.get_weight("claude", domain="unknown_domain")
        assert w == market.get_weight("claude")


def test_resolve_no_matching_predictions():
    with tempfile.TemporaryDirectory() as d:
        market = PredictionMarket(path=Path(d) / "predictions.json")
        market.place_prediction(Prediction(
            provider_name="claude",
            claim="Claim A",
            confidence_score=0.8,
        ))
        # Resolve a different claim
        updated = market.resolve(Resolution(claim="Claim B", outcome=True))
        assert updated == []
