"""Integration tests for the multi-model panel."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import FakeProvider
from epistemic_agents.panel import ModelPanel
from epistemic_agents.schema import (
    AgreementPoint,
    Belief,
    ConfidenceLevel,
    PanelSynthesis,
    ProviderPosition,
)


def test_debate_multi_round():
    """Test that debate runs multiple rounds with FakeProviders."""
    providers = [
        FakeProvider(name="provider-a", response="Analysis from A about Redis caching"),
        FakeProvider(name="provider-b", response="Analysis from B about Redis caching"),
    ]
    panel = ModelPanel(providers, extract_beliefs=False)

    rounds = panel.debate("Test task", rounds=2)
    assert len(rounds) == 2
    assert len(rounds[0]) == 2  # Both providers respond in round 1
    assert len(rounds[1]) == 2  # Both providers respond in round 2

    # Verify both providers were called
    assert len(providers[0].calls) == 2
    assert len(providers[1].calls) == 2


def test_refute_with_synthesis():
    """Test refutation round with a synthesis."""
    providers = [
        FakeProvider(name="provider-a", response="Refutation from A"),
        FakeProvider(name="provider-b", response="Refutation from B"),
    ]
    panel = ModelPanel(providers, extract_beliefs=False)

    rounds = [[
        ProviderPosition(
            provider_name="provider-a",
            model_id="a-v1",
            beliefs=[],
            raw_analysis="Analysis A",
        ),
    ]]
    synthesis = PanelSynthesis(
        task="Test",
        provider_positions=[],
        agreements=[
            AgreementPoint(
                claim="Use Redis",
                supporting_providers=["provider-a"],
                combined_confidence=ConfidenceLevel.HIGH,
            ),
        ],
        synthesized_strategy="Use Redis for caching",
        meta_confidence="High",
    )

    refutations = panel.refute("Test task", rounds, synthesis)
    assert len(refutations) == 2


def test_belief_extraction_integration():
    """Test that belief extraction populates beliefs when enabled."""
    providers = [
        FakeProvider(name="test-provider", response="Redis is great for caching. High confidence."),
    ]

    mock_beliefs = [
        Belief(
            id="test-provider-b1",
            claim="Redis is great for caching",
            confidence=ConfidenceLevel.HIGH,
            justification="test",
        ),
    ]

    with patch("epistemic_agents.belief_extractor.client") as mock_client:
        from epistemic_agents.belief_extractor import ExtractedBeliefs
        mock_client.structured_request.return_value = ExtractedBeliefs(beliefs=mock_beliefs)

        panel = ModelPanel(providers, extract_beliefs=True, extraction_model="haiku")
        positions = panel.run("Test task")

        assert len(positions) == 1
        assert len(positions[0].beliefs) == 1
        assert positions[0].beliefs[0].id == "test-provider-b1"


def test_belief_extraction_graceful_degradation():
    """Test that extraction failure doesn't break the panel."""
    providers = [
        FakeProvider(name="test-provider", response="Some analysis"),
    ]

    with patch("epistemic_agents.belief_extractor.client") as mock_client:
        mock_client.structured_request.side_effect = RuntimeError("Extraction failed")

        panel = ModelPanel(providers, extract_beliefs=True)
        positions = panel.run("Test task")

        assert len(positions) == 1
        assert positions[0].beliefs == []  # Graceful fallback


def test_targeted_prompting_with_beliefs():
    """Test that targeted prompting is used when beliefs are populated."""
    providers = [
        FakeProvider(name="provider-a", response="Analysis A with beliefs"),
        FakeProvider(name="provider-b", response="Analysis B with beliefs"),
    ]

    mock_beliefs_a = [
        Belief(id="a-b1", claim="Use Redis", confidence=ConfidenceLevel.HIGH, justification="test"),
    ]
    mock_beliefs_b = [
        Belief(id="b-b1", claim="Use Memcached", confidence=ConfidenceLevel.MODERATE, justification="test"),
    ]

    call_count = [0]

    def mock_extract(raw, name, model="haiku"):
        call_count[0] += 1
        if name == "provider-a":
            return mock_beliefs_a
        return mock_beliefs_b

    with patch("epistemic_agents.panel.ModelPanel._query_provider") as mock_query:
        # Simulate Round 1 with populated beliefs
        mock_query.side_effect = lambda p, t, s, eb=False, em="haiku": ProviderPosition(
            provider_name=p.name,
            model_id=p.model_id,
            beliefs=mock_beliefs_a if p.name == "provider-a" else mock_beliefs_b,
            raw_analysis=f"Analysis from {p.name}",
        )

        panel = ModelPanel(providers, extract_beliefs=True)
        rounds = panel.debate("Test task", rounds=2)

        # Should have 2 rounds
        assert len(rounds) == 2
