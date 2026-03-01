"""Tests for programmatic agreement and tension detection."""

from epistemic_agents.agreement_detector import detect_agreements, detect_tensions
from epistemic_agents.schema import (
    Belief,
    ConfidenceLevel,
    ProviderPosition,
)


def _pos(name: str, beliefs: list[Belief]) -> ProviderPosition:
    return ProviderPosition(
        provider_name=name,
        model_id=f"{name}-model",
        beliefs=beliefs,
        raw_analysis="test",
    )


def test_detect_agreements_similar_beliefs():
    positions = [
        _pos("claude", [
            Belief(id="c-b1", claim="Redis caching improves API latency significantly",
                   confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
        _pos("gemini", [
            Belief(id="g-b1", claim="Redis caching improves API latency performance",
                   confidence=ConfidenceLevel.MODERATE, justification="test"),
        ]),
    ]
    agreements = detect_agreements(positions, similarity_threshold=0.4)
    assert len(agreements) >= 1
    assert len(agreements[0].supporting_providers) >= 2
    assert agreements[0].combined_confidence_score is not None
    assert len(agreements[0].source_refs) >= 2


def test_detect_agreements_no_overlap():
    positions = [
        _pos("claude", [
            Belief(id="c-b1", claim="Use Redis for caching",
                   confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
        _pos("gemini", [
            Belief(id="g-b1", claim="Deploy Kubernetes pods to production cluster",
                   confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
    ]
    agreements = detect_agreements(positions, similarity_threshold=0.4)
    assert len(agreements) == 0


def test_detect_agreements_empty_positions():
    assert detect_agreements([]) == []


def test_detect_agreements_no_beliefs():
    positions = [
        _pos("claude", []),
        _pos("gemini", []),
    ]
    assert detect_agreements(positions) == []


def test_detect_agreements_single_provider():
    positions = [
        _pos("claude", [
            Belief(id="c-b1", claim="Use Redis", confidence=ConfidenceLevel.HIGH, justification="test"),
            Belief(id="c-b2", claim="Use Redis cache", confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
    ]
    # Same provider beliefs should not form an agreement
    agreements = detect_agreements(positions, similarity_threshold=0.4)
    assert len(agreements) == 0


def test_detect_tensions_confidence_gap():
    positions = [
        _pos("claude", [
            Belief(id="c-b1", claim="GraphQL is better than REST for this API",
                   confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
        _pos("gemini", [
            Belief(id="g-b1", claim="GraphQL versus REST for the API layer",
                   confidence=ConfidenceLevel.SPECULATIVE, justification="test"),
        ]),
    ]
    tensions = detect_tensions(positions, similarity_threshold=0.3, confidence_gap_threshold=0.3)
    assert len(tensions) >= 1
    assert len(tensions[0].source_refs) == 2


def test_detect_tensions_no_gap():
    positions = [
        _pos("claude", [
            Belief(id="c-b1", claim="Redis caching solution for API",
                   confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
        _pos("gemini", [
            Belief(id="g-b1", claim="Redis caching solution for API",
                   confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
    ]
    # Same confidence → no tension
    tensions = detect_tensions(positions, similarity_threshold=0.3, confidence_gap_threshold=0.3)
    assert len(tensions) == 0


def test_detect_tensions_empty():
    assert detect_tensions([]) == []
