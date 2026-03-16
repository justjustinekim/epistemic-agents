"""Tests for automated belief verification pipeline (Change 6B)."""

from epistemic_agents.auto_verify import classify_verifiable, auto_verify
from epistemic_agents.schema import Belief, ConfidenceLevel


def _belief(claim: str) -> Belief:
    return Belief(id="b1", claim=claim, confidence=ConfidenceLevel.HIGH, justification="test")


def test_classify_factual():
    b = _belief("According to the study, 80 percent of users prefer dark mode")
    assert classify_verifiable(b) == "factual"


def test_classify_technical():
    b = _belief("The function implementation has a runtime error in the API code")
    assert classify_verifiable(b) == "technical"


def test_classify_unverifiable():
    b = _belief("This approach is elegant and well-designed")
    assert classify_verifiable(b) == "unverifiable"


def test_classify_mixed_prefers_factual():
    b = _belief("Research data shows the API has performance issues according to the study")
    result = classify_verifiable(b)
    assert result in ("factual", "technical")


def test_auto_verify_returns_none_for_unverifiable():
    b = _belief("The design is aesthetically pleasing")
    assert auto_verify(b) is None


def test_auto_verify_returns_none_for_factual_without_provider():
    b = _belief("According to the research study data shows 50 percent improvement")
    result = auto_verify(b)
    assert result is None  # No provider available


def test_auto_verify_returns_none_for_technical_without_provider():
    b = _belief("The function code has an algorithm bug in the test implementation")
    result = auto_verify(b)
    assert result is None  # No provider available


def test_classify_empty_claim():
    b = _belief("")
    assert classify_verifiable(b) == "unverifiable"
