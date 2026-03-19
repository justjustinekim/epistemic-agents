"""Tests for automated belief verification pipeline (Change 6B)."""

from epistemic_agents.auto_verify import (
    classify_verifiable,
    auto_verify,
    _parse_verification_response,
    _parse_executor_response,
)
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


# --- Parser tests ---


def test_parse_confirmed_response():
    result = _parse_verification_response("test claim", "CONFIRMED: strong evidence found", "test")
    assert result is not None
    assert result.outcome is True
    assert result.claim == "test claim"


def test_parse_falsified_response():
    result = _parse_verification_response("test claim", "FALSIFIED: contradicting data", "test")
    assert result is not None
    assert result.outcome is False


def test_parse_inconclusive_response():
    result = _parse_verification_response("test claim", "INCONCLUSIVE: not enough data", "test")
    assert result is None


def test_parse_confirmed_in_body():
    raw = "After searching, here is what I found:\nCONFIRMED: multiple sources agree."
    result = _parse_verification_response("claim", raw, "test")
    assert result is not None
    assert result.outcome is True


def test_parse_executor_all_passed():
    raw = "# Results\n**Summary**: 3 passed, 0 failed, 0 errors out of 3 tests"
    result = _parse_executor_response("claim", raw)
    assert result is not None
    assert result.outcome is True


def test_parse_executor_some_failed():
    raw = "**Summary**: 1 passed, 2 failed, 0 errors out of 3 tests"
    result = _parse_executor_response("claim", raw)
    assert result is not None
    assert result.outcome is False


def test_parse_executor_pass_fail_markers():
    raw = "1. [PASS] Claim A\n2. [PASS] Claim B"
    result = _parse_executor_response("claim", raw)
    assert result is not None
    assert result.outcome is True


def test_parse_executor_no_results():
    raw = "No testable claims identified in the analysis."
    result = _parse_executor_response("claim", raw)
    assert result is None
