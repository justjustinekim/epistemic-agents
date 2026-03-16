"""Automated belief verification pipeline."""

from __future__ import annotations

from epistemic_agents.rag import _tokenize
from epistemic_agents.schema import Belief
from epistemic_agents.prediction_market import Resolution


def classify_verifiable(belief: Belief) -> str:
    """Classify a belief as 'factual', 'technical', or 'unverifiable'."""
    claim_lower = belief.claim.lower()
    tokens = _tokenize(belief.claim)

    factual_signals = {"percent", "million", "billion", "study", "research", "data", "statistic", "survey", "report", "according"}
    technical_signals = {"code", "function", "algorithm", "implementation", "api", "library", "framework", "error", "bug", "test", "compile", "runtime"}

    factual_count = len(tokens & factual_signals)
    technical_count = len(tokens & technical_signals)

    if factual_count > technical_count and factual_count > 0:
        return "factual"
    if technical_count > 0:
        return "technical"
    return "unverifiable"


def verify_factual(belief: Belief) -> Resolution | None:
    """Verify factual claims using web search (requires Perplexity provider)."""
    # Stub — requires active provider. Returns None when unavailable.
    return None


def verify_technical(belief: Belief) -> Resolution | None:
    """Verify technical claims using code execution (requires CodeExecutorProvider)."""
    # Stub — requires active provider. Returns None when unavailable.
    return None


def auto_verify(belief: Belief) -> Resolution | None:
    """Route belief to appropriate verifier based on classification."""
    category = classify_verifiable(belief)
    if category == "factual":
        return verify_factual(belief)
    elif category == "technical":
        return verify_technical(belief)
    return None
