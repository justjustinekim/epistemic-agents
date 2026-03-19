"""Automated belief verification pipeline.

Routes beliefs to appropriate verifiers based on content classification:
- factual → Perplexity (web search grounded)
- technical → CodeExecutorProvider (generates + runs validation code)
- unverifiable → returns None
"""

from __future__ import annotations

import os
import re

from epistemic_agents.rag import _tokenize
from epistemic_agents.schema import Belief
from epistemic_agents.prediction_market import Resolution


def classify_verifiable(belief: Belief) -> str:
    """Classify a belief as 'factual', 'technical', or 'unverifiable'."""
    tokens = _tokenize(belief.claim)

    factual_signals = {
        "percent", "million", "billion", "study", "research", "data",
        "statistic", "survey", "report", "according", "published",
        "measured", "observed", "documented", "proven", "evidence",
    }
    technical_signals = {
        "code", "function", "algorithm", "implementation", "api",
        "library", "framework", "error", "bug", "test", "compile",
        "runtime", "complexity", "performance", "memory", "latency",
        "query", "database", "sql", "index", "cache",
    }

    factual_count = len(tokens & factual_signals)
    technical_count = len(tokens & technical_signals)

    if factual_count > technical_count and factual_count > 0:
        return "factual"
    if technical_count > 0:
        return "technical"
    return "unverifiable"


def verify_factual(belief: Belief) -> Resolution | None:
    """Verify factual claims using Perplexity (web search grounded).

    Sends the claim to Perplexity's sonar-pro model, which has real-time
    web search capabilities. Parses the response for confirmation/falsification.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        return None

    from epistemic_agents.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider.perplexity(api_key=api_key)

    prompt = (
        "You are a fact-checker. Evaluate the following claim using your web search capabilities.\n\n"
        f"Claim: {belief.claim}\n\n"
        "Respond with EXACTLY one of these formats:\n"
        "CONFIRMED: <brief explanation of supporting evidence>\n"
        "FALSIFIED: <brief explanation of contradicting evidence>\n"
        "INCONCLUSIVE: <brief explanation of why evidence is insufficient>\n\n"
        "Be precise. Only say CONFIRMED if you found strong supporting evidence. "
        "Only say FALSIFIED if you found clear contradicting evidence."
    )

    try:
        raw = provider.analyze(belief.claim, prompt)
        return _parse_verification_response(belief.claim, raw, "perplexity_web_search")
    except Exception:
        return None


def verify_technical(belief: Belief) -> Resolution | None:
    """Verify technical claims by generating and executing validation code.

    Uses CodeExecutorProvider to generate Python test code for the claim,
    then runs it in a sandbox and parses results.
    """
    try:
        from epistemic_agents.providers.code_executor import CodeExecutorProvider

        executor = CodeExecutorProvider()
        if not executor.available:
            return None

        prompt = (
            "Verify this technical claim by writing and running test code:\n\n"
            f"Claim: {belief.claim}\n\n"
            "Focus on empirically testing the specific technical assertion."
        )

        raw = executor.analyze(belief.claim, prompt)
        return _parse_executor_response(belief.claim, raw)
    except Exception:
        return None


def _parse_verification_response(
    claim: str, raw: str, method: str
) -> Resolution | None:
    """Parse a CONFIRMED/FALSIFIED/INCONCLUSIVE response."""
    raw_upper = raw.strip().upper()

    if raw_upper.startswith("CONFIRMED"):
        return Resolution(claim=claim, outcome=True, method=method)
    elif raw_upper.startswith("FALSIFIED"):
        return Resolution(claim=claim, outcome=False, method=method)

    # Try to find the keyword anywhere in the response
    lines = raw.strip().split("\n")
    for line in lines:
        line_upper = line.strip().upper()
        if line_upper.startswith("CONFIRMED"):
            return Resolution(claim=claim, outcome=True, method=method)
        if line_upper.startswith("FALSIFIED"):
            return Resolution(claim=claim, outcome=False, method=method)

    return None  # Inconclusive or unparseable


def _parse_executor_response(claim: str, raw: str) -> Resolution | None:
    """Parse CodeExecutorProvider output for pass/fail verdict."""
    raw_lower = raw.lower()

    # Look for summary line like "2 passed, 0 failed, 0 errors"
    match = re.search(r"(\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+error", raw_lower)
    if match:
        passed = int(match.group(1))
        failed = int(match.group(2))
        errors = int(match.group(3))
        if passed > 0 and failed == 0 and errors == 0:
            return Resolution(claim=claim, outcome=True, method="code_execution")
        if failed > 0:
            return Resolution(claim=claim, outcome=False, method="code_execution")

    # Fallback: check for [PASS] / [FAIL] markers
    pass_count = raw.count("[PASS]")
    fail_count = raw.count("[FAIL]")
    if pass_count > 0 and fail_count == 0:
        return Resolution(claim=claim, outcome=True, method="code_execution")
    if fail_count > 0:
        return Resolution(claim=claim, outcome=False, method="code_execution")

    return None


def auto_verify(belief: Belief) -> Resolution | None:
    """Route belief to appropriate verifier based on classification."""
    category = classify_verifiable(belief)
    if category == "factual":
        return verify_factual(belief)
    elif category == "technical":
        return verify_technical(belief)
    return None
