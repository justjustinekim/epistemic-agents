"""Tests for the Code Executor provider."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from epistemic_agents.providers.code_executor import (
    ClaimTest,
    ClaimTestPlan,
    CodeExecutorProvider,
    _SAFETY_PREAMBLE,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction():
    p = CodeExecutorProvider()
    assert p.name == "code-executor"
    assert p.model_id == "code-exec-sonnet"
    assert p.available is True


def test_construction_custom_model():
    p = CodeExecutorProvider(model_id="haiku")
    assert p.model_id == "code-exec-haiku"


# ---------------------------------------------------------------------------
# _execute_test — real execution
# ---------------------------------------------------------------------------


def test_execute_real_math():
    """Execute a real Python math test."""
    p = CodeExecutorProvider()
    test = ClaimTest(
        claim="2 + 2 = 4",
        python_code='import json; print(json.dumps({"passed": 2 + 2 == 4, "detail": f"2+2={2+2}"}))',
        rationale="Basic arithmetic check",
    )
    result = p._execute_test(test)
    assert result["status"] == "passed"
    assert "4" in result["detail"]


def test_execute_failing_test():
    """A test that produces passed=false should return status=failed."""
    p = CodeExecutorProvider()
    test = ClaimTest(
        claim="2 + 2 = 5",
        python_code='import json; print(json.dumps({"passed": 2 + 2 == 5, "detail": "math disagrees"}))',
        rationale="Bad arithmetic",
    )
    result = p._execute_test(test)
    assert result["status"] == "failed"


def test_execute_timeout():
    """Code that runs too long should timeout."""
    p = CodeExecutorProvider()
    test = ClaimTest(
        claim="Infinite loop",
        python_code="import time; time.sleep(60)",
        rationale="Should timeout",
    )
    # Override timeout to be shorter for the test
    import subprocess

    original_run = subprocess.run

    def patched_run(*args, **kwargs):
        kwargs["timeout"] = 2  # 2 seconds instead of 30
        return original_run(*args, **kwargs)

    with patch("epistemic_agents.providers.code_executor.subprocess.run", side_effect=patched_run):
        result = p._execute_test(test)
    assert result["status"] == "error"
    assert "timed out" in result["detail"].lower()


def test_execute_error_handling():
    """Code that raises should return error status."""
    p = CodeExecutorProvider()
    test = ClaimTest(
        claim="Error test",
        python_code="raise ValueError('oops')",
        rationale="Should error",
    )
    result = p._execute_test(test)
    assert result["status"] == "error"
    assert "oops" in result["detail"] or "ValueError" in result["detail"]


def test_execute_non_json_output():
    """Code that prints non-JSON should return error status."""
    p = CodeExecutorProvider()
    test = ClaimTest(
        claim="Non-JSON output",
        python_code='print("this is not json")',
        rationale="Should error on non-JSON",
    )
    result = p._execute_test(test)
    assert result["status"] == "error"
    assert "Non-JSON" in result["detail"]


# ---------------------------------------------------------------------------
# _safe_env
# ---------------------------------------------------------------------------


def test_safe_env_strips_keys():
    """_safe_env should remove API keys and secrets from the environment."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "secret",
        "GOOGLE_API_KEY": "secret",
        "MY_SECRET_TOKEN": "secret",
        "ANTHROPIC_KEY": "secret",
        "PATH": "/usr/bin",
        "HOME": "/Users/test",
    }):
        env = CodeExecutorProvider._safe_env()
        assert "OPENAI_API_KEY" not in env
        assert "GOOGLE_API_KEY" not in env
        assert "MY_SECRET_TOKEN" not in env
        assert "ANTHROPIC_KEY" not in env
        assert env.get("PATH") == "/usr/bin"
        assert env.get("HOME") == "/Users/test"


# ---------------------------------------------------------------------------
# analyze (mocked Claude call)
# ---------------------------------------------------------------------------


def test_analyze_mocked():
    """Test the full analyze flow with a mocked Claude call."""
    p = CodeExecutorProvider()

    plan = ClaimTestPlan(
        tests=[
            ClaimTest(
                claim="1 + 1 = 2",
                python_code='import json; print(json.dumps({"passed": True, "detail": "1+1=2"}))',
                rationale="Basic math",
            ),
        ],
        untestable_claims=["The future is uncertain"],
    )

    with patch("epistemic_agents.providers.code_executor.client.structured_request", return_value=plan):
        result = p.analyze("Check if 1+1=2", "system prompt")

    assert "PASS" in result
    assert "1 + 1 = 2" in result
    assert "Untestable Claims" in result
    assert "The future is uncertain" in result


def test_analyze_no_testable_claims():
    """When no claims are testable, return a descriptive message."""
    p = CodeExecutorProvider()

    plan = ClaimTestPlan(tests=[], untestable_claims=[])

    with patch("epistemic_agents.providers.code_executor.client.structured_request", return_value=plan):
        result = p.analyze("Pure opinion", "system prompt")

    assert "No testable claims" in result


# ---------------------------------------------------------------------------
# ClaimTest / ClaimTestPlan models
# ---------------------------------------------------------------------------


def test_claim_test_model():
    t = ClaimTest(claim="X", python_code="print(1)", rationale="test")
    assert t.claim == "X"


def test_claim_test_plan_model():
    plan = ClaimTestPlan(
        tests=[ClaimTest(claim="X", python_code="print(1)", rationale="test")],
        untestable_claims=["Y"],
    )
    assert len(plan.tests) == 1
    assert len(plan.untestable_claims) == 1
