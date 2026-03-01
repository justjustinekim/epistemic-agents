"""Tests for per-call cost tracking and API credit status."""

import json
import threading
from unittest.mock import MagicMock, patch

from epistemic_agents.client import (
    CallCostTracker,
    CallUsage,
    _store_usage,
    _thread_local,
    compute_cost,
    get_last_usage,
)
from epistemic_agents.schema import ConfidenceLevel, Verdict


# ---------------------------------------------------------------------------
# 1. CallUsage defaults
# ---------------------------------------------------------------------------


def test_call_usage_defaults():
    usage = CallUsage()
    assert usage.model == ""
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cost_usd == 0.0


# ---------------------------------------------------------------------------
# 2. CallCostTracker properties
# ---------------------------------------------------------------------------


def test_call_cost_tracker_properties():
    tracker = CallCostTracker()
    tracker.calls.append(CallUsage(model="opus", input_tokens=1000, output_tokens=500, cost_usd=0.0525))
    tracker.calls.append(CallUsage(model="sonnet", input_tokens=2000, output_tokens=1000, cost_usd=0.021))
    assert tracker.total_cost_usd == 0.0525 + 0.021
    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 1500


# ---------------------------------------------------------------------------
# 3. CallCostTracker __str__
# ---------------------------------------------------------------------------


def test_call_cost_tracker_str():
    tracker = CallCostTracker()
    tracker.calls.append(CallUsage(model="opus", input_tokens=1000, output_tokens=500, cost_usd=0.05))
    s = str(tracker)
    assert "$0.0500" in s
    assert "1 calls" in s
    assert "1000 in" in s
    assert "500 out" in s


# ---------------------------------------------------------------------------
# 4. Thread-local isolation
# ---------------------------------------------------------------------------


def test_thread_local_isolation():
    results = {}

    def worker(name, usage):
        _store_usage(usage)
        results[name] = get_last_usage()

    u1 = CallUsage(model="opus", input_tokens=100, cost_usd=0.01)
    u2 = CallUsage(model="sonnet", input_tokens=200, cost_usd=0.02)

    t1 = threading.Thread(target=worker, args=("t1", u1))
    t2 = threading.Thread(target=worker, args=("t2", u2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["t1"].model == "opus"
    assert results["t1"].input_tokens == 100
    assert results["t2"].model == "sonnet"
    assert results["t2"].input_tokens == 200


# ---------------------------------------------------------------------------
# 5. get_last_usage returns None initially (fresh thread)
# ---------------------------------------------------------------------------


def test_get_last_usage_none_initially():
    result = {}

    def worker():
        result["usage"] = get_last_usage()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["usage"] is None


# ---------------------------------------------------------------------------
# 6. structured_request stores usage
# ---------------------------------------------------------------------------


def test_structured_request_stores_usage():
    from epistemic_agents.client import structured_request
    from pydantic import BaseModel

    class SimpleModel(BaseModel):
        answer: str = ""

    envelope = {
        "structured_output": {"answer": "hello"},
        "input_tokens": 500,
        "output_tokens": 200,
    }
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(envelope)
    mock_result.stderr = ""

    with patch("epistemic_agents.client.subprocess.run", return_value=mock_result):
        result = structured_request(
            model="opus",
            system="test",
            user_message="test",
            response_model=SimpleModel,
        )
        assert result.answer == "hello"
        usage = get_last_usage()
        assert usage is not None
        assert usage.model == "opus"
        assert usage.input_tokens == 500
        assert usage.output_tokens == 200
        assert usage.cost_usd > 0


# ---------------------------------------------------------------------------
# 7. structured_request estimates on missing fields
# ---------------------------------------------------------------------------


def test_structured_request_estimates_on_missing_fields():
    from epistemic_agents.client import structured_request
    from pydantic import BaseModel

    class SimpleModel(BaseModel):
        answer: str = ""

    envelope = {
        "structured_output": {"answer": "hello"},
        "result": "some result text here for estimation",
    }
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(envelope)
    mock_result.stderr = ""

    with patch("epistemic_agents.client.subprocess.run", return_value=mock_result):
        structured_request(
            model="opus",
            system="test",
            user_message="test message with some length",
            response_model=SimpleModel,
        )
        usage = get_last_usage()
        assert usage is not None
        # Should have estimated tokens from text lengths
        assert usage.input_tokens > 0
        assert usage.cost_usd > 0


# ---------------------------------------------------------------------------
# 8. OpenAI-compat extracts usage
# ---------------------------------------------------------------------------


def test_openai_extracts_usage():
    from epistemic_agents.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider.gpt(api_key="test-key")

    response_body = json.dumps({
        "choices": [{"message": {"content": "test response"}}],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 80,
        },
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = provider.analyze("test task", "test system")
        assert result == "test response"
        assert provider.last_usage is not None
        assert provider.last_usage.model == "gpt-4o-mini"
        assert provider.last_usage.input_tokens == 150
        assert provider.last_usage.output_tokens == 80
        assert provider.last_usage.cost_usd >= 0


# ---------------------------------------------------------------------------
# 9. Gemini extracts usageMetadata
# ---------------------------------------------------------------------------


def test_gemini_extracts_usage():
    from epistemic_agents.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-key", model_id="gemini-2.0-flash")

    response_body = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": "gemini response"}],
            },
        }],
        "usageMetadata": {
            "promptTokenCount": 300,
            "candidatesTokenCount": 120,
        },
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = provider.analyze("test task", "test system")
        assert result == "gemini response"
        assert provider.last_usage is not None
        assert provider.last_usage.model == "gemini-2.0-flash"
        assert provider.last_usage.input_tokens == 300
        assert provider.last_usage.output_tokens == 120
        assert provider.last_usage.cost_usd == 0.0  # Free tier


# ---------------------------------------------------------------------------
# 10. Thinker cost tracking
# ---------------------------------------------------------------------------


def test_thinker_cost_tracking():
    from epistemic_agents.thinker import Thinker

    thinker = Thinker(model="opus")

    # Mock structured_request and side-effect to store usage
    def mock_structured_req(*args, **kwargs):
        from epistemic_agents.schema import StrategicHandoff, Belief
        _store_usage(CallUsage(model="opus", input_tokens=1000, output_tokens=500, cost_usd=0.0525))
        return StrategicHandoff(
            intent="Test intent",
            beliefs=[
                Belief(
                    id="b1",
                    claim="Test claim",
                    confidence=ConfidenceLevel.HIGH,
                    justification="Test justification",
                ),
            ],
            plan_steps=["Step 1"],
        )

    with patch("epistemic_agents.thinker.structured_request", side_effect=mock_structured_req):
        handoff = thinker.analyze("test task")
        assert handoff.intent == "Test intent"
        assert len(thinker.cost.calls) == 1
        assert thinker.cost.total_cost_usd == 0.0525
        assert thinker.cost.total_input_tokens == 1000


# ---------------------------------------------------------------------------
# 11. Executor cost tracking
# ---------------------------------------------------------------------------


def test_executor_cost_tracking():
    from epistemic_agents.executor import Executor
    from epistemic_agents.schema import StrategicHandoff, Belief

    executor = Executor(model="sonnet")

    handoff = StrategicHandoff(
        intent="Test",
        beliefs=[
            Belief(
                id="b1",
                claim="Test",
                confidence=ConfidenceLevel.HIGH,
                justification="Test",
            ),
        ],
        plan_steps=["Step 1"],
    )

    def mock_structured_req(*args, **kwargs):
        from epistemic_agents.schema import ExecutorFeedback
        _store_usage(CallUsage(model="sonnet", input_tokens=800, output_tokens=300, cost_usd=0.0069))
        return ExecutorFeedback(step_completed=0, observations=["Done"])

    with patch("epistemic_agents.executor.structured_request", side_effect=mock_structured_req):
        feedback = executor.execute(handoff)
        assert feedback.step_completed == 0
        assert len(executor.cost.calls) == 1
        assert executor.cost.total_cost_usd == 0.0069


# ---------------------------------------------------------------------------
# 12. Cost accumulates across calls
# ---------------------------------------------------------------------------


def test_cost_accumulates_across_calls():
    import pytest

    tracker = CallCostTracker()
    for i in range(5):
        tracker.calls.append(CallUsage(
            model="opus",
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
            cost_usd=0.01 * (i + 1),
        ))
    assert len(tracker.calls) == 5
    assert tracker.total_cost_usd == pytest.approx(0.15)
    assert tracker.total_input_tokens == 1500
    assert tracker.total_output_tokens == 750


# ---------------------------------------------------------------------------
# 13. Verdict cost_usd field
# ---------------------------------------------------------------------------


def test_verdict_cost_usd_field():
    verdict = Verdict(
        decision_point="Should we proceed?",
        recommendation="Yes, proceed with caution.",
        confidence=ConfidenceLevel.HIGH,
        key_risk="Might not work",
        tier_used="quick",
        cost_tokens=5000,
        cost_usd=0.1234,
    )
    assert verdict.cost_usd == 0.1234
    assert verdict.cost_tokens == 5000

    # Also check that None is the default
    verdict2 = Verdict(
        decision_point="Test?",
        recommendation="Yes",
        confidence=ConfidenceLevel.MODERATE,
        key_risk="Risk",
    )
    assert verdict2.cost_usd is None
    assert verdict2.cost_tokens is None
