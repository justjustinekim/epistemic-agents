"""Tests for the tiered orchestrator — routing and classification logic."""

from epistemic_agents.orchestrator import Orchestrator, Tier, _extract_final_handoff
from epistemic_agents.schema import (
    Belief,
    ConfidenceLevel,
    ConversationLog,
    StrategicHandoff,
    ThinkerAmendment,
    AmendmentType,
    Verdict,
)


def test_tier_enum():
    assert Tier.QUICK.value == "quick"
    assert Tier.STANDARD.value == "standard"
    assert Tier.DEEP.value == "deep"


def test_classify_quick():
    orch = Orchestrator()
    tier = orch._classify_task("What is a REST API?")
    assert tier == Tier.QUICK


def test_classify_standard():
    orch = Orchestrator()
    tier = orch._classify_task(
        "Build a user authentication system with JWT tokens and refresh flow"
    )
    assert tier == Tier.STANDARD


def test_classify_deep_by_keywords():
    orch = Orchestrator()
    tier = orch._classify_task(
        "Evaluate the tradeoffs of our current architecture strategy for long-term scalability"
    )
    assert tier == Tier.DEEP


def test_classify_deep_by_length():
    orch = Orchestrator()
    long_task = "Analyze this system. " * 100  # > 1000 chars
    tier = orch._classify_task(long_task)
    assert tier == Tier.DEEP


def test_verdict_schema():
    v = Verdict(
        decision_point="Should we migrate to microservices?",
        recommendation="Not yet. Invest in modular monolith first.",
        confidence=ConfidenceLevel.MODERATE,
        key_risk="If growth accelerates, monolith bottlenecks could block scaling",
        dissent="Some argue early microservice investment prevents costly migration later",
        tier_used="standard",
        cost_tokens=15000,
    )
    assert v.decision_point.endswith("?")
    assert v.confidence == ConfidenceLevel.MODERATE
    assert v.dissent is not None


def test_verdict_minimal():
    v = Verdict(
        decision_point="Use Redis or Memcached?",
        recommendation="Redis. More features, same speed.",
        confidence=ConfidenceLevel.HIGH,
        key_risk="Redis complexity may be overkill for simple KV cache",
    )
    assert v.dissent is None
    assert v.tier_used == "unknown"
    assert v.cost_tokens is None


def test_extract_final_handoff_simple():
    log = ConversationLog(task="test")
    h = StrategicHandoff(
        intent="Test",
        beliefs=[
            Belief(id="b1", claim="X", confidence=ConfidenceLevel.HIGH, justification="Y"),
        ],
        plan_steps=["Step 1"],
    )
    log.add(role="thinker", entry_type="handoff", content=h)
    result = _extract_final_handoff(log)
    assert result is not None
    assert result.intent == "Test"


def test_extract_final_handoff_with_amendment():
    log = ConversationLog(task="test")
    h = StrategicHandoff(
        intent="Original",
        beliefs=[
            Belief(id="b1", claim="X", confidence=ConfidenceLevel.HIGH, justification="Y"),
        ],
        plan_steps=["Step 1"],
    )
    log.add(role="thinker", entry_type="handoff", content=h)

    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        updated_beliefs=[
            Belief(id="b1", claim="Updated X", confidence=ConfidenceLevel.LOW, justification="New Y"),
        ],
        guidance="Updated b1",
    )
    log.add(role="thinker", entry_type="amendment", content=amendment)

    result = _extract_final_handoff(log)
    assert result is not None
    assert result.beliefs[0].claim == "Updated X"
