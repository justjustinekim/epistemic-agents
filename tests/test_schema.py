"""Tests for the epistemic schema — validate construction, serialization, round-tripping."""

import json

from epistemic_agents.schema import (
    AmendmentType,
    Belief,
    ChallengedBelief,
    ConfidenceLevel,
    ConversationLog,
    DecisionBoundary,
    EscalationType,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)


def test_belief_construction():
    b = Belief(
        id="b1",
        claim="Redis is the best caching solution for this use case",
        confidence=ConfidenceLevel.MODERATE,
        justification="Redis is widely used, supports TTL, and handles 100k+ ops/sec",
        falsification_conditions=[
            "If the infrastructure cannot support Redis (e.g. serverless with no persistent connections)",
            "If the data changes so frequently that any TTL-based cache serves stale data",
        ],
        key_assumptions=[
            "The team can add Redis to their infrastructure",
            "Data update frequency is on the order of minutes, not seconds",
        ],
    )
    assert b.id == "b1"
    assert b.confidence == ConfidenceLevel.MODERATE
    assert len(b.falsification_conditions) == 2


def test_strategic_handoff_serialization():
    handoff = StrategicHandoff(
        intent="Reduce API latency from 200ms to <50ms p95 by adding a caching layer",
        beliefs=[
            Belief(
                id="b1",
                claim="In-memory caching is the right approach",
                confidence=ConfidenceLevel.HIGH,
                justification="10k rps with 50ms target requires sub-millisecond cache hits",
                falsification_conditions=["If the deployment is serverless with no persistent memory"],
                key_assumptions=["Application runs on persistent servers"],
            ),
        ],
        plan_steps=[
            "Analyze access patterns to determine cache key design",
            "Implement Redis-based caching with 60s TTL",
            "Add cache invalidation on write paths",
        ],
        decision_boundaries=[
            DecisionBoundary(
                condition="If cache hit rate is below 70%",
                action="Re-evaluate cache key design",
                escalate=True,
            ),
        ],
        open_questions=["What is the exact data update frequency?"],
    )

    # Serialize and deserialize
    json_str = handoff.model_dump_json(indent=2)
    parsed = json.loads(json_str)
    roundtripped = StrategicHandoff.model_validate(parsed)

    assert roundtripped.intent == handoff.intent
    assert len(roundtripped.beliefs) == 1
    assert roundtripped.beliefs[0].id == "b1"
    assert len(roundtripped.plan_steps) == 3
    assert roundtripped.decision_boundaries[0].escalate is True


def test_executor_feedback_with_escalation():
    feedback = ExecutorFeedback(
        step_completed=0,
        observations=[
            "Analyzed access patterns — 80% of traffic hits top 50 products",
            "Discovered data updates every 30 seconds for flash sales",
        ],
        escalation_type=EscalationType.ASSUMPTION_VIOLATION,
        challenged_beliefs=[
            ChallengedBelief(
                belief_id="b1",
                evidence="Infrastructure is serverless (AWS Lambda) — no persistent in-memory state",
            ),
        ],
        new_evidence=[
            "Database already responds in 15ms — latency is from API gateway hops, not queries",
        ],
        decision_needed=True,
        executor_recommendation="Focus on reducing API gateway hops rather than adding a cache",
    )

    assert feedback.escalation_type == EscalationType.ASSUMPTION_VIOLATION
    assert len(feedback.challenged_beliefs) == 1
    assert feedback.decision_needed is True


def test_executor_feedback_no_escalation():
    feedback = ExecutorFeedback(
        step_completed=2,
        observations=["All steps completed successfully"],
        execution_result="Caching layer deployed, p95 latency now 35ms",
    )

    assert feedback.escalation_type is None
    assert feedback.decision_needed is False
    assert feedback.execution_result is not None


def test_thinker_amendment():
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        updated_beliefs=[
            Belief(
                id="b1-revised",
                claim="The bottleneck is API gateway latency, not database speed",
                confidence=ConfidenceLevel.HIGH,
                justification="Executor confirmed DB responds in 15ms but total latency is 200ms",
                falsification_conditions=["If removing gateway hops doesn't reduce latency"],
                key_assumptions=["API gateway is the primary latency contributor"],
            ),
        ],
        revised_steps=[
            "Audit API gateway configuration for unnecessary hops",
            "Implement edge caching at the CDN level for the top 50 products",
            "Use cache-control headers with short max-age for frequently updated data",
        ],
        guidance=(
            "The original Redis approach was wrong — the problem isn't database speed. "
            "Focus on reducing network hops and using edge caching for the hot path."
        ),
        continue_from_step=0,
    )

    assert amendment.amendment_type == AmendmentType.REVISE
    assert len(amendment.updated_beliefs) == 1
    assert amendment.revised_steps is not None
    assert len(amendment.revised_steps) == 3


def test_conversation_log():
    log = ConversationLog(task="Test task")

    handoff = StrategicHandoff(
        intent="Test intent",
        beliefs=[],
        plan_steps=["Step 1"],
    )
    log.add(role="thinker", entry_type="handoff", content=handoff)

    feedback = ExecutorFeedback(
        observations=["All good"],
        execution_result="Done",
    )
    log.add(role="executor", entry_type="feedback", content=feedback)

    assert len(log.entries) == 2
    assert log.entries[0].role == "thinker"
    assert log.entries[1].role == "executor"

    # Verify JSON serialization
    json_str = log.model_dump_json(indent=2)
    assert "Test intent" in json_str
    assert "All good" in json_str


def test_schema_json_generation():
    """Verify we can generate a clean JSON schema for tool_use."""
    schema = StrategicHandoff.model_json_schema()
    assert "properties" in schema
    assert "intent" in schema["properties"]
    assert "beliefs" in schema["properties"]

    feedback_schema = ExecutorFeedback.model_json_schema()
    assert "escalation_type" in feedback_schema["properties"]
