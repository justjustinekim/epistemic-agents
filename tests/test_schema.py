"""Tests for the epistemic schema — validate construction, serialization, round-tripping."""

import json

from epistemic_agents.schema import (
    AmendmentType,
    Belief,
    ChallengedBelief,
    ConfidenceLevel,
    ConversationLog,
    DecisionBoundary,
    Escalation,
    EscalationSeverity,
    EscalationType,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
    Verdict,
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
    assert b.depends_on == []


def test_belief_with_dependencies():
    b1 = Belief(
        id="b1",
        claim="The API is RESTful",
        confidence=ConfidenceLevel.HIGH,
        justification="Documentation says REST",
    )
    b2 = Belief(
        id="b2",
        claim="We can use standard HTTP caching",
        confidence=ConfidenceLevel.MODERATE,
        justification="REST APIs support cache-control headers",
        depends_on=["b1"],
    )
    assert b2.depends_on == ["b1"]


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
        meta_reasoning="I might be anchoring on Redis without considering alternatives.",
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
    assert roundtripped.meta_reasoning == "I might be anchoring on Redis without considering alternatives."


def test_strategic_handoff_meta_reasoning_default():
    handoff = StrategicHandoff(
        intent="Test",
        beliefs=[],
        plan_steps=[],
    )
    assert handoff.meta_reasoning == ""


def test_escalation_model():
    esc = Escalation(
        type=EscalationType.CONTEXT_SHIFT,
        severity=EscalationSeverity.DEGRADED,
        detail="The API endpoint was deprecated since the thinker analyzed",
    )
    assert esc.type == EscalationType.CONTEXT_SHIFT
    assert esc.severity == EscalationSeverity.DEGRADED


def test_new_escalation_types():
    assert EscalationType.CONTEXT_SHIFT.value == "context_shift"
    assert EscalationType.RESOURCE_OPPORTUNITY.value == "resource_opportunity"
    assert EscalationType.PARTIAL_SUCCESS.value == "partial_success"
    assert EscalationType.CONVERGENCE_FAILURE.value == "convergence_failure"


def test_escalation_severity_levels():
    assert EscalationSeverity.BLOCKING.value == "blocking"
    assert EscalationSeverity.DEGRADED.value == "degraded"
    assert EscalationSeverity.INFORMATIONAL.value == "informational"


def test_executor_feedback_with_escalation():
    feedback = ExecutorFeedback(
        step_completed=0,
        observations=[
            "Analyzed access patterns — 80% of traffic hits top 50 products",
            "Discovered data updates every 30 seconds for flash sales",
        ],
        escalation_type=EscalationType.ASSUMPTION_VIOLATION,
        escalations=[
            Escalation(
                type=EscalationType.ASSUMPTION_VIOLATION,
                severity=EscalationSeverity.BLOCKING,
                detail="Infrastructure is serverless — no persistent memory",
            ),
        ],
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
    assert len(feedback.escalations) == 1
    assert feedback.escalations[0].severity == EscalationSeverity.BLOCKING
    assert len(feedback.challenged_beliefs) == 1
    assert feedback.decision_needed is True


def test_executor_feedback_multiple_escalations():
    feedback = ExecutorFeedback(
        step_completed=1,
        observations=["Partial progress"],
        escalations=[
            Escalation(
                type=EscalationType.PARTIAL_SUCCESS,
                severity=EscalationSeverity.DEGRADED,
                detail="Cache works but hit rate is only 40%",
            ),
            Escalation(
                type=EscalationType.DISCOVERY,
                severity=EscalationSeverity.INFORMATIONAL,
                detail="Found an undocumented batch API that could be more efficient",
            ),
        ],
    )
    assert len(feedback.escalations) == 2


def test_executor_feedback_no_escalation():
    feedback = ExecutorFeedback(
        step_completed=2,
        observations=["All steps completed successfully"],
        execution_result="Caching layer deployed, p95 latency now 35ms",
    )

    assert feedback.escalation_type is None
    assert feedback.escalations == []
    assert feedback.decision_needed is False
    assert feedback.execution_result is not None


def test_executor_feedback_proposed_adjustments():
    feedback = ExecutorFeedback(
        step_completed=1,
        observations=["Found issue"],
        proposed_adjustments="Skip step 3 and go directly to step 4",
    )
    assert feedback.proposed_adjustments is not None


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


def test_thinker_amendment_new_fields():
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        updated_beliefs=[],
        revised_decision_boundaries=[
            DecisionBoundary(
                condition="If latency exceeds 100ms",
                action="Switch to edge caching",
                escalate=False,
            ),
        ],
        resolved_questions=["What is the exact data update frequency?"],
        new_open_questions=["Is the CDN configured for dynamic content?"],
        guidance="Updated boundaries and resolved the frequency question.",
        continue_from_step=1,
    )
    assert amendment.revised_decision_boundaries is not None
    assert len(amendment.revised_decision_boundaries) == 1
    assert amendment.resolved_questions == ["What is the exact data update frequency?"]
    assert len(amendment.new_open_questions) == 1


def test_thinker_amendment_defaults():
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.CLARIFY,
        guidance="Just proceed as planned.",
    )
    assert amendment.revised_decision_boundaries is None
    assert amendment.resolved_questions == []
    assert amendment.new_open_questions == []


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


def test_verdict_construction():
    v = Verdict(
        decision_point="Should we use Redis?",
        recommendation="Yes, use Redis with 60s TTL for the hot path.",
        confidence=ConfidenceLevel.HIGH,
        key_risk="Redis adds infrastructure complexity",
        dissent="Memcached is simpler if we only need KV cache",
        tier_used="deep",
        cost_tokens=50000,
    )
    assert v.confidence == ConfidenceLevel.HIGH
    assert v.dissent is not None
    assert v.cost_tokens == 50000


def test_verdict_defaults():
    v = Verdict(
        decision_point="Use X or Y?",
        recommendation="Use X.",
        confidence=ConfidenceLevel.MODERATE,
        key_risk="X has less community support",
    )
    assert v.dissent is None
    assert v.tier_used == "unknown"
    assert v.cost_tokens is None


def test_verdict_json_schema():
    schema = Verdict.model_json_schema()
    assert "decision_point" in schema["properties"]
    assert "recommendation" in schema["properties"]
    assert "key_risk" in schema["properties"]


def test_schema_json_generation():
    """Verify we can generate a clean JSON schema for tool_use."""
    schema = StrategicHandoff.model_json_schema()
    assert "properties" in schema
    assert "intent" in schema["properties"]
    assert "beliefs" in schema["properties"]
    assert "meta_reasoning" in schema["properties"]

    feedback_schema = ExecutorFeedback.model_json_schema()
    assert "escalation_type" in feedback_schema["properties"]
    assert "escalations" in feedback_schema["properties"]
    assert "proposed_adjustments" in feedback_schema["properties"]
