"""Tests for the loop's amendment merge logic."""

from epistemic_agents.schema import (
    AmendmentType,
    Belief,
    ConfidenceLevel,
    DecisionBoundary,
    StrategicHandoff,
    ThinkerAmendment,
)
from epistemic_agents.loop import _apply_amendment


def _make_handoff() -> StrategicHandoff:
    return StrategicHandoff(
        intent="Test intent",
        beliefs=[
            Belief(
                id="b1",
                claim="First belief",
                confidence=ConfidenceLevel.HIGH,
                justification="Reason 1",
            ),
            Belief(
                id="b2",
                claim="Second belief",
                confidence=ConfidenceLevel.MODERATE,
                justification="Reason 2",
            ),
        ],
        plan_steps=["Step 1", "Step 2"],
        decision_boundaries=[
            DecisionBoundary(
                condition="If X happens",
                action="Do Y",
                escalate=True,
            ),
        ],
        open_questions=["Question A", "Question B"],
    )


def test_merge_beliefs_by_id():
    handoff = _make_handoff()
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        updated_beliefs=[
            Belief(
                id="b1",
                claim="Updated first belief",
                confidence=ConfidenceLevel.LOW,
                justification="New reason",
            ),
        ],
        guidance="Updated b1 only",
    )

    result = _apply_amendment(handoff, amendment)

    # b2 should be preserved, b1 should be updated
    ids = {b.id: b for b in result.beliefs}
    assert "b1" in ids
    assert "b2" in ids
    assert ids["b1"].claim == "Updated first belief"
    assert ids["b1"].confidence == ConfidenceLevel.LOW
    assert ids["b2"].claim == "Second belief"  # unchanged


def test_preserve_decision_boundaries():
    handoff = _make_handoff()
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.CLARIFY,
        guidance="Just clarifying, no changes",
    )

    result = _apply_amendment(handoff, amendment)
    assert len(result.decision_boundaries) == 1
    assert result.decision_boundaries[0].condition == "If X happens"


def test_update_decision_boundaries():
    handoff = _make_handoff()
    new_boundary = DecisionBoundary(
        condition="If Z happens",
        action="Do W",
        escalate=False,
    )
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        revised_decision_boundaries=[new_boundary],
        guidance="Changed boundaries",
    )

    result = _apply_amendment(handoff, amendment)
    assert len(result.decision_boundaries) == 1
    assert result.decision_boundaries[0].condition == "If Z happens"


def test_resolve_and_add_open_questions():
    handoff = _make_handoff()
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        resolved_questions=["Question A"],
        new_open_questions=["Question C"],
        guidance="Resolved A, discovered C",
    )

    result = _apply_amendment(handoff, amendment)
    assert "Question A" not in result.open_questions
    assert "Question B" in result.open_questions
    assert "Question C" in result.open_questions


def test_preserve_steps_when_not_revised():
    handoff = _make_handoff()
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.CLARIFY,
        guidance="No step changes",
    )

    result = _apply_amendment(handoff, amendment)
    assert result.plan_steps == ["Step 1", "Step 2"]


def test_update_steps_when_revised():
    handoff = _make_handoff()
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        revised_steps=["New step 1", "New step 2", "New step 3"],
        guidance="New plan",
    )

    result = _apply_amendment(handoff, amendment)
    assert len(result.plan_steps) == 3
    assert result.plan_steps[0] == "New step 1"


def test_add_new_beliefs():
    handoff = _make_handoff()
    amendment = ThinkerAmendment(
        amendment_type=AmendmentType.REVISE,
        updated_beliefs=[
            Belief(
                id="b3",
                claim="New third belief",
                confidence=ConfidenceLevel.SPECULATIVE,
                justification="New discovery",
            ),
        ],
        guidance="Added b3",
    )

    result = _apply_amendment(handoff, amendment)
    ids = {b.id for b in result.beliefs}
    assert ids == {"b1", "b2", "b3"}
