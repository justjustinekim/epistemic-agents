"""Tests for Belief Importance Scoring and cascade falsification."""

from epistemic_agents.bis import cascade_falsify, importance_scores, rank_beliefs
from epistemic_agents.schema import Belief, ConfidenceLevel


def _make_beliefs() -> list[Belief]:
    """Create a dependency chain: b1 <- b2 <- b3, b1 <- b4."""
    return [
        Belief(
            id="b1",
            claim="Foundation belief",
            confidence=ConfidenceLevel.HIGH,
            justification="Root",
        ),
        Belief(
            id="b2",
            claim="Depends on b1",
            confidence=ConfidenceLevel.MODERATE,
            justification="Built on b1",
            depends_on=["b1"],
        ),
        Belief(
            id="b3",
            claim="Depends on b2",
            confidence=ConfidenceLevel.LOW,
            justification="Built on b2",
            depends_on=["b2"],
        ),
        Belief(
            id="b4",
            claim="Also depends on b1",
            confidence=ConfidenceLevel.MODERATE,
            justification="Another b1 dependent",
            depends_on=["b1"],
        ),
    ]


def test_importance_scores_root_highest():
    beliefs = _make_beliefs()
    scores = importance_scores(beliefs)
    # b1 has 3 transitive dependents (b2, b3, b4) and HIGH confidence boost
    assert scores["b1"] > scores["b2"]
    assert scores["b1"] > scores["b3"]
    assert scores["b1"] > scores["b4"]


def test_importance_scores_chain():
    beliefs = _make_beliefs()
    scores = importance_scores(beliefs)
    # b2 has 1 transitive dependent (b3), b3 has none
    assert scores["b2"] > scores["b3"]


def test_importance_scores_leaf_is_one():
    beliefs = _make_beliefs()
    scores = importance_scores(beliefs)
    # b3 has no dependents, so base score is 1.0
    assert scores["b3"] == 1.0


def test_rank_beliefs_order():
    beliefs = _make_beliefs()
    ranked = rank_beliefs(beliefs)
    # b1 should be first (highest importance)
    assert ranked[0][0].id == "b1"
    # b3 and b4 are both leaf nodes with score 1.0, so either can be last
    leaf_ids = {ranked[-1][0].id, ranked[-2][0].id}
    assert leaf_ids == {"b3", "b4"}


def test_cascade_falsify_root():
    beliefs = _make_beliefs()
    affected = cascade_falsify(beliefs, "b1")
    # Falsifying b1 should cascade to b2, b3, b4
    assert set(affected) == {"b2", "b3", "b4"}


def test_cascade_falsify_middle():
    beliefs = _make_beliefs()
    affected = cascade_falsify(beliefs, "b2")
    # Falsifying b2 should cascade to b3 only
    assert affected == ["b3"]


def test_cascade_falsify_leaf():
    beliefs = _make_beliefs()
    affected = cascade_falsify(beliefs, "b3")
    # Falsifying a leaf node affects nothing
    assert affected == []


def test_cascade_falsify_unknown_id():
    beliefs = _make_beliefs()
    affected = cascade_falsify(beliefs, "b999")
    assert affected == []


def test_no_dependencies():
    beliefs = [
        Belief(id="b1", claim="A", confidence=ConfidenceLevel.HIGH, justification="X"),
        Belief(id="b2", claim="B", confidence=ConfidenceLevel.LOW, justification="Y"),
    ]
    scores = importance_scores(beliefs)
    # b1 has HIGH boost: 1.0 * 1.5 = 1.5
    assert scores["b1"] == 1.5
    assert scores["b2"] == 1.0

    affected = cascade_falsify(beliefs, "b1")
    assert affected == []
