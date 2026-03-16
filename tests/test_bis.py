"""Tests for Belief Importance Scoring and cascade falsification."""

from epistemic_agents.bis import cascade_falsify, detect_cycles, importance_scores, rank_beliefs, topological_sort
from epistemic_agents.schema import Belief, ConfidenceLevel


def _make_beliefs() -> list[Belief]:
    """Create a dependency chain: b1 <- b2 <- b3, b1 <- b4."""
    return [
        Belief(
            id="b1",
            claim="Foundation belief",
            confidence=ConfidenceLevel.HIGH,
            justification="Root",
            falsification_conditions=["If X fails"],
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
    # b1 has 3 transitive dependents and HIGH confidence
    assert scores["b1"] > scores["b2"]
    assert scores["b1"] > scores["b3"]
    assert scores["b1"] > scores["b4"]


def test_importance_scores_chain():
    beliefs = _make_beliefs()
    scores = importance_scores(beliefs)
    # b2 has 1 transitive dependent (b3), b3 has none
    assert scores["b2"] > scores["b3"]


def test_importance_scores_leaf_baseline():
    beliefs = _make_beliefs()
    scores = importance_scores(beliefs)
    # b3 has no dependents, LOW confidence (0.4 + 0.5 = 0.9), no falsification
    # score = 1.0 * 0.9 * 1.0 = 0.9
    assert scores["b3"] == 1.0 * (0.4 + 0.5) * 1.0


def test_rank_beliefs_order():
    beliefs = _make_beliefs()
    ranked = rank_beliefs(beliefs)
    # b1 should be first (highest importance)
    assert ranked[0][0].id == "b1"


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
    # b1 HIGH: 1.0 * (0.9+0.5) * 1.0 = 1.4
    assert scores["b1"] == 1.0 * 1.4 * 1.0
    # b2 LOW: 1.0 * (0.4+0.5) * 1.0 = 0.9
    assert scores["b2"] == 1.0 * 0.9 * 1.0

    affected = cascade_falsify(beliefs, "b1")
    assert affected == []


# ---------------------------------------------------------------------------
# WP2: Cycle detection tests
# ---------------------------------------------------------------------------


def test_detect_cycles_no_cycles():
    beliefs = _make_beliefs()
    cycles = detect_cycles(beliefs)
    assert cycles == []


def test_detect_cycles_simple_cycle():
    beliefs = [
        Belief(id="a", claim="A", confidence=ConfidenceLevel.HIGH, justification="X", depends_on=["b"]),
        Belief(id="b", claim="B", confidence=ConfidenceLevel.HIGH, justification="Y", depends_on=["a"]),
    ]
    cycles = detect_cycles(beliefs)
    assert len(cycles) >= 1
    # The cycle should contain both a and b
    all_nodes = set()
    for c in cycles:
        all_nodes.update(c)
    assert "a" in all_nodes
    assert "b" in all_nodes


def test_detect_cycles_self_reference():
    beliefs = [
        Belief(id="a", claim="A", confidence=ConfidenceLevel.HIGH, justification="X", depends_on=["a"]),
    ]
    cycles = detect_cycles(beliefs)
    assert len(cycles) >= 1


def test_importance_scores_with_cycle_no_infinite_loop():
    """BIS should handle cycles without infinite recursion."""
    beliefs = [
        Belief(id="a", claim="A", confidence=ConfidenceLevel.HIGH, justification="X", depends_on=["b"]),
        Belief(id="b", claim="B", confidence=ConfidenceLevel.HIGH, justification="Y", depends_on=["a"]),
        Belief(id="c", claim="C", confidence=ConfidenceLevel.LOW, justification="Z", depends_on=["a"]),
    ]
    # Should not hang or raise
    scores = importance_scores(beliefs)
    assert "a" in scores
    assert "b" in scores
    assert "c" in scores


# ---------------------------------------------------------------------------
# WP2: Weighted scoring & testability boost tests
# ---------------------------------------------------------------------------


def test_testability_boost():
    b_no_fc = Belief(id="a", claim="A", confidence=ConfidenceLevel.HIGH, justification="X")
    b_many_fc = Belief(
        id="b",
        claim="B",
        confidence=ConfidenceLevel.HIGH,
        justification="Y",
        falsification_conditions=["c1", "c2", "c3"],
    )
    scores_no = importance_scores([b_no_fc])
    scores_many = importance_scores([b_many_fc])
    # b with 3 falsification conditions should score higher than b without
    assert scores_many["b"] > scores_no["a"]


def test_testability_boost_capped_at_5():
    b_5 = Belief(
        id="a",
        claim="A",
        confidence=ConfidenceLevel.HIGH,
        justification="X",
        falsification_conditions=["c1", "c2", "c3", "c4", "c5"],
    )
    b_10 = Belief(
        id="b",
        claim="B",
        confidence=ConfidenceLevel.HIGH,
        justification="Y",
        falsification_conditions=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"],
    )
    scores_5 = importance_scores([b_5])
    scores_10 = importance_scores([b_10])
    # Capped at 5, so scores should be equal
    assert scores_5["a"] == scores_10["b"]


def test_weighted_scoring_with_numeric_confidence():
    b_parent = Belief(
        id="parent",
        claim="Parent",
        confidence=ConfidenceLevel.HIGH,
        justification="X",
    )
    b_child_high = Belief(
        id="child",
        claim="Child",
        confidence=ConfidenceLevel.HIGH,
        justification="Y",
        depends_on=["parent"],
        confidence_score=0.95,
    )
    scores = importance_scores([b_parent, b_child_high])
    # Parent gets weighted by child's effective score (0.95)
    assert scores["parent"] > scores["child"]


# ---------------------------------------------------------------------------
# Topological sort tests
# ---------------------------------------------------------------------------


def test_topological_sort_linear_chain():
    """Linear chain: b1 <- b2 <- b3 should sort as [b1, b2, b3]."""
    beliefs = [
        Belief(id="b3", claim="C", confidence=ConfidenceLevel.LOW, justification="Z", depends_on=["b2"]),
        Belief(id="b1", claim="A", confidence=ConfidenceLevel.HIGH, justification="X"),
        Belief(id="b2", claim="B", confidence=ConfidenceLevel.MODERATE, justification="Y", depends_on=["b1"]),
    ]
    sorted_beliefs = topological_sort(beliefs)
    ids = [b.id for b in sorted_beliefs]
    assert ids.index("b1") < ids.index("b2") < ids.index("b3")


def test_topological_sort_diamond_dag():
    """Diamond: b1 <- b2, b1 <- b3, b2 <- b4, b3 <- b4."""
    beliefs = [
        Belief(id="b4", claim="D", confidence=ConfidenceLevel.LOW, justification="Z", depends_on=["b2", "b3"]),
        Belief(id="b2", claim="B", confidence=ConfidenceLevel.MODERATE, justification="Y", depends_on=["b1"]),
        Belief(id="b3", claim="C", confidence=ConfidenceLevel.MODERATE, justification="Y", depends_on=["b1"]),
        Belief(id="b1", claim="A", confidence=ConfidenceLevel.HIGH, justification="X"),
    ]
    sorted_beliefs = topological_sort(beliefs)
    ids = [b.id for b in sorted_beliefs]
    assert ids[0] == "b1"
    assert ids[-1] == "b4"
    assert ids.index("b2") < ids.index("b4")
    assert ids.index("b3") < ids.index("b4")


def test_topological_sort_cycles_graceful():
    """Cycles should be handled gracefully (broken, all beliefs still present)."""
    beliefs = [
        Belief(id="a", claim="A", confidence=ConfidenceLevel.HIGH, justification="X", depends_on=["b"]),
        Belief(id="b", claim="B", confidence=ConfidenceLevel.HIGH, justification="Y", depends_on=["a"]),
        Belief(id="c", claim="C", confidence=ConfidenceLevel.LOW, justification="Z"),
    ]
    sorted_beliefs = topological_sort(beliefs)
    assert len(sorted_beliefs) == 3
    ids = [b.id for b in sorted_beliefs]
    assert set(ids) == {"a", "b", "c"}
    # c has no deps, should come first
    assert ids[0] == "c"


def test_topological_sort_no_deps():
    """All independent beliefs should appear in result (any order)."""
    beliefs = [
        Belief(id="b1", claim="A", confidence=ConfidenceLevel.HIGH, justification="X"),
        Belief(id="b2", claim="B", confidence=ConfidenceLevel.LOW, justification="Y"),
        Belief(id="b3", claim="C", confidence=ConfidenceLevel.MODERATE, justification="Z"),
    ]
    sorted_beliefs = topological_sort(beliefs)
    assert len(sorted_beliefs) == 3


def test_topological_sort_empty():
    """Empty list should return empty."""
    assert topological_sort([]) == []
