"""Tests for adversarial attack graph."""

from epistemic_agents.adversarial_graph import AttackGraph, build_attack_graph
from epistemic_agents.schema import Belief, ConfidenceLevel


def test_build_attack_graph_basic():
    beliefs = [
        Belief(
            id="b1",
            claim="Foundation claim",
            confidence=ConfidenceLevel.HIGH,
            justification="Strong evidence",
            falsification_conditions=["If X fails"],
            key_assumptions=["Assumes Y"],
        ),
        Belief(
            id="b2",
            claim="Dependent claim",
            confidence=ConfidenceLevel.SPECULATIVE,
            justification="Weak guess",
            falsification_conditions=["If A or B"],
            key_assumptions=["Assumes C", "Assumes D"],
            depends_on=["b1"],
        ),
    ]
    graph = build_attack_graph(beliefs)
    assert isinstance(graph, AttackGraph)
    assert len(graph.nodes) == 2
    assert len(graph.recommended_test_order) == 2


def test_attack_graph_prioritizes_vulnerable():
    beliefs = [
        Belief(
            id="strong",
            claim="Very confident claim",
            confidence=ConfidenceLevel.HIGH,
            justification="Rock solid",
        ),
        Belief(
            id="weak",
            claim="Speculative claim",
            confidence=ConfidenceLevel.SPECULATIVE,
            justification="Just a guess",
            falsification_conditions=["Easy to test"],
            key_assumptions=["Big assumption 1", "Big assumption 2"],
        ),
    ]
    graph = build_attack_graph(beliefs)
    # Weak and testable should rank higher in attack priority
    weak_node = next(n for n in graph.nodes if n.belief_id == "weak")
    strong_node = next(n for n in graph.nodes if n.belief_id == "strong")
    assert weak_node.vulnerability > strong_node.vulnerability
    assert weak_node.testable is True
    assert strong_node.testable is False


def test_attack_graph_cascade_impact():
    beliefs = [
        Belief(
            id="root",
            claim="Root claim",
            confidence=ConfidenceLevel.HIGH,
            justification="Base",
            falsification_conditions=["If root fails"],
        ),
        Belief(id="c1", claim="Child 1", confidence=ConfidenceLevel.MODERATE,
               justification="test", depends_on=["root"]),
        Belief(id="c2", claim="Child 2", confidence=ConfidenceLevel.MODERATE,
               justification="test", depends_on=["root"]),
    ]
    graph = build_attack_graph(beliefs)
    root_node = next(n for n in graph.nodes if n.belief_id == "root")
    assert root_node.cascade_impact == 2  # c1 and c2


def test_attack_graph_empty():
    graph = build_attack_graph([])
    assert graph.nodes == []
    assert graph.recommended_test_order == []


def test_attack_graph_single_belief():
    beliefs = [
        Belief(id="b1", claim="Solo", confidence=ConfidenceLevel.MODERATE,
               justification="test", falsification_conditions=["FC1"]),
    ]
    graph = build_attack_graph(beliefs)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].testable is True
    assert graph.nodes[0].cascade_impact == 0
