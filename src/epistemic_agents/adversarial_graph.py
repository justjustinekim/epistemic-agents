"""Adversarial attack graph — identify the most impactful beliefs to test."""

from __future__ import annotations

from dataclasses import dataclass, field

from epistemic_agents.bis import importance_scores
from epistemic_agents.schema import Belief


@dataclass
class AttackNode:
    """A belief node in the attack graph with vulnerability analysis."""

    belief_id: str
    claim: str
    importance_score: float
    vulnerability: float
    cascade_impact: float
    testable: bool
    falsification_conditions: list[str] = field(default_factory=list)


@dataclass
class AttackGraph:
    """A prioritized graph of beliefs to test adversarially."""

    nodes: list[AttackNode]
    recommended_test_order: list[str]  # belief IDs sorted by priority


def build_attack_graph(beliefs: list[Belief]) -> AttackGraph:
    """Build an adversarial attack graph from beliefs.

    Priority = importance * vulnerability * testability_multiplier

    Where:
    - importance: BIS score (how many downstream beliefs depend on this)
    - vulnerability: (1 - effective_score) * (1 + 0.2 * assumption_count)
    - testability: 2.0 if has falsification conditions, 0.5 otherwise
    """
    scores = importance_scores(beliefs)
    belief_map = {b.id: b for b in beliefs}

    nodes: list[AttackNode] = []
    for b in beliefs:
        imp = scores.get(b.id, 1.0)
        assumption_count = len(b.key_assumptions)
        vuln = (1 - b.effective_score) * (1 + 0.2 * assumption_count)
        testable = len(b.falsification_conditions) > 0
        testability_mult = 2.0 if testable else 0.5

        # Cascade impact: number of downstream beliefs
        from epistemic_agents.bis import cascade_falsify
        cascade = cascade_falsify(beliefs, b.id)
        cascade_impact = len(cascade)

        priority = imp * vuln * testability_mult

        nodes.append(AttackNode(
            belief_id=b.id,
            claim=b.claim,
            importance_score=imp,
            vulnerability=vuln,
            cascade_impact=cascade_impact,
            testable=testable,
            falsification_conditions=list(b.falsification_conditions),
        ))

    # Sort by priority (highest first)
    nodes.sort(
        key=lambda n: n.importance_score * n.vulnerability * (2.0 if n.testable else 0.5),
        reverse=True,
    )
    recommended_order = [n.belief_id for n in nodes]

    return AttackGraph(nodes=nodes, recommended_test_order=recommended_order)
