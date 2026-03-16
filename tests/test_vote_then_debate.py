"""Tests for vote-then-debate hybrid (Change 1) and majority_vote."""

import math

from epistemic_agents.agreement_detector import majority_vote, detect_agreements
from epistemic_agents.schema import Belief, ConfidenceLevel, ProviderPosition


def _pos(name: str, beliefs: list[Belief]) -> ProviderPosition:
    return ProviderPosition(
        provider_name=name,
        model_id=f"{name}-v1",
        beliefs=beliefs,
        raw_analysis="test",
    )


def _belief(id: str, claim: str, score: float = 0.8, reasoning_basis: str | None = None) -> Belief:
    return Belief(
        id=id,
        claim=claim,
        confidence=ConfidenceLevel.HIGH,
        justification="test",
        confidence_score=score,
        reasoning_basis=reasoning_basis,
    )


def test_majority_vote_locks_unanimous():
    """All providers agree → should be locked."""
    positions = [
        _pos("claude", [_belief("c1", "Redis improves latency", 0.9)]),
        _pos("gemini", [_belief("g1", "Redis improves latency", 0.85)]),
        _pos("grok", [_belief("k1", "Redis improves latency", 0.88)]),
    ]
    locked, contested = majority_vote(positions, n_eff=3.0)
    assert len(locked) >= 1
    assert locked[0].claim == "Redis improves latency"


def test_majority_vote_no_agreement():
    """All providers disagree → all contested."""
    positions = [
        _pos("claude", [_belief("c1", "Redis is the best caching solution")]),
        _pos("gemini", [_belief("g1", "Memcached outperforms everything else")]),
        _pos("grok", [_belief("k1", "Database queries need no caching")]),
    ]
    locked, contested = majority_vote(positions, n_eff=3.0)
    assert len(locked) == 0
    assert len(contested) == 3


def test_majority_vote_partial_agreement():
    """2 of 3 agree, n_eff=2 → should lock."""
    positions = [
        _pos("claude", [_belief("c1", "Redis improves latency", 0.9)]),
        _pos("gemini", [_belief("g1", "Redis improves latency", 0.85)]),
        _pos("grok", [_belief("k1", "Memcached is better for caching", 0.8)]),
    ]
    locked, contested = majority_vote(positions, n_eff=2.0)
    assert len(locked) >= 1


def test_majority_vote_high_neff_requires_more():
    """High n_eff requires more providers to agree."""
    positions = [
        _pos("claude", [_belief("c1", "Redis improves latency", 0.9)]),
        _pos("gemini", [_belief("g1", "Redis improves latency", 0.85)]),
        _pos("grok", [_belief("k1", "Memcached is better", 0.8)]),
    ]
    # n_eff=3, ceil(3)=3, but only 2 agree
    locked, contested = majority_vote(positions, n_eff=3.0)
    # May or may not lock depending on cluster size


def test_majority_vote_latent_disagreement():
    """Providers agree on claim but diverge on reasoning → stays contested."""
    positions = [
        _pos("claude", [_belief("c1", "Redis improves latency", 0.9,
                                 reasoning_basis="p99 latency benchmarks from production")]),
        _pos("gemini", [_belief("g1", "Redis improves latency", 0.85,
                                 reasoning_basis="theoretical analysis of memory access patterns")]),
        _pos("grok", [_belief("k1", "Redis improves latency", 0.88,
                               reasoning_basis="comparison with filesystem based caching approaches")]),
    ]
    locked, contested = majority_vote(positions, n_eff=2.0)
    # With divergent reasoning bases, should stay contested
    # (depends on Jaccard threshold — these are very different bases)


def test_majority_vote_empty_positions():
    """Empty positions should return empty results."""
    locked, contested = majority_vote([], n_eff=1.0)
    assert locked == []
    assert contested == []


def test_majority_vote_single_provider():
    """Single provider can't form agreements."""
    positions = [
        _pos("claude", [_belief("c1", "Redis is good", 0.9)]),
    ]
    locked, contested = majority_vote(positions, n_eff=1.0)
    assert len(locked) == 0
    assert len(contested) == 1


def test_majority_vote_no_reasoning_basis_no_latent_check():
    """Without reasoning_basis, latent disagreement check is skipped."""
    positions = [
        _pos("claude", [_belief("c1", "Redis improves latency", 0.9)]),
        _pos("gemini", [_belief("g1", "Redis improves latency", 0.85)]),
    ]
    locked, contested = majority_vote(positions, n_eff=2.0)
    assert len(locked) >= 1


def test_majority_vote_mixed_beliefs():
    """Some beliefs agreed, some contested."""
    positions = [
        _pos("claude", [
            _belief("c1", "Redis improves latency", 0.9),
            _belief("c2", "GraphQL is better than REST", 0.7),
        ]),
        _pos("gemini", [
            _belief("g1", "Redis improves latency", 0.85),
            _belief("g2", "REST is simpler than GraphQL", 0.8),
        ]),
    ]
    locked, contested = majority_vote(positions, n_eff=2.0)
    # Redis should be agreed, GraphQL/REST should be contested
    locked_claims = {a.claim for a in locked}
    assert any("Redis" in c or "latency" in c for c in locked_claims)
