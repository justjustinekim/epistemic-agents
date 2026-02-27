"""Belief Importance Scoring (BIS) and dependency cascade falsification."""

from __future__ import annotations

from epistemic_agents.schema import Belief, ConfidenceLevel


def importance_scores(beliefs: list[Belief]) -> dict[str, float]:
    """Score each belief by how many downstream beliefs depend on it.

    A belief's importance = 1 + number of transitive dependents.
    Beliefs that are depended upon by many others are more critical to verify.

    Returns a dict mapping belief_id -> importance score.
    """
    # Build adjacency: belief_id -> set of direct dependents
    dependents: dict[str, set[str]] = {b.id: set() for b in beliefs}
    for b in beliefs:
        for dep_id in b.depends_on:
            if dep_id in dependents:
                dependents[dep_id].add(b.id)

    # Count transitive dependents via DFS
    def _count_transitive(bid: str, visited: set[str] | None = None) -> int:
        if visited is None:
            visited = set()
        count = 0
        for child in dependents.get(bid, set()):
            if child not in visited:
                visited.add(child)
                count += 1 + _count_transitive(child, visited)
        return count

    scores: dict[str, float] = {}
    for b in beliefs:
        transitive = _count_transitive(b.id)
        # Base score of 1 + transitive dependents
        score = 1.0 + transitive
        # Boost for high confidence (more damage if wrong)
        if b.confidence == ConfidenceLevel.HIGH:
            score *= 1.5
        scores[b.id] = score

    return scores


def rank_beliefs(beliefs: list[Belief]) -> list[tuple[Belief, float]]:
    """Return beliefs sorted by importance score (highest first)."""
    scores = importance_scores(beliefs)
    ranked = [(b, scores.get(b.id, 1.0)) for b in beliefs]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def cascade_falsify(
    beliefs: list[Belief],
    falsified_id: str,
) -> list[str]:
    """When a belief is falsified, return IDs of all downstream beliefs that should be reviewed.

    Walks the dependency graph transitively: if B depends on A and C depends on B,
    falsifying A returns [B, C].
    """
    # Build adjacency: belief_id -> set of direct dependents
    dependents: dict[str, set[str]] = {b.id: set() for b in beliefs}
    for b in beliefs:
        for dep_id in b.depends_on:
            if dep_id in dependents:
                dependents[dep_id].add(b.id)

    # BFS from the falsified belief
    affected: list[str] = []
    queue = list(dependents.get(falsified_id, set()))
    visited = {falsified_id}

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        affected.append(current)
        queue.extend(dependents.get(current, set()))

    return affected
