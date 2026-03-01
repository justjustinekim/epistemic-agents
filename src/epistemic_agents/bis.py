"""Belief Importance Scoring (BIS) and dependency cascade falsification."""

from __future__ import annotations

import logging

from epistemic_agents.schema import Belief, CONFIDENCE_LEVEL_TO_SCORE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

_WHITE, _GRAY, _BLACK = 0, 1, 2


def detect_cycles(beliefs: list[Belief]) -> list[list[str]]:
    """Detect cycles in the belief dependency graph using DFS coloring.

    Returns a list of cycles, where each cycle is a list of belief IDs.
    Logs a warning for each cycle found.
    """
    # Build adjacency: belief_id -> depends_on (upstream)
    id_set = {b.id for b in beliefs}
    adj: dict[str, list[str]] = {b.id: [] for b in beliefs}
    for b in beliefs:
        for dep in b.depends_on:
            if dep in id_set:
                adj[b.id].append(dep)

    color: dict[str, int] = {bid: _WHITE for bid in id_set}
    parent: dict[str, str | None] = {bid: None for bid in id_set}
    cycles: list[list[str]] = []

    def _dfs(node: str) -> None:
        color[node] = _GRAY
        for neighbor in adj[node]:
            if color[neighbor] == _GRAY:
                # Found a cycle — reconstruct it
                cycle = [neighbor, node]
                current = node
                while parent[current] is not None and parent[current] != neighbor:
                    current = parent[current]  # type: ignore[assignment]
                    cycle.append(current)
                cycle.reverse()
                cycles.append(cycle)
                logger.warning("Cycle detected in belief graph: %s", " -> ".join(cycle))
            elif color[neighbor] == _WHITE:
                parent[neighbor] = node
                _dfs(neighbor)
        color[node] = _BLACK

    for bid in id_set:
        if color[bid] == _WHITE:
            _dfs(bid)

    return cycles


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


def importance_scores(beliefs: list[Belief]) -> dict[str, float]:
    """Score each belief by how many downstream beliefs depend on it.

    Uses weighted scoring:
    - Each dependent is weighted by its effective_score
    - Confidence multiplier: effective_score + 0.5 (range 0.7-1.4)
    - Testability boost: 1.0 + 0.1 * min(len(falsification_conditions), 5)
    """
    belief_map = {b.id: b for b in beliefs}

    # Build adjacency: belief_id -> set of direct dependents
    dependents: dict[str, set[str]] = {b.id: set() for b in beliefs}
    for b in beliefs:
        for dep_id in b.depends_on:
            if dep_id in dependents:
                dependents[dep_id].add(b.id)

    # Detect cycles first so we know about them
    detect_cycles(beliefs)

    # Count transitive dependents via DFS (with cycle safety)
    def _count_transitive(bid: str) -> float:
        visited = {bid}  # Include starting node to prevent infinite recursion on cycles
        stack = list(dependents.get(bid, set()))
        total = 0.0
        while stack:
            child = stack.pop()
            if child not in visited:
                visited.add(child)
                # Weight by the dependent's effective score
                child_belief = belief_map.get(child)
                weight = child_belief.effective_score if child_belief else 0.5
                total += weight
                stack.extend(dependents.get(child, set()))
        return total

    scores: dict[str, float] = {}
    for b in beliefs:
        transitive = _count_transitive(b.id)
        # Base score of 1 + weighted transitive dependents
        score = 1.0 + transitive
        # Confidence multiplier: effective_score + 0.5 (range 0.7-1.4)
        confidence_mult = b.effective_score + 0.5
        score *= confidence_mult
        # Testability boost: 1.0 + 0.1 * min(len(falsification_conditions), 5) (cap +50%)
        testability = 1.0 + 0.1 * min(len(b.falsification_conditions), 5)
        score *= testability
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
