"""Position tracking — detect how model stances shift across debate rounds."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_agents.rag import _tokenize, _jaccard_similarity
from epistemic_agents.schema import ProviderPosition


@dataclass
class StanceShift:
    """A detected shift in a provider's confidence on a claim."""

    provider_name: str
    claim: str
    old_confidence_score: float
    new_confidence_score: float
    direction: str  # "strengthened", "weakened", or "reversed"


def track_positions(
    rounds: list[list[ProviderPosition]],
    similarity_threshold: float = 0.4,
    min_delta: float = 0.1,
) -> list[StanceShift]:
    """Compare beliefs across rounds to detect stance shifts.

    Args:
        rounds: List of round results (each round is a list of positions).
        similarity_threshold: Jaccard threshold for claim matching.
        min_delta: Minimum confidence change to count as a shift.

    Returns:
        List of detected stance shifts.
    """
    if len(rounds) < 2:
        return []

    shifts: list[StanceShift] = []

    # Compare each consecutive pair of rounds
    for r_idx in range(1, len(rounds)):
        prev_round = rounds[r_idx - 1]
        curr_round = rounds[r_idx]

        # Index previous round beliefs by provider
        prev_by_provider: dict[str, list[tuple[str, float, set[str]]]] = {}
        for pos in prev_round:
            for belief in pos.beliefs:
                tokens = _tokenize(belief.claim)
                prev_by_provider.setdefault(pos.provider_name, []).append(
                    (belief.claim, belief.effective_score, tokens)
                )

        # Match current round beliefs to previous
        for pos in curr_round:
            prev_beliefs = prev_by_provider.get(pos.provider_name, [])
            for belief in pos.beliefs:
                curr_tokens = _tokenize(belief.claim)
                curr_score = belief.effective_score

                # Find best matching previous belief
                best_sim = 0.0
                best_prev: tuple[str, float, set[str]] | None = None
                for prev_claim, prev_score, prev_tokens in prev_beliefs:
                    sim = _jaccard_similarity(curr_tokens, prev_tokens)
                    if sim > best_sim:
                        best_sim = sim
                        best_prev = (prev_claim, prev_score, prev_tokens)

                if best_prev and best_sim >= similarity_threshold:
                    delta = curr_score - best_prev[1]
                    if abs(delta) >= min_delta:
                        if delta > 0:
                            direction = "strengthened"
                        elif delta < -0.5:
                            direction = "reversed"
                        else:
                            direction = "weakened"

                        shifts.append(StanceShift(
                            provider_name=pos.provider_name,
                            claim=belief.claim,
                            old_confidence_score=best_prev[1],
                            new_confidence_score=curr_score,
                            direction=direction,
                        ))

    return shifts


def format_position_summary(shifts: list[StanceShift]) -> str:
    """Format stance shifts into text for injection into synthesis context."""
    if not shifts:
        return ""

    lines = ["POSITION SHIFTS DETECTED:"]
    for s in shifts:
        lines.append(
            f"  - {s.provider_name}: {s.direction} on '{s.claim}' "
            f"({s.old_confidence_score:.2f} → {s.new_confidence_score:.2f})"
        )
    return "\n".join(lines)
