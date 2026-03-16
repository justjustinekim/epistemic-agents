"""Position tracking — detect how model stances shift across debate rounds."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    substantive_engagement: float = 1.0
    is_sycophantic: bool = False


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


def detect_sycophancy(
    shifts: list[StanceShift],
    rounds: list[list[ProviderPosition]],
    jaccard_threshold: float = 0.6,
    novel_token_threshold: float = 0.2,
    confidence_delta_threshold: float = 0.2,
) -> list[StanceShift]:
    """Detect sycophantic stance shifts and annotate them.

    A shift is sycophantic when:
    1. The new position has high Jaccard similarity with the influencer's claims
    2. The new justification has few novel tokens not present in any prior round
    3. The confidence change is significant (> threshold)

    Returns the same shifts list with is_sycophantic and substantive_engagement updated.
    """
    if len(rounds) < 2:
        return shifts

    for shift in shifts:
        if abs(shift.new_confidence_score - shift.old_confidence_score) < confidence_delta_threshold:
            continue

        # Find the shifted belief in the current round
        shifted_tokens = _tokenize(shift.claim)
        shifted_justification_tokens: set[str] = set()
        for pos in rounds[-1]:
            if pos.provider_name == shift.provider_name:
                for b in pos.beliefs:
                    if _jaccard_similarity(_tokenize(b.claim), shifted_tokens) > 0.4:
                        shifted_justification_tokens = _tokenize(b.justification)
                        break

        # Collect all tokens from other providers in the previous round
        influencer_claim_tokens: set[str] = set()
        for pos in rounds[-2]:
            if pos.provider_name != shift.provider_name:
                for b in pos.beliefs:
                    influencer_claim_tokens |= _tokenize(b.claim)
                    influencer_claim_tokens |= _tokenize(b.justification)

        # Collect all tokens from this provider's prior rounds
        own_prior_tokens: set[str] = set()
        for rnd in rounds[:-1]:
            for pos in rnd:
                if pos.provider_name == shift.provider_name:
                    for b in pos.beliefs:
                        own_prior_tokens |= _tokenize(b.claim)
                        own_prior_tokens |= _tokenize(b.justification)

        # Check similarity with influencer
        if not shifted_justification_tokens or not influencer_claim_tokens:
            continue

        jaccard_with_influencer = _jaccard_similarity(
            shifted_justification_tokens, influencer_claim_tokens
        )

        # Novel token ratio: tokens in new justification not in any prior round
        all_prior = influencer_claim_tokens | own_prior_tokens
        novel_tokens = shifted_justification_tokens - all_prior
        novel_ratio = len(novel_tokens) / max(len(shifted_justification_tokens), 1)

        is_syc = (
            jaccard_with_influencer > jaccard_threshold
            and novel_ratio < novel_token_threshold
            and abs(shift.new_confidence_score - shift.old_confidence_score) > confidence_delta_threshold
        )

        if is_syc:
            shift.is_sycophantic = True
            # Continuous score: higher = more sycophantic
            syc_score = min(1.0, jaccard_with_influencer * (1.0 - novel_ratio))
            shift.substantive_engagement = max(0.0, 1.0 - syc_score)

    return shifts


def compute_deltas(
    rounds: list[list[ProviderPosition]],
    similarity_threshold: float = 0.4,
) -> str:
    """Compute state deltas between rounds for compact debate context.

    For rounds 2+, produces compressed summaries (200 chars + belief IDs per provider)
    plus detailed deltas ("Model X increased confidence on claim Y from 0.4 to 0.8").
    """
    if len(rounds) < 2:
        return ""

    sections: list[str] = []

    # Compressed summary of each provider's latest position
    latest = rounds[-1]
    sections.append("PROVIDER SUMMARIES:")
    for pos in latest:
        belief_ids = [b.id for b in pos.beliefs]
        summary = pos.raw_analysis[:200].replace("\n", " ")
        sections.append(f"  {pos.provider_name}: {summary}... [beliefs: {', '.join(belief_ids)}]")

    # Detailed deltas
    shifts = track_positions(rounds, similarity_threshold=similarity_threshold)
    if shifts:
        sections.append("\nDETAILED DELTAS:")
        for s in shifts:
            syc = " (SYCOPHANTIC)" if s.is_sycophantic else ""
            sections.append(
                f"  {s.provider_name} {s.direction} on '{s.claim}': "
                f"{s.old_confidence_score:.2f} → {s.new_confidence_score:.2f}{syc}"
            )

    # New beliefs (in latest round but not in any prior)
    prior_claims: dict[str, set[str]] = {}
    for rnd in rounds[:-1]:
        for pos in rnd:
            for b in pos.beliefs:
                prior_claims.setdefault(pos.provider_name, set()).add(
                    " ".join(sorted(_tokenize(b.claim)))
                )

    new_beliefs: list[str] = []
    for pos in latest:
        prior = prior_claims.get(pos.provider_name, set())
        for b in pos.beliefs:
            claim_key = " ".join(sorted(_tokenize(b.claim)))
            if claim_key and claim_key not in prior:
                new_beliefs.append(f"  {pos.provider_name}: NEW belief '{b.claim}'")

    if new_beliefs:
        sections.append("\nNEW BELIEFS:")
        sections.extend(new_beliefs)

    # Dropped beliefs
    current_claims: dict[str, set[str]] = {}
    for pos in latest:
        for b in pos.beliefs:
            current_claims.setdefault(pos.provider_name, set()).add(
                " ".join(sorted(_tokenize(b.claim)))
            )

    if len(rounds) >= 2:
        prev_round = rounds[-2]
        dropped: list[str] = []
        for pos in prev_round:
            current = current_claims.get(pos.provider_name, set())
            for b in pos.beliefs:
                claim_key = " ".join(sorted(_tokenize(b.claim)))
                if claim_key and claim_key not in current:
                    dropped.append(f"  {pos.provider_name}: DROPPED '{b.claim}'")
        if dropped:
            sections.append("\nDROPPED BELIEFS:")
            sections.extend(dropped)

    return "\n".join(sections)


def format_position_summary(shifts: list[StanceShift]) -> str:
    """Format stance shifts into text for injection into synthesis context."""
    if not shifts:
        return ""

    lines = ["POSITION SHIFTS DETECTED:"]
    for s in shifts:
        syc_flag = " [SYCOPHANTIC]" if s.is_sycophantic else ""
        lines.append(
            f"  - {s.provider_name}: {s.direction} on '{s.claim}' "
            f"({s.old_confidence_score:.2f} → {s.new_confidence_score:.2f}){syc_flag}"
        )
    return "\n".join(lines)
