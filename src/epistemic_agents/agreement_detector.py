"""Programmatic agreement and tension detection from structured beliefs."""

from __future__ import annotations

from epistemic_agents.confidence import aggregate_confidence
from epistemic_agents.rag import _tokenize, _jaccard_similarity
from epistemic_agents.schema import (
    AgreementPoint,
    ConfidenceLevel,
    ProviderPosition,
    TensionPoint,
    score_to_confidence_level,
)


def detect_agreements(
    positions: list[ProviderPosition],
    similarity_threshold: float = 0.4,
) -> list[AgreementPoint]:
    """Detect agreements by clustering beliefs across providers.

    Beliefs from different providers with high claim similarity are grouped.
    Groups with 2+ providers become AgreementPoints.
    """
    # Collect all beliefs with their provider names
    belief_entries: list[tuple[str, str, str, float]] = []  # (provider, belief_id, claim, score)
    for pos in positions:
        for belief in pos.beliefs:
            belief_entries.append((
                pos.provider_name,
                belief.id,
                belief.claim,
                belief.effective_score,
            ))

    if not belief_entries:
        return []

    # Cluster by similarity
    clusters: list[list[int]] = []
    assigned: set[int] = set()

    for i in range(len(belief_entries)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        tokens_i = _tokenize(belief_entries[i][2])

        for j in range(i + 1, len(belief_entries)):
            if j in assigned:
                continue
            # Only cluster beliefs from different providers
            if belief_entries[j][0] == belief_entries[i][0]:
                continue
            tokens_j = _tokenize(belief_entries[j][2])
            if _jaccard_similarity(tokens_i, tokens_j) >= similarity_threshold:
                cluster.append(j)
                assigned.add(j)

        if len(cluster) >= 2:
            # Check that at least 2 different providers
            providers_in_cluster = {belief_entries[idx][0] for idx in cluster}
            if len(providers_in_cluster) >= 2:
                clusters.append(cluster)

    # Convert clusters to AgreementPoints
    agreements: list[AgreementPoint] = []
    for cluster in clusters:
        providers = list({belief_entries[idx][0] for idx in cluster})
        scores = [belief_entries[idx][3] for idx in cluster]
        source_refs = [
            f"{belief_entries[idx][0]}:{belief_entries[idx][1]}"
            for idx in cluster
        ]
        combined_score = aggregate_confidence(scores)
        combined_level = score_to_confidence_level(combined_score)

        # Use the first belief's claim as representative
        claim = belief_entries[cluster[0]][2]

        agreements.append(AgreementPoint(
            claim=claim,
            supporting_providers=providers,
            combined_confidence=combined_level,
            source_refs=source_refs,
            combined_confidence_score=combined_score,
        ))

    return agreements


def detect_tensions(
    positions: list[ProviderPosition],
    similarity_threshold: float = 0.3,
    confidence_gap_threshold: float = 0.3,
) -> list[TensionPoint]:
    """Detect tensions — beliefs with similar topics but divergent confidence.

    Finds belief pairs from different providers with high topic similarity
    but significant confidence gap.
    """
    # Collect all beliefs with their provider names
    belief_entries: list[tuple[str, str, str, float]] = []
    for pos in positions:
        for belief in pos.beliefs:
            belief_entries.append((
                pos.provider_name,
                belief.id,
                belief.claim,
                belief.effective_score,
            ))

    if not belief_entries:
        return []

    tensions: list[TensionPoint] = []
    seen_pairs: set[tuple[int, int]] = set()

    for i in range(len(belief_entries)):
        tokens_i = _tokenize(belief_entries[i][2])
        for j in range(i + 1, len(belief_entries)):
            if (i, j) in seen_pairs:
                continue
            # Only compare beliefs from different providers
            if belief_entries[i][0] == belief_entries[j][0]:
                continue

            tokens_j = _tokenize(belief_entries[j][2])
            sim = _jaccard_similarity(tokens_i, tokens_j)
            if sim < similarity_threshold:
                continue

            gap = abs(belief_entries[i][3] - belief_entries[j][3])
            if gap < confidence_gap_threshold:
                continue

            seen_pairs.add((i, j))

            tensions.append(TensionPoint(
                claim=belief_entries[i][2],
                positions={
                    belief_entries[i][0]: f"[{belief_entries[i][3]:.2f}] {belief_entries[i][2]}",
                    belief_entries[j][0]: f"[{belief_entries[j][3]:.2f}] {belief_entries[j][2]}",
                },
                synthesis_notes=f"Confidence gap of {gap:.2f} on similar topic (similarity: {sim:.2f})",
                source_refs=[
                    f"{belief_entries[i][0]}:{belief_entries[i][1]}",
                    f"{belief_entries[j][0]}:{belief_entries[j][1]}",
                ],
            ))

    return tensions
