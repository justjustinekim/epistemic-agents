"""Context management — summarize and budget debate context for token limits."""

from __future__ import annotations

from epistemic_agents.schema import ProviderPosition


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def summarize_round(
    positions: list[ProviderPosition],
    model: str = "haiku",
) -> str:
    """Summarize a round's positions into a compact text using a cheap model."""
    from epistemic_agents import client

    sections = []
    for pos in positions:
        sections.append(
            f"{pos.provider_name} ({pos.model_id}): {pos.raw_analysis[:500]}"
        )
    prompt = (
        "Summarize these model analyses into 2-3 sentences each, "
        "preserving key claims and disagreements:\n\n"
        + "\n\n".join(sections)
    )
    return client.plain_request(
        model=model,
        system="You are a concise summarizer. Keep it brief.",
        user_message=prompt,
    )


def manage_context(
    task: str,
    all_rounds: list[list[ProviderPosition]],
    max_tokens: int = 100_000,
    summarize_model: str = "haiku",
) -> str:
    """Build debate context within a token budget.

    Strategy:
    - Always include the task description
    - Always keep the latest round verbatim
    - If total exceeds budget, summarize earlier rounds

    Returns formatted context string.
    """
    sections = [f"# Original Task\n{task}\n"]

    if not all_rounds:
        return "\n".join(sections)

    # Build verbatim content for the latest round
    latest_round = all_rounds[-1]
    latest_section = _format_round(latest_round, len(all_rounds))
    latest_tokens = estimate_tokens(latest_section)

    # Build verbatim content for earlier rounds
    earlier_sections: list[str] = []
    for i, positions in enumerate(all_rounds[:-1]):
        earlier_sections.append(_format_round(positions, i + 1))

    # Check total
    task_tokens = estimate_tokens(sections[0])
    earlier_tokens = sum(estimate_tokens(s) for s in earlier_sections)
    total = task_tokens + earlier_tokens + latest_tokens

    if total <= max_tokens:
        # Everything fits verbatim
        sections.extend(earlier_sections)
        sections.append(latest_section)
    else:
        # Need to summarize earlier rounds
        for i, positions in enumerate(all_rounds[:-1]):
            try:
                summary = summarize_round(positions, model=summarize_model)
                label = "Initial Analysis" if i == 0 else f"Response Round {i}"
                sections.append(f"---\n# {label} (summarized)\n{summary}\n")
            except Exception:
                # Fallback: truncate
                sections.append(_format_round(positions, i + 1, truncate=500))
        sections.append(latest_section)

    sections.append(
        "---\n"
        "Now respond to the above. Challenge what you disagree with, "
        "build on what's good, defend or update your positions."
    )
    return "\n".join(sections)


def _format_round(
    positions: list[ProviderPosition],
    round_num: int,
    truncate: int | None = None,
) -> str:
    label = "Initial Analysis" if round_num == 1 else f"Response Round {round_num - 1}"
    parts = [f"---\n# {label}\n"]
    for pos in positions:
        text = pos.raw_analysis
        if truncate:
            text = text[:truncate] + "..." if len(text) > truncate else text
        parts.append(
            f"## {pos.provider_name} ({pos.model_id}):\n{text}\n"
        )
    return "\n".join(parts)
