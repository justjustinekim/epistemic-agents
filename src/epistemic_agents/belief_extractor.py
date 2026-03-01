"""Belief extraction from raw model analyses using a cheap model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from epistemic_agents import client
from epistemic_agents.schema import Belief, BeliefGrounding, ConfidenceLevel


class ExtractedBeliefs(BaseModel):
    """Container for beliefs extracted from raw analysis text."""

    beliefs: list[Belief] = Field(
        description="3-8 structured beliefs extracted from the analysis"
    )


EXTRACTION_SYSTEM = """\
You are a belief extraction engine. Given a raw analysis from an AI model, \
extract 3-8 key beliefs as structured objects.

For each belief:
- Set the `id` to `{provider_name}-b{N}` where N is the belief number (1-indexed)
- Extract the core claim as a concise sentence
- Determine confidence: high, moderate, low, or speculative
- Provide justification from the original text
- Extract falsification conditions (what would prove this wrong)
- Extract key assumptions the belief depends on
- Set `grounding` to "single_model" (since this is from one model's analysis)
- Leave `depends_on` empty unless clear dependencies exist between extracted beliefs

Focus on substantive claims, not procedural statements. \
Ignore hedging language and extract the actual position being taken."""


def extract_beliefs(
    raw_analysis: str,
    provider_name: str,
    model: str = "haiku",
) -> list[Belief]:
    """Extract structured beliefs from a raw analysis using a cheap model.

    Args:
        raw_analysis: The raw text analysis from a provider.
        provider_name: Name of the provider for belief ID prefixing.
        model: Model to use for extraction (default: haiku for cost).

    Returns:
        List of extracted Belief objects with proper grounding.
    """
    prompt = EXTRACTION_SYSTEM.replace("{provider_name}", provider_name)
    user_msg = (
        f"Extract structured beliefs from this analysis by {provider_name}:\n\n"
        f"{raw_analysis}"
    )

    result = client.structured_request(
        model=model,
        system=prompt,
        user_message=user_msg,
        response_model=ExtractedBeliefs,
    )

    # Ensure grounding is set correctly
    for belief in result.beliefs:
        belief.grounding = BeliefGrounding.SINGLE_MODEL
        # Ensure IDs are prefixed
        if not belief.id.startswith(f"{provider_name}-"):
            belief.id = f"{provider_name}-{belief.id}"

    return result.beliefs
