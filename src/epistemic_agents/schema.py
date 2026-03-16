"""Epistemic protocol schema — structured models for belief exchange between agents."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Qualitative confidence levels for epistemic claims."""

    HIGH = "high"  # Act on this without questioning
    MODERATE = "moderate"  # Verify if convenient, but proceed
    LOW = "low"  # Actively verify before acting
    SPECULATIVE = "speculative"  # Best guess, treat with caution


class BeliefGrounding(str, Enum):
    """How a belief was established — its evidential basis."""

    EMPIRICAL = "empirical"  # Verified by execution or observation
    MODEL_CONSENSUS = "model_consensus"  # Multiple models independently agree
    SINGLE_MODEL = "single_model"  # One model's analysis
    ASSUMED = "assumed"  # Background assumption, not verified


class VerificationMethod(str, Enum):
    """How a belief was verified after initial formation."""

    EXECUTOR_CHALLENGE = "executor_challenge"  # Executor tested and confirmed
    CODE_EXECUTION = "code_execution"  # Code ran and produced expected result
    MODEL_CONSENSUS = "model_consensus"  # Multiple models independently confirmed
    UNVERIFIED = "unverified"  # Not yet verified


# ---------------------------------------------------------------------------
# Confidence score mappings (qualitative <-> quantitative)
# ---------------------------------------------------------------------------

CONFIDENCE_LEVEL_TO_SCORE: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.HIGH: 0.9,
    ConfidenceLevel.MODERATE: 0.7,
    ConfidenceLevel.LOW: 0.4,
    ConfidenceLevel.SPECULATIVE: 0.2,
}

SCORE_TO_CONFIDENCE_LEVEL: list[tuple[float, ConfidenceLevel]] = [
    (0.85, ConfidenceLevel.HIGH),
    (0.55, ConfidenceLevel.MODERATE),
    (0.3, ConfidenceLevel.LOW),
    (0.0, ConfidenceLevel.SPECULATIVE),
]


def score_to_confidence_level(score: float) -> ConfidenceLevel:
    """Convert a numeric confidence score (0.0-1.0) to a qualitative level."""
    for threshold, level in SCORE_TO_CONFIDENCE_LEVEL:
        if score >= threshold:
            return level
    return ConfidenceLevel.SPECULATIVE


class Belief(BaseModel):
    """A structured epistemic claim with justification and falsification conditions."""

    id: str = Field(description="Short identifier for this belief, e.g. 'b1', 'cache-strategy'")
    claim: str = Field(description="The substantive claim being made")
    confidence: ConfidenceLevel
    justification: str = Field(description="Why this belief is held — the reasoning chain")
    falsification_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that would prove this belief wrong. "
        "E.g. 'If the data shows X, this claim fails.'",
    )
    key_assumptions: list[str] = Field(
        default_factory=list,
        description="Background assumptions that must hold for this belief to be valid",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of beliefs this one depends on — if upstream beliefs are "
        "challenged, this belief should be reviewed too",
    )
    grounding: BeliefGrounding = Field(
        default=BeliefGrounding.ASSUMED,
        description="How this belief was established",
    )
    confidence_score: float | None = Field(
        default=None,
        description="Numeric confidence 0.0-1.0, optional alongside qualitative level",
    )
    reasoning_basis: str | None = Field(
        default=None,
        description="The core reasoning path used to arrive at this belief — "
        "enables latent disagreement detection when claims agree but reasoning diverges",
    )

    @property
    def effective_score(self) -> float:
        """Return confidence_score if set, else map from qualitative level."""
        if self.confidence_score is not None:
            return self.confidence_score
        return CONFIDENCE_LEVEL_TO_SCORE.get(self.confidence, 0.5)


class DecisionBoundary(BaseModel):
    """A pre-committed response to a scenario the executor might encounter."""

    condition: str = Field(description="The scenario: 'If you encounter X...'")
    action: str = Field(description="What to do: '...then do Y'")
    escalate: bool = Field(
        default=False,
        description="If true, this scenario should be reported back to the thinker",
    )


class StrategicHandoff(BaseModel):
    """The thinker's structured output — a strategy with epistemic metadata."""

    intent: str = Field(description="What we are trying to achieve and why (the commander's intent)")
    beliefs: list[Belief] = Field(description="The thinker's conclusions with epistemic metadata")
    plan_steps: list[str] = Field(description="Ordered steps for the executor to follow")
    decision_boundaries: list[DecisionBoundary] = Field(
        default_factory=list,
        description="Pre-committed responses to anticipated scenarios",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="What the thinker explicitly does not know",
    )
    meta_reasoning: str = Field(
        default="",
        description="The thinker's self-reflection: potential flaws in its own plan, "
        "assumptions it might be overconfident about, reasoning patterns it notices",
    )


class EscalationType(str, Enum):
    """Categories of executor feedback that warrant thinker attention."""

    CONTRADICTION = "contradiction"  # Evidence directly contradicts a belief
    AMBIGUITY = "ambiguity"  # Strategy doesn't cover this case
    DISCOVERY = "discovery"  # Found something the thinker didn't consider
    ASSUMPTION_VIOLATION = "assumption_violation"  # A key assumption was wrong
    RESOURCE_CONSTRAINT = "resource_constraint"  # Can't do what was asked
    CONTEXT_SHIFT = "context_shift"  # Environment changed since analysis
    RESOURCE_OPPORTUNITY = "resource_opportunity"  # Found a shortcut or better path
    PARTIAL_SUCCESS = "partial_success"  # Worked but suboptimally
    CONVERGENCE_FAILURE = "convergence_failure"  # Can't make further progress


class EscalationSeverity(str, Enum):
    """How urgently the thinker needs to respond."""

    BLOCKING = "blocking"  # Cannot continue without thinker input
    DEGRADED = "degraded"  # Can continue but quality/approach is compromised
    INFORMATIONAL = "informational"  # FYI — no action needed, but thinker should know


class Escalation(BaseModel):
    """A single escalation from the executor to the thinker."""

    type: EscalationType
    severity: EscalationSeverity
    detail: str = Field(description="What happened and why this is being escalated")


class ChallengedBelief(BaseModel):
    """A reference to a thinker belief that the executor is challenging."""

    belief_id: str = Field(description="ID of the challenged belief")
    evidence: str = Field(description="What the executor found that contradicts this belief")


class ExecutorFeedback(BaseModel):
    """The executor's structured report back to the thinker."""

    step_completed: Optional[int] = Field(
        default=None,
        description="Index of the last plan step completed (0-based), or None if blocked before starting",
    )
    observations: list[str] = Field(
        default_factory=list,
        description="What the executor observed during execution",
    )
    escalation_type: Optional[EscalationType] = Field(
        default=None,
        description="Primary escalation category, if escalating. None means no escalation needed. "
        "For multiple escalations, use the escalations list.",
    )
    escalations: list[Escalation] = Field(
        default_factory=list,
        description="All escalations with type, severity, and detail. "
        "Supports multiple simultaneous escalations.",
    )
    challenged_beliefs: list[ChallengedBelief] = Field(
        default_factory=list,
        description="Thinker beliefs that were contradicted by evidence",
    )
    new_evidence: list[str] = Field(
        default_factory=list,
        description="Information the thinker didn't have that may affect strategy",
    )
    decision_needed: bool = Field(
        default=False,
        description="Whether the executor needs guidance before continuing",
    )
    executor_recommendation: Optional[str] = Field(
        default=None,
        description="The executor's suggested course of action, if it has one",
    )
    proposed_adjustments: Optional[str] = Field(
        default=None,
        description="Specific changes the executor suggests to the strategy or plan",
    )
    execution_result: Optional[str] = Field(
        default=None,
        description="Final output or result of execution, if completed",
    )


class AmendmentType(str, Enum):
    """How the thinker responds to executor feedback."""

    REVISE = "revise"  # Substantively change the strategy
    CLARIFY = "clarify"  # Provide more context, keep strategy unchanged
    DELEGATE = "delegate"  # Trust the executor's judgment, provide guiding principle
    ABORT = "abort"  # Abandon this approach entirely


class ThinkerAmendment(BaseModel):
    """The thinker's response to executor feedback — a strategic revision."""

    amendment_type: AmendmentType
    updated_beliefs: list[Belief] = Field(
        default_factory=list,
        description="Revised or new beliefs incorporating the executor's evidence",
    )
    revised_steps: Optional[list[str]] = Field(
        default=None,
        description="Updated plan steps, if the strategy changed. None means keep original steps.",
    )
    revised_decision_boundaries: Optional[list[DecisionBoundary]] = Field(
        default=None,
        description="Updated decision boundaries, if any changed. None means keep original.",
    )
    resolved_questions: list[str] = Field(
        default_factory=list,
        description="Open questions from the original handoff that are now answered",
    )
    new_open_questions: list[str] = Field(
        default_factory=list,
        description="New open questions discovered during this revision",
    )
    guidance: str = Field(description="Explanation of the revision and how to proceed")
    continue_from_step: Optional[int] = Field(
        default=None,
        description="Which step to resume from (0-based). None means start over.",
    )


class ConversationEntry(BaseModel):
    """A single entry in the epistemic conversation log."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: str = Field(description="'thinker' or 'executor'")
    entry_type: str = Field(description="'handoff', 'feedback', or 'amendment'")
    content: StrategicHandoff | ExecutorFeedback | ThinkerAmendment


class ConversationLog(BaseModel):
    """Full log of an epistemic exchange between thinker and executor."""

    task: str
    entries: list[ConversationEntry] = Field(default_factory=list)
    round_trips: int = 0
    converged: bool = False

    def add(
        self,
        role: str,
        entry_type: str,
        content: StrategicHandoff | ExecutorFeedback | ThinkerAmendment,
    ) -> None:
        self.entries.append(
            ConversationEntry(role=role, entry_type=entry_type, content=content)
        )


# ---------------------------------------------------------------------------
# Multi-model panel models
# ---------------------------------------------------------------------------


class ProviderPosition(BaseModel):
    """A single model's analysis of a task."""

    provider_name: str = Field(description="Name of the provider (e.g. 'claude', 'gemini')")
    model_id: str = Field(description="Specific model used (e.g. 'opus', 'gemini-2.0-flash')")
    beliefs: list[Belief] = Field(
        default_factory=list,
        description="Structured beliefs extracted from the analysis",
    )
    raw_analysis: str = Field(description="The model's full text analysis")


class AgreementPoint(BaseModel):
    """A point where multiple models agree."""

    claim: str = Field(description="The shared claim or conclusion")
    supporting_providers: list[str] = Field(
        description="Names of providers that support this claim"
    )
    combined_confidence: ConfidenceLevel = Field(
        description="Synthesized confidence level across providers"
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description="Traceability refs, e.g. 'provider:belief_id'",
    )
    combined_confidence_score: float | None = Field(
        default=None,
        description="Numeric combined confidence via log-odds averaging",
    )


class TensionPoint(BaseModel):
    """A point where models disagree or take different stances."""

    claim: str = Field(description="The contested claim or topic")
    positions: dict[str, str] = Field(
        description="Mapping of provider name to their stance on the claim"
    )
    synthesis_notes: str = Field(
        description="The synthesizer's assessment of this tension"
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description="Traceability refs, e.g. 'provider:belief_id'",
    )


class BlindSpot(BaseModel):
    """Something only one model noticed that others missed."""

    observation: str = Field(description="The insight or observation that was missed")
    identified_by: str = Field(description="Provider that caught this")
    missed_by: list[str] = Field(description="Providers that missed this")
    source_refs: list[str] = Field(
        default_factory=list,
        description="Traceability refs, e.g. 'provider:belief_id'",
    )


class UniqueInsight(BaseModel):
    """A novel contribution from one model."""

    insight: str = Field(description="The unique insight or framing")
    source_provider: str = Field(description="Provider that contributed this")
    relevance: str = Field(description="Why this insight matters for the task")
    source_refs: list[str] = Field(
        default_factory=list,
        description="Traceability refs, e.g. 'provider:belief_id'",
    )


class PanelSynthesis(BaseModel):
    """The synthesizer's cross-model analysis output."""

    task: str = Field(description="The original task that was analyzed")
    provider_positions: list[ProviderPosition] = Field(
        description="Each provider's individual analysis"
    )
    agreements: list[AgreementPoint] = Field(
        default_factory=list,
        description="Points where models converge",
    )
    tensions: list[TensionPoint] = Field(
        default_factory=list,
        description="Points where models diverge",
    )
    blind_spots: list[BlindSpot] = Field(
        default_factory=list,
        description="Observations caught by only one model",
    )
    unique_insights: list[UniqueInsight] = Field(
        default_factory=list,
        description="Novel contributions from individual models",
    )
    synthesized_strategy: str = Field(
        description="Final strategy incorporating the best of all perspectives"
    )
    meta_confidence: str = Field(
        description="Overall confidence assessment and caveats"
    )


# ---------------------------------------------------------------------------
# Verdict — concise TL;DR output
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    """Concise decision-oriented summary distilled from a full analysis."""

    decision_point: str = Field(
        description="The key decision the user faces, stated as a question"
    )
    recommendation: str = Field(
        description="1-3 sentence actionable recommendation"
    )
    confidence: ConfidenceLevel = Field(
        description="Overall confidence in the recommendation"
    )
    key_risk: str = Field(
        description="The single biggest failure mode or downside"
    )
    dissent: Optional[str] = Field(
        default=None,
        description="The strongest counterargument to the recommendation, if any",
    )
    tier_used: str = Field(
        default="unknown",
        description="Which analysis tier produced this: quick / standard / deep",
    )
    cost_tokens: Optional[int] = Field(
        default=None,
        description="Approximate total tokens consumed across all models",
    )
    cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated total cost in USD across all models",
    )


class PanelResponse(BaseModel):
    """Structured response from a provider that supports structured output."""

    beliefs: list[Belief]
    raw_analysis: str = Field(description="Free-text analysis preserved for debate context")


class DebatePlan(BaseModel):
    """LLM-generated routing decision for task complexity."""

    tier: str = Field(description="'quick', 'standard', or 'deep'")
    reasoning: str = Field(description="Why this tier was chosen")
    confidence_in_routing: float = Field(
        description="0.0-1.0 confidence in the routing decision"
    )


class DebateCheckpoint(BaseModel):
    """Checkpoint for resuming failed deep-tier debates."""

    task: str
    phase: int = Field(description="1=debate, 2=synthesis, 3=refutation, 4=resynthesis, 5=verdict")
    phase_name: str
    rounds: list = Field(default_factory=list)
    synthesis: PanelSynthesis | None = None
    refutations: list = Field(default_factory=list)
    final_synthesis: PanelSynthesis | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
