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


class EscalationType(str, Enum):
    """Categories of executor feedback that warrant thinker attention."""

    CONTRADICTION = "contradiction"  # Evidence directly contradicts a belief
    AMBIGUITY = "ambiguity"  # Strategy doesn't cover this case
    DISCOVERY = "discovery"  # Found something the thinker didn't consider
    ASSUMPTION_VIOLATION = "assumption_violation"  # A key assumption was wrong
    RESOURCE_CONSTRAINT = "resource_constraint"  # Can't do what was asked


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
        description="Category of the issue, if escalating. None means no escalation needed.",
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
