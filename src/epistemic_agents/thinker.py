"""Thinker agent — deep strategic reasoning with epistemic metadata."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from epistemic_agents.bis import cascade_falsify, rank_beliefs
from epistemic_agents.client import CallCostTracker, get_last_usage, structured_request
from epistemic_agents.schema import (
    ConfidenceLevel,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)

if TYPE_CHECKING:
    from epistemic_agents.panel import ModelPanel

ANALYZE_SYSTEM = """\
You are a strategic thinker. Your job is to deeply analyze a task and produce a \
structured strategic handoff for an executor agent.

You must think carefully and output:

1. INTENT: What we are trying to achieve and why. This is the "commander's intent" — \
the executor should understand the goal well enough to adapt if circumstances change.

2. BELIEFS: Your analytical conclusions. For each belief:
   - State the claim clearly
   - Assign a confidence level: high (act without questioning), moderate (verify if convenient), \
low (actively verify first), or speculative (best guess, be cautious)
   - Provide your justification — the reasoning chain
   - State falsification conditions — what evidence would prove this belief wrong
   - List key assumptions that must hold
   - If this belief depends on another belief being true, list those belief IDs in \
depends_on. If a foundational belief is challenged, all downstream beliefs should be reviewed.

3. PLAN STEPS: Ordered steps for the executor to follow.

4. DECISION BOUNDARIES: Pre-committed rules for scenarios the executor might encounter. \
For each, specify whether the executor should handle it locally or escalate back to you.

5. OPEN QUESTIONS: What you explicitly do not know. Be honest about uncertainty.

6. META-REASONING: After completing your analysis, step back and reflect:
   - Identify one potential flaw in your own plan
   - Name one assumption you might be overconfident about
   - Note any reasoning pattern you notice yourself falling into \
(e.g., anchoring on the first solution, optimism bias, over-engineering)

Think deeply. Do not hedge everything — take analytical positions and defend them. \
But be precise about where your confidence is genuinely low.\
"""

PANEL_AUGMENTED_SYSTEM = """\
You are a strategic thinker. You previously analyzed a task and identified beliefs \
where your confidence was low or speculative. A multi-model panel has now debated \
those uncertain beliefs — multiple AI models from different families challenged and \
built on each other's positions.

Review the panel synthesis below and produce a REVISED strategic handoff that \
incorporates the panel's insights. Specifically:
- Adjust confidence levels on beliefs the panel debated
- Add new beliefs from blind spots or unique insights the panel surfaced
- Revise plan steps if the panel's analysis changes the approach
- Update your meta-reasoning to reflect what the panel taught you

Do not blindly adopt the panel's conclusions — weigh them critically. But take \
seriously any tensions, blind spots, or agreements the panel identified.\
"""

REVISE_SYSTEM = """\
You are a strategic thinker receiving feedback from an executor who encountered \
something unexpected while carrying out your strategy.

Review the executor's feedback carefully:
- What observations did they make?
- Which of your beliefs are being challenged, and with what evidence?
- What new information did they discover?

Then produce a strategic amendment:

1. AMENDMENT TYPE:
   - "revise" if you need to substantively change the strategy
   - "clarify" if the strategy is fine but the executor needs more context
   - "delegate" if you trust the executor's judgment — provide a guiding principle
   - "abort" if the new evidence makes the entire approach unviable

2. UPDATED BELIEFS: Only include beliefs that changed. Use the same belief IDs \
so they merge correctly with unchanged beliefs. Add new beliefs with new IDs if \
the executor's evidence warrants it. Do not repeat unchanged beliefs.

3. REVISED STEPS: If the strategy changed, provide updated steps. If only clarifying, \
leave this empty.

4. REVISED DECISION BOUNDARIES: If any decision boundaries need updating based on \
new evidence, include the full revised list. Leave empty if boundaries are unchanged.

5. RESOLVED QUESTIONS: List any open questions from the original handoff that the \
executor's feedback has now answered.

6. NEW OPEN QUESTIONS: List any new uncertainties that emerged from the feedback.

7. GUIDANCE: Explain your reasoning. Why did you revise (or not)? What principle \
should the executor apply going forward?

8. CONTINUE FROM STEP: Which step should the executor resume from?

Be willing to change your mind. The executor has ground truth you lacked. \
But also push back if the evidence doesn't actually warrant a revision.\
"""

# Beliefs at or below this threshold trigger panel consultation
_PANEL_THRESHOLDS = {ConfidenceLevel.LOW, ConfidenceLevel.SPECULATIVE}


class Thinker:
    """Strategic reasoning agent backed by Claude Opus, optionally panel-informed."""

    def __init__(
        self,
        model: str = "opus",
        panel: ModelPanel | None = None,
        panel_threshold: ConfidenceLevel = ConfidenceLevel.LOW,
        calibration_context: str = "",
    ):
        self.model = model
        self._panel = panel
        self._panel_threshold = panel_threshold
        self._calibration_context = calibration_context
        self.cost = CallCostTracker()
        # Build the set of confidence levels that trigger panel
        self._trigger_levels = set()
        for level in ConfidenceLevel:
            self._trigger_levels.add(level)
            if level == panel_threshold:
                break

    def analyze(self, task: str) -> StrategicHandoff:
        """Perform deep analysis, optionally consulting the panel on uncertain beliefs."""
        system = ANALYZE_SYSTEM
        if self._calibration_context:
            system = system + "\n\n" + self._calibration_context

        handoff = structured_request(
            model=self.model,
            system=system,
            user_message=f"Analyze this task and produce a strategic handoff:\n\n{task}",
            response_model=StrategicHandoff,
        )
        usage = get_last_usage()
        if usage:
            self.cost.calls.append(usage)

        # If panel is configured, consult it on uncertain beliefs
        if self._panel and self._has_uncertain_beliefs(handoff):
            handoff = self._panel_augment(task, handoff)

        return handoff

    def revise(
        self,
        feedback: ExecutorFeedback,
        original_handoff: StrategicHandoff,
    ) -> ThinkerAmendment:
        """Revise strategy based on executor feedback, with cascade awareness."""
        # Build cascade context: if beliefs were challenged, identify downstream impact
        cascade_context = ""
        if feedback.challenged_beliefs and original_handoff.beliefs:
            all_affected: set[str] = set()
            for cb in feedback.challenged_beliefs:
                affected = cascade_falsify(original_handoff.beliefs, cb.belief_id)
                all_affected.update(affected)
            if all_affected:
                # Include BIS ranking for context
                ranked = rank_beliefs(original_handoff.beliefs)
                ranked_str = ", ".join(
                    f"{b.id} (importance: {s:.1f})" for b, s in ranked[:5]
                )
                cascade_context = (
                    f"\n\nCASCADE ALERT: Falsifying {[cb.belief_id for cb in feedback.challenged_beliefs]} "
                    f"affects downstream beliefs: {sorted(all_affected)}. "
                    f"These beliefs depend (directly or transitively) on the challenged beliefs "
                    f"and should be reviewed.\n"
                    f"Belief importance ranking: {ranked_str}"
                )

        user_message = (
            "Here is the original strategic handoff:\n\n"
            f"{original_handoff.model_dump_json(indent=2)}\n\n"
            "The executor has reported the following feedback:\n\n"
            f"{feedback.model_dump_json(indent=2)}\n\n"
            f"{cascade_context}\n\n"
            "Review the feedback and produce a strategic amendment."
        )
        result = structured_request(
            model=self.model,
            system=REVISE_SYSTEM,
            user_message=user_message,
            response_model=ThinkerAmendment,
        )
        usage = get_last_usage()
        if usage:
            self.cost.calls.append(usage)
        return result

    def _has_uncertain_beliefs(self, handoff: StrategicHandoff) -> bool:
        """Check if any beliefs fall at or below the panel threshold."""
        return any(b.confidence in self._trigger_levels for b in handoff.beliefs)

    def _panel_augment(self, task: str, handoff: StrategicHandoff) -> StrategicHandoff:
        """Consult the panel on uncertain beliefs, then re-analyze."""
        from epistemic_agents.synthesizer import Synthesizer

        uncertain = [
            b for b in handoff.beliefs if b.confidence in self._trigger_levels
        ]

        # Build a focused query for the panel
        belief_summary = "\n".join(
            f"- [{b.confidence.value}] {b.id}: {b.claim}" for b in uncertain
        )
        panel_task = (
            f"# Original Task\n{task}\n\n"
            f"# Thinker's Uncertain Beliefs\n"
            f"The thinker produced the following beliefs with low confidence. "
            f"Debate whether these are correct, what's missing, and what the "
            f"thinker might be wrong about:\n\n{belief_summary}\n\n"
            f"# Thinker's Full Analysis\n"
            f"Intent: {handoff.intent}\n\n"
            f"Meta-reasoning: {handoff.meta_reasoning}\n\n"
            f"Challenge these uncertain beliefs. What is the thinker missing?"
        )

        print(
            f"[panel] Consulting panel on {len(uncertain)} uncertain belief(s)...",
            file=sys.stderr,
        )

        # Single-round panel query (fast, focused)
        positions = self._panel.run(panel_task)

        # Synthesize panel positions
        synthesizer = Synthesizer(model=self.model)
        synthesis = synthesizer.synthesize(panel_task, positions)

        print(
            f"[panel] Panel returned {len(positions)} positions, synthesized.",
            file=sys.stderr,
        )

        # Re-run the thinker with panel context
        user_message = (
            f"Analyze this task and produce a strategic handoff:\n\n{task}\n\n"
            f"--- PANEL SYNTHESIS (multi-model debate on your uncertain beliefs) ---\n\n"
            f"Agreements:\n"
            + "\n".join(
                f"- [{a.combined_confidence.value}] {a.claim}"
                for a in synthesis.agreements
            )
            + f"\n\nTensions:\n"
            + "\n".join(f"- {t.claim}: {t.synthesis_notes}" for t in synthesis.tensions)
            + f"\n\nBlind Spots:\n"
            + "\n".join(
                f"- {bs.observation} (caught by {bs.identified_by})"
                for bs in synthesis.blind_spots
            )
            + f"\n\nSynthesized Strategy:\n{synthesis.synthesized_strategy}"
        )

        result = structured_request(
            model=self.model,
            system=PANEL_AUGMENTED_SYSTEM,
            user_message=user_message,
            response_model=StrategicHandoff,
        )
        usage = get_last_usage()
        if usage:
            self.cost.calls.append(usage)
        return result
