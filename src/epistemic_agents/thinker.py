"""Thinker agent — deep strategic reasoning with epistemic metadata."""

from __future__ import annotations

from epistemic_agents.client import structured_request
from epistemic_agents.schema import (
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)

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

3. PLAN STEPS: Ordered steps for the executor to follow.

4. DECISION BOUNDARIES: Pre-committed rules for scenarios the executor might encounter. \
For each, specify whether the executor should handle it locally or escalate back to you.

5. OPEN QUESTIONS: What you explicitly do not know. Be honest about uncertainty.

Think deeply. Do not hedge everything — take analytical positions and defend them. \
But be precise about where your confidence is genuinely low.\
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

2. UPDATED BELIEFS: Revise any beliefs that were challenged. Add new beliefs if \
the executor's evidence warrants it. Do not just repeat old beliefs unchanged — \
engage with the evidence.

3. REVISED STEPS: If the strategy changed, provide updated steps. If only clarifying, \
leave this empty.

4. GUIDANCE: Explain your reasoning. Why did you revise (or not)? What principle \
should the executor apply going forward?

5. CONTINUE FROM STEP: Which step should the executor resume from?

Be willing to change your mind. The executor has ground truth you lacked. \
But also push back if the evidence doesn't actually warrant a revision.\
"""


class Thinker:
    """Strategic reasoning agent backed by Claude Opus."""

    def __init__(self, model: str = "opus"):
        self.model = model

    def analyze(self, task: str) -> StrategicHandoff:
        """Perform deep analysis of a task and produce a strategic handoff."""
        return structured_request(
            model=self.model,
            system=ANALYZE_SYSTEM,
            user_message=f"Analyze this task and produce a strategic handoff:\n\n{task}",
            response_model=StrategicHandoff,
        )

    def revise(
        self,
        feedback: ExecutorFeedback,
        original_handoff: StrategicHandoff,
    ) -> ThinkerAmendment:
        """Revise strategy based on executor feedback."""
        user_message = (
            "Here is the original strategic handoff:\n\n"
            f"{original_handoff.model_dump_json(indent=2)}\n\n"
            "The executor has reported the following feedback:\n\n"
            f"{feedback.model_dump_json(indent=2)}\n\n"
            "Review the feedback and produce a strategic amendment."
        )
        return structured_request(
            model=self.model,
            system=REVISE_SYSTEM,
            user_message=user_message,
            response_model=ThinkerAmendment,
        )
