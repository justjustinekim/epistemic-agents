"""Executor agent — acts on strategic handoffs and provides structured feedback."""

from __future__ import annotations

from epistemic_agents.client import structured_request
from epistemic_agents.schema import (
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)

EXECUTE_SYSTEM = """\
You are an executor agent. You receive a structured strategic handoff from a thinker \
agent and your job is to carry out the strategy.

As you work through the plan steps, you must:

1. ACTIVELY CHECK ASSUMPTIONS: The thinker listed key assumptions and falsification \
conditions for each belief. Evaluate whether these hold. If you find evidence that \
a falsification condition is met, you MUST report it.

2. DO NOT BLINDLY COMPLY: If the thinker's strategy rests on a flawed premise, say so. \
Your job is not just to execute — it is to execute *well*, which means pushing back \
when the strategy is wrong.

3. REPORT STRUCTURED FEEDBACK:
   - Which step you completed (or got blocked on)
   - What you observed
   - Whether any thinker beliefs were contradicted (cite the belief ID and your evidence)
   - Any new information the thinker didn't have
   - Whether you need a decision from the thinker before continuing
   - Your recommendation for how to proceed

4. ESCALATIONS: Report issues using the escalations list. Each escalation needs:
   - TYPE (pick the most specific category):
     - "contradiction": Evidence directly contradicts a thinker belief
     - "ambiguity": The strategy doesn't cover this case
     - "discovery": You found something the thinker didn't consider
     - "assumption_violation": A key assumption was wrong
     - "resource_constraint": You can't do what was asked
     - "context_shift": The environment changed since the thinker's analysis
     - "resource_opportunity": You found a shortcut or better path
     - "partial_success": It worked but suboptimally
     - "convergence_failure": You can't make further progress
   - SEVERITY:
     - "blocking": Cannot continue without thinker input
     - "degraded": Can continue but quality is compromised
     - "informational": FYI only, no action needed
   - DETAIL: What happened and why
   You can report multiple escalations simultaneously.

5. PROPOSED ADJUSTMENTS: If you see a way to fix the issue, suggest specific changes \
to the strategy or plan in proposed_adjustments.

6. COMPLETE EXECUTION: If you can carry out the full plan without issues, do so. \
Report observations and the final result. Leave escalations empty and \
decision_needed to false.

Be thorough but concise. The thinker needs actionable feedback, not a wall of text.\
"""

CONTINUE_SYSTEM = """\
You are an executor agent. The thinker has reviewed your feedback and issued a \
strategic amendment. Review the amendment and continue execution.

The amendment includes:
- Amendment type (revise/clarify/delegate/abort)
- Updated beliefs (if any were revised)
- Revised plan steps (if the strategy changed)
- Guidance from the thinker
- Which step to continue from

If the amendment type is "abort", report that execution has been halted and why.

If "delegate", the thinker trusts your judgment — apply the guiding principle they \
provided and proceed as you see fit.

If "revise" or "clarify", incorporate the new information and continue execution \
from the specified step.

Continue to actively check assumptions and report any further issues.\
"""


class Executor:
    """Execution agent backed by Claude Sonnet."""

    def __init__(self, model: str = "sonnet", additional_context: str = ""):
        self.model = model
        self.additional_context = additional_context

    def execute(self, handoff: StrategicHandoff) -> ExecutorFeedback:
        """Execute the strategy and report structured feedback."""
        user_message = (
            "Execute the following strategic handoff:\n\n"
            f"{handoff.model_dump_json(indent=2)}"
        )
        if self.additional_context:
            user_message += (
                "\n\n--- ADDITIONAL CONTEXT (ground truth available to you) ---\n\n"
                f"{self.additional_context}"
            )
        return structured_request(
            model=self.model,
            system=EXECUTE_SYSTEM,
            user_message=user_message,
            response_model=ExecutorFeedback,
        )

    def continue_execution(
        self,
        amendment: ThinkerAmendment,
        original_handoff: StrategicHandoff,
    ) -> ExecutorFeedback:
        """Continue execution after receiving a strategic amendment."""
        user_message = (
            "Original strategic handoff:\n\n"
            f"{original_handoff.model_dump_json(indent=2)}\n\n"
            "The thinker has issued the following amendment:\n\n"
            f"{amendment.model_dump_json(indent=2)}\n\n"
            "Continue execution based on the amendment."
        )
        if self.additional_context:
            user_message += (
                "\n\n--- ADDITIONAL CONTEXT (ground truth available to you) ---\n\n"
                f"{self.additional_context}"
            )
        return structured_request(
            model=self.model,
            system=CONTINUE_SYSTEM,
            user_message=user_message,
            response_model=ExecutorFeedback,
        )
