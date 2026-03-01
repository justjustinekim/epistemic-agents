"""Feedback loop controller — orchestrates the thinker-executor conversation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from epistemic_agents.executor import Executor
from epistemic_agents.schema import (
    AmendmentType,
    ConversationLog,
    EscalationSeverity,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)
from epistemic_agents.thinker import Thinker

if TYPE_CHECKING:
    from epistemic_agents.ledger import BeliefLedger
    from epistemic_agents.panel import ModelPanel

console = Console()


class EpistemicLoop:
    """Manages the epistemic feedback loop between thinker and executor."""

    def __init__(
        self,
        thinker: Thinker,
        executor: Executor,
        max_rounds: int = 5,
        verbose: bool = True,
        ledger: BeliefLedger | None = None,
        panel: ModelPanel | None = None,
    ):
        self.thinker = thinker
        self.executor = executor
        self.max_rounds = max_rounds
        self.verbose = verbose
        self._ledger = ledger
        self._panel = panel

    def run(self, task: str) -> ConversationLog:
        """Run the full epistemic loop for a given task."""
        log = ConversationLog(task=task)

        # Phase 1: Thinker analyzes the task
        if self.verbose:
            console.print(Panel("THINKER: Analyzing task...", style="bold blue"))

        handoff = self.thinker.analyze(task)
        log.add(role="thinker", entry_type="handoff", content=handoff)

        if self.verbose:
            _print_handoff(handoff)

        # Phase 2: Execute and iterate
        current_handoff = handoff
        for round_num in range(self.max_rounds):
            log.round_trips = round_num + 1

            if self.verbose:
                console.print(
                    Panel(f"EXECUTOR: Round {round_num + 1}...", style="bold green")
                )

            # Executor attempts to carry out the strategy
            if round_num == 0:
                feedback = self.executor.execute(current_handoff)
            else:
                # After round 1, the executor continues from the last amendment
                last_amendment = _last_amendment(log)
                if last_amendment:
                    feedback = self.executor.continue_execution(
                        last_amendment, current_handoff
                    )
                else:
                    feedback = self.executor.execute(current_handoff)

            log.add(role="executor", entry_type="feedback", content=feedback)

            if self.verbose:
                _print_feedback(feedback)

            # Step 5: If executor hits BLOCKING escalation and panel is available,
            # trigger a targeted panel query for additional perspectives
            if self._panel and _has_blocking_escalation(feedback):
                feedback = self._panel_escalation(feedback, current_handoff, task)

            # Check if we've converged (no escalation needed)
            has_escalation = (
                feedback.escalation_type is not None or len(feedback.escalations) > 0
            )
            if not feedback.decision_needed and not has_escalation:
                log.converged = True
                if self.verbose:
                    console.print(
                        Panel(
                            "CONVERGED — Executor completed without escalation.",
                            style="bold cyan",
                        )
                    )
                break

            # Thinker reviews feedback and amends strategy
            if self.verbose:
                console.print(
                    Panel("THINKER: Reviewing feedback...", style="bold blue")
                )

            amendment = self.thinker.revise(feedback, current_handoff)
            log.add(role="thinker", entry_type="amendment", content=amendment)

            if self.verbose:
                _print_amendment(amendment)

            # If thinker aborts, we're done
            if amendment.amendment_type == AmendmentType.ABORT:
                if self.verbose:
                    console.print(
                        Panel("ABORTED — Thinker abandoned the approach.", style="bold red")
                    )
                break

            # Update the handoff with amendment changes
            current_handoff = _apply_amendment(current_handoff, amendment)

        else:
            if self.verbose:
                console.print(
                    Panel(
                        f"MAX ROUNDS ({self.max_rounds}) reached without convergence.",
                        style="bold yellow",
                    )
                )

        # Record belief outcomes to ledger (must be outside the for/else)
        if self._ledger:
            records = self._ledger.record_outcomes(log)
            if self.verbose and records:
                console.print(
                    f"\n  [dim]Ledger: recorded {len(records)} belief outcome(s)[/dim]"
                )

        # Print cost summary
        if self.verbose:
            t_cost = self.thinker.cost.total_cost_usd
            t_calls = len(self.thinker.cost.calls)
            e_cost = self.executor.cost.total_cost_usd
            e_calls = len(self.executor.cost.calls)
            total = t_cost + e_cost
            console.print(
                f"\n  [dim]Cost: ${total:.4f} "
                f"(thinker: ${t_cost:.4f} / {t_calls} calls, "
                f"executor: ${e_cost:.4f} / {e_calls} calls)[/dim]"
            )

        return log


    def _panel_escalation(
        self,
        feedback: ExecutorFeedback,
        handoff: StrategicHandoff,
        task: str,
    ) -> ExecutorFeedback:
        """Trigger a targeted panel query when executor hits a blocking escalation."""
        from epistemic_agents.synthesizer import Synthesizer

        blocking = [
            e for e in feedback.escalations
            if e.severity == EscalationSeverity.BLOCKING
        ]
        if not blocking:
            return feedback

        # Build a focused query about the blocking issues
        issues = "\n".join(
            f"- [{e.type.value}] {e.detail}" for e in blocking
        )
        panel_task = (
            f"# Original Task\n{task}\n\n"
            f"# Current Strategy\nIntent: {handoff.intent}\n\n"
            f"# BLOCKING ESCALATION from executor\n"
            f"The executor has hit blocking issues and cannot continue:\n{issues}\n\n"
            f"# Executor's Observations\n"
            + "\n".join(f"- {o}" for o in feedback.observations)
            + "\n\nProvide your analysis: Is the executor right to escalate? "
            "What approaches could unblock this? What is the executor missing?"
        )

        if self.verbose:
            console.print(
                Panel(
                    f"PANEL ESCALATION: {len(blocking)} blocking issue(s) — consulting panel...",
                    style="bold magenta",
                )
            )

        positions = self._panel.run(panel_task)

        # Synthesize panel advice
        synthesizer = Synthesizer()
        synthesis = synthesizer.synthesize(panel_task, positions)

        if self.verbose:
            console.print(
                f"  [dim]Panel returned {len(positions)} positions on the escalation[/dim]"
            )

        # Enrich the feedback with panel context so the thinker gets it
        panel_context = (
            f"\n\n--- PANEL CONSULTATION ON BLOCKING ESCALATION ---\n"
            f"Strategy: {synthesis.synthesized_strategy}\n"
            f"Confidence: {synthesis.meta_confidence}"
        )
        if synthesis.agreements:
            panel_context += "\nAgreements: " + "; ".join(
                a.claim for a in synthesis.agreements
            )
        if synthesis.tensions:
            panel_context += "\nTensions: " + "; ".join(
                t.claim for t in synthesis.tensions
            )

        # Append panel context to executor recommendation
        enriched = feedback.model_copy()
        current_rec = enriched.executor_recommendation or ""
        enriched.executor_recommendation = current_rec + panel_context
        return enriched


def _has_blocking_escalation(feedback: ExecutorFeedback) -> bool:
    """Check if feedback contains any BLOCKING severity escalation."""
    return any(
        e.severity == EscalationSeverity.BLOCKING for e in feedback.escalations
    )


def _apply_amendment(
    handoff: StrategicHandoff, amendment: ThinkerAmendment
) -> StrategicHandoff:
    """Merge amendment into handoff, preserving untouched state."""
    # Merge beliefs by ID: updated beliefs override, others preserved
    if amendment.updated_beliefs:
        updated_ids = {b.id for b in amendment.updated_beliefs}
        merged = [b for b in handoff.beliefs if b.id not in updated_ids]
        merged.extend(amendment.updated_beliefs)
        beliefs = merged
    else:
        beliefs = handoff.beliefs

    # Use revised steps if provided, otherwise keep original
    steps = amendment.revised_steps or handoff.plan_steps

    # Use revised boundaries if provided, otherwise keep original
    boundaries = (
        amendment.revised_decision_boundaries
        if amendment.revised_decision_boundaries is not None
        else handoff.decision_boundaries
    )

    # Update open questions: remove resolved, add new
    open_qs = [
        q for q in handoff.open_questions if q not in amendment.resolved_questions
    ]
    open_qs.extend(amendment.new_open_questions)

    return StrategicHandoff(
        intent=handoff.intent,
        beliefs=beliefs,
        plan_steps=steps,
        decision_boundaries=boundaries,
        open_questions=open_qs,
    )


def _last_amendment(log: ConversationLog) -> ThinkerAmendment | None:
    for entry in reversed(log.entries):
        if entry.entry_type == "amendment" and isinstance(
            entry.content, ThinkerAmendment
        ):
            return entry.content
    return None


def _print_handoff(handoff: StrategicHandoff) -> None:
    console.print(f"\n[bold]Intent:[/bold] {handoff.intent}\n")
    for b in handoff.beliefs:
        conf_color = {
            "high": "green",
            "moderate": "yellow",
            "low": "red",
            "speculative": "magenta",
        }.get(b.confidence.value, "white")
        console.print(
            f"  [{conf_color}][{b.confidence.value.upper()}][/{conf_color}] "
            f"[bold]{b.id}[/bold]: {b.claim}"
        )
        if b.depends_on:
            console.print(f"    Depends on: {', '.join(b.depends_on)}")
        if b.falsification_conditions:
            for fc in b.falsification_conditions:
                console.print(f"    Falsifiable if: {fc}")
    console.print(f"\n[bold]Plan ({len(handoff.plan_steps)} steps):[/bold]")
    for i, step in enumerate(handoff.plan_steps):
        console.print(f"  {i}. {step}")
    if handoff.open_questions:
        console.print(f"\n[bold]Open questions:[/bold]")
        for q in handoff.open_questions:
            console.print(f"  ? {q}")
    if handoff.meta_reasoning:
        console.print(f"\n[bold]Meta-reasoning:[/bold]")
        console.print(f"  {handoff.meta_reasoning}")
    console.print()


def _print_feedback(feedback: ExecutorFeedback) -> None:
    if feedback.step_completed is not None:
        console.print(f"\n  Completed through step: {feedback.step_completed}")
    if feedback.escalation_type:
        console.print(
            f"  [bold red]ESCALATION: {feedback.escalation_type.value}[/bold red]"
        )
    for esc in feedback.escalations:
        sev_color = {
            "blocking": "red",
            "degraded": "yellow",
            "informational": "dim",
        }.get(esc.severity.value, "white")
        console.print(
            f"  [{sev_color}][{esc.severity.value.upper()}] "
            f"{esc.type.value}:[/{sev_color}] {esc.detail}"
        )
    for obs in feedback.observations:
        console.print(f"  Observation: {obs}")
    for cb in feedback.challenged_beliefs:
        console.print(
            f"  [bold red]Challenges belief '{cb.belief_id}':[/bold red] {cb.evidence}"
        )
    if feedback.executor_recommendation:
        console.print(f"  Recommendation: {feedback.executor_recommendation}")
    if feedback.proposed_adjustments:
        console.print(f"  Proposed adjustments: {feedback.proposed_adjustments}")
    if feedback.execution_result:
        console.print(f"  Result: {feedback.execution_result}")
    console.print()


def _print_amendment(amendment: ThinkerAmendment) -> None:
    type_color = {
        "revise": "yellow",
        "clarify": "cyan",
        "delegate": "green",
        "abort": "red",
    }.get(amendment.amendment_type.value, "white")
    console.print(
        f"\n  [{type_color}]Amendment: {amendment.amendment_type.value.upper()}[/{type_color}]"
    )
    console.print(f"  Guidance: {amendment.guidance}")
    if amendment.updated_beliefs:
        console.print(f"  Updated {len(amendment.updated_beliefs)} belief(s)")
    if amendment.revised_steps:
        console.print(f"  Revised plan to {len(amendment.revised_steps)} steps")
    if amendment.continue_from_step is not None:
        console.print(f"  Continue from step: {amendment.continue_from_step}")
    console.print()
