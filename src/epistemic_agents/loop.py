"""Feedback loop controller — orchestrates the thinker-executor conversation."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from epistemic_agents.executor import Executor
from epistemic_agents.schema import (
    AmendmentType,
    ConversationLog,
    ExecutorFeedback,
    StrategicHandoff,
    ThinkerAmendment,
)
from epistemic_agents.thinker import Thinker

console = Console()


class EpistemicLoop:
    """Manages the epistemic feedback loop between thinker and executor."""

    def __init__(
        self,
        thinker: Thinker,
        executor: Executor,
        max_rounds: int = 5,
        verbose: bool = True,
    ):
        self.thinker = thinker
        self.executor = executor
        self.max_rounds = max_rounds
        self.verbose = verbose

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

            # Check if we've converged (no escalation needed)
            if not feedback.decision_needed and feedback.escalation_type is None:
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

            # Update the handoff if steps were revised
            if amendment.revised_steps:
                current_handoff = StrategicHandoff(
                    intent=current_handoff.intent,
                    beliefs=amendment.updated_beliefs or current_handoff.beliefs,
                    plan_steps=amendment.revised_steps,
                    decision_boundaries=current_handoff.decision_boundaries,
                    open_questions=current_handoff.open_questions,
                )

        else:
            if self.verbose:
                console.print(
                    Panel(
                        f"MAX ROUNDS ({self.max_rounds}) reached without convergence.",
                        style="bold yellow",
                    )
                )

        return log


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
    console.print()


def _print_feedback(feedback: ExecutorFeedback) -> None:
    if feedback.step_completed is not None:
        console.print(f"\n  Completed through step: {feedback.step_completed}")
    if feedback.escalation_type:
        console.print(
            f"  [bold red]ESCALATION: {feedback.escalation_type.value}[/bold red]"
        )
    for obs in feedback.observations:
        console.print(f"  Observation: {obs}")
    for cb in feedback.challenged_beliefs:
        console.print(
            f"  [bold red]Challenges belief '{cb.belief_id}':[/bold red] {cb.evidence}"
        )
    if feedback.executor_recommendation:
        console.print(f"  Recommendation: {feedback.executor_recommendation}")
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
