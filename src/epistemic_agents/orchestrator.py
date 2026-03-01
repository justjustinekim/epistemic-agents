"""Tiered orchestrator — routes tasks to quick / standard / deep analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from epistemic_agents import client
from epistemic_agents.executor import Executor
from epistemic_agents.ledger import BeliefLedger
from epistemic_agents.loop import EpistemicLoop
from epistemic_agents.schema import (
    ConversationLog,
    PanelSynthesis,
    StrategicHandoff,
    Verdict,
)
from epistemic_agents.synthesizer import Synthesizer
from epistemic_agents.thinker import Thinker

if TYPE_CHECKING:
    from epistemic_agents.feedback import FeedbackLog
    from epistemic_agents.panel import ModelPanel
    from epistemic_agents.tracker import UsageTracker


class Tier(str, Enum):
    """Analysis depth tiers."""

    QUICK = "quick"  # Single model, no loop, no panel
    STANDARD = "standard"  # Thinker-executor loop, no panel
    DEEP = "deep"  # Full panel debate + synthesis + refutation


@dataclass
class OrchestratorResult:
    """Unified result from any tier."""

    tier: Tier
    verdict: Verdict | None = None
    handoff: StrategicHandoff | None = None
    conversation_log: ConversationLog | None = None
    panel_synthesis: PanelSynthesis | None = None
    elapsed_seconds: float = 0.0
    token_estimate: int = 0


class Orchestrator:
    """Routes tasks to the appropriate analysis tier.

    Tiers:
    - quick: Single Claude call. Fast, cheap, good for simple questions.
    - standard: Thinker-executor loop. Good for tasks needing iteration.
    - deep: Full multi-model panel debate. For high-stakes decisions.
    """

    def __init__(
        self,
        panel: ModelPanel | None = None,
        ledger: BeliefLedger | None = None,
        thinker_model: str = "opus",
        executor_model: str = "sonnet",
        verdict_model: str = "haiku",
        verbose: bool = True,
        feedback_log: FeedbackLog | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self._panel = panel
        self._ledger = ledger
        self._thinker_model = thinker_model
        self._executor_model = executor_model
        self._verdict_model = verdict_model
        self._verbose = verbose
        self._feedback_log = feedback_log
        self._tracker = tracker

    def run(self, task: str, tier: Tier | str = Tier.STANDARD) -> OrchestratorResult:
        """Run the appropriate tier and return a unified result."""
        if isinstance(tier, str):
            tier = Tier(tier)

        start = time.time()

        if tier == Tier.QUICK:
            result = self._run_quick(task)
        elif tier == Tier.STANDARD:
            result = self._run_standard(task)
        elif tier == Tier.DEEP:
            result = self._run_deep(task)
        else:
            raise ValueError(f"Unknown tier: {tier}")

        result.elapsed_seconds = time.time() - start
        return result

    def auto_route(self, task: str) -> OrchestratorResult:
        """Automatically choose a tier based on task complexity.

        Heuristics:
        - Short tasks (< 200 chars) with simple structure -> quick
        - Tasks mentioning stakes, tradeoffs, strategy -> deep
        - Everything else -> standard
        """
        tier = self._classify_task(task)
        return self.run(task, tier)

    def _classify_task(self, task: str) -> Tier:
        """Simple heuristic classification of task complexity."""
        task_lower = task.lower()
        length = len(task)

        # Deep indicators
        deep_signals = [
            "tradeoff", "trade-off", "strategy", "architecture",
            "high-stakes", "critical", "multi-model", "debate",
            "compare", "evaluate options", "pros and cons",
            "long-term", "investment", "risk",
        ]
        deep_count = sum(1 for s in deep_signals if s in task_lower)

        # Quick indicators
        quick_signals = [
            "what is", "how do i", "explain", "define",
            "quick", "simple", "just",
        ]
        quick_count = sum(1 for s in quick_signals if s in task_lower)

        if deep_count >= 2 or length > 1000:
            return Tier.DEEP
        if quick_count >= 1 and length < 200:
            return Tier.QUICK
        return Tier.STANDARD

    def _run_quick(self, task: str) -> OrchestratorResult:
        """Quick tier: single thinker call, no loop, no panel."""
        thinker = Thinker(model=self._thinker_model)
        handoff = thinker.analyze(task)

        # Record usage to tracker if available
        if self._tracker:
            for cu in thinker.cost.calls:
                self._tracker.record_usage("claude", cu.model, cu.input_tokens, cu.output_tokens)

        # Generate verdict directly from handoff
        total_tokens = thinker.cost.total_input_tokens + thinker.cost.total_output_tokens
        total_cost = thinker.cost.total_cost_usd
        verdict = generate_verdict(
            task=task,
            tier="quick",
            handoff=handoff,
            model=self._verdict_model,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

        return OrchestratorResult(
            tier=Tier.QUICK,
            verdict=verdict,
            handoff=handoff,
        )

    def _run_standard(self, task: str) -> OrchestratorResult:
        """Standard tier: thinker-executor loop with optional ledger."""
        from epistemic_agents.rag import build_rag_context

        calibration = build_rag_context(
            task=task,
            ledger=self._ledger,
            feedback_log=self._feedback_log,
            tracker=self._tracker,
        )
        # Fall back to plain calibration if RAG returned nothing
        if not calibration and self._ledger:
            calibration = self._ledger.calibration_context()

        thinker = Thinker(
            model=self._thinker_model,
            calibration_context=calibration,
        )
        executor = Executor(model=self._executor_model)
        loop = EpistemicLoop(
            thinker=thinker,
            executor=executor,
            verbose=self._verbose,
            ledger=self._ledger,
        )

        log = loop.run(task)

        # Record usage to tracker if available
        if self._tracker:
            for cu in thinker.cost.calls:
                self._tracker.record_usage("claude", cu.model, cu.input_tokens, cu.output_tokens)
            for cu in executor.cost.calls:
                self._tracker.record_usage("claude", cu.model, cu.input_tokens, cu.output_tokens)

        # Get the final handoff state
        handoff = _extract_final_handoff(log)

        total_tokens = (
            thinker.cost.total_input_tokens + thinker.cost.total_output_tokens
            + executor.cost.total_input_tokens + executor.cost.total_output_tokens
        )
        total_cost = thinker.cost.total_cost_usd + executor.cost.total_cost_usd
        verdict = generate_verdict(
            task=task,
            tier="standard",
            handoff=handoff,
            conversation_log=log,
            model=self._verdict_model,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

        return OrchestratorResult(
            tier=Tier.STANDARD,
            verdict=verdict,
            handoff=handoff,
            conversation_log=log,
        )

    def _run_deep(self, task: str) -> OrchestratorResult:
        """Deep tier: full panel debate + synthesis + refutation, then thinker loop."""
        import sys
        from epistemic_agents.panel import ModelPanel
        from epistemic_agents.rag import build_rag_context

        if not self._panel:
            # Fall back to standard if no panel configured
            return self._run_standard(task)

        def _log(msg: str) -> None:
            if self._verbose:
                print(msg, file=sys.stderr)

        # Build RAG context for the panel's initial round
        rag_context = build_rag_context(
            task=task,
            ledger=self._ledger,
            feedback_log=self._feedback_log,
            tracker=self._tracker,
        )
        initial_prompt = None
        if rag_context:
            from epistemic_agents.panel import PANEL_SYSTEM_PROMPT
            initial_prompt = (
                f"{PANEL_SYSTEM_PROMPT}\n\n"
                f"--- CONTEXT FROM PAST SESSIONS ---\n{rag_context}"
            )

        # Phase 1: Panel debate
        _log("[deep] Phase 1: Multi-round panel debate...")
        t0 = time.time()

        def _on_round(round_num: int, positions: list) -> None:
            label = "Initial Analysis" if round_num == 1 else f"Debate Round {round_num - 1}"
            names = [p.provider_name for p in positions]
            _log(f"[deep]   {label} — {len(positions)} responses: {', '.join(names)}")

        rounds = self._panel.debate(
            task=task, rounds=3, on_round=_on_round,
            initial_system_prompt=initial_prompt,
        )
        if not any(rounds):
            return self._run_standard(task)
        total_responses = sum(len(r) for r in rounds)
        _log(f"[deep]   Debate complete: {len(rounds)} rounds, {total_responses} responses in {time.time()-t0:.1f}s")

        # Phase 2: Synthesize
        _log("[deep] Phase 2: Cross-model synthesis...")
        t0 = time.time()
        synthesizer = Synthesizer(model=self._thinker_model)
        synthesis = synthesizer.synthesize_debate(task, rounds)
        _log(f"[deep]   Synthesis complete in {time.time()-t0:.1f}s")

        # Phase 3: Refutation
        _log("[deep] Phase 3: Panel refutation...")
        t0 = time.time()
        refutations = self._panel.refute(task, rounds, synthesis)
        _log(f"[deep]   {len(refutations)} refutations in {time.time()-t0:.1f}s")

        # Phase 4: Re-synthesis
        _log("[deep] Phase 4: Final re-synthesis (post-refutation)...")
        t0 = time.time()
        final_synthesis = synthesizer.resynthesize(task, rounds, synthesis, refutations)
        _log(f"[deep]   Re-synthesis complete in {time.time()-t0:.1f}s")

        # Phase 5: Generate verdict directly from panel synthesis
        # The panel debate + synthesis + refutation + re-synthesis IS the deep analysis.
        # Running an additional thinker-executor loop is redundant and risks context overflow.
        _log("[deep] Phase 5: Generating verdict...")
        t0 = time.time()
        verdict = generate_verdict(
            task=task,
            tier="deep",
            panel_synthesis=final_synthesis,
            model=self._verdict_model,
        )
        _log(f"[deep]   Verdict generated in {time.time()-t0:.1f}s")

        # Record usage to tracker if available
        if self._tracker:
            # Deep tier doesn't use thinker/executor directly, but we can
            # record the verdict generation cost from the thread-local
            from epistemic_agents.client import get_last_usage
            verdict_usage = get_last_usage()
            if verdict_usage:
                self._tracker.record_usage(
                    "claude", verdict_usage.model,
                    verdict_usage.input_tokens, verdict_usage.output_tokens,
                )

        return OrchestratorResult(
            tier=Tier.DEEP,
            verdict=verdict,
            panel_synthesis=final_synthesis,
        )


# ---------------------------------------------------------------------------
# Verdict generation (Step 4)
# ---------------------------------------------------------------------------

VERDICT_SYSTEM = """\
You are a verdict generator. You take a full strategic analysis and distill it \
into a concise, decision-oriented summary.

Your output must be actionable — a busy person should be able to read ONLY the \
verdict and know what to do. Be opinionated. Don't hedge.

Focus on:
- What decision the user actually faces (stated as a question)
- Your recommendation (1-3 sentences, concrete and specific)
- The single biggest risk if they follow the recommendation
- The strongest counterargument (dissent), if one exists
- Your confidence level in the recommendation\
"""


def generate_verdict(
    task: str,
    tier: str,
    handoff: StrategicHandoff | None = None,
    conversation_log: ConversationLog | None = None,
    panel_synthesis: PanelSynthesis | None = None,
    model: str = "haiku",
    total_tokens: int | None = None,
    total_cost_usd: float | None = None,
) -> Verdict:
    """Distill a full analysis into a concise Verdict using a cheap model."""
    sections = [f"# Original Task\n{task}\n"]

    if handoff:
        sections.append(f"# Strategic Analysis\n")
        sections.append(f"Intent: {handoff.intent}\n")
        for b in handoff.beliefs:
            sections.append(
                f"- [{b.confidence.value}] {b.id}: {b.claim}\n"
            )
        if handoff.meta_reasoning:
            sections.append(f"\nMeta-reasoning: {handoff.meta_reasoning}\n")

    if conversation_log and conversation_log.round_trips > 0:
        sections.append(
            f"\n# Execution Summary\n"
            f"Rounds: {conversation_log.round_trips}, "
            f"Converged: {conversation_log.converged}\n"
        )

    if panel_synthesis:
        sections.append(f"\n# Panel Synthesis\n")
        sections.append(f"Strategy: {panel_synthesis.synthesized_strategy}\n")
        sections.append(f"Confidence: {panel_synthesis.meta_confidence}\n")
        if panel_synthesis.tensions:
            sections.append("Key tensions:\n")
            for t in panel_synthesis.tensions:
                sections.append(f"  - {t.claim}: {t.synthesis_notes}\n")

    sections.append(
        f"\n---\nDistill the above into a verdict. Tier used: {tier}"
    )

    verdict = client.structured_request(
        model=model,
        system=VERDICT_SYSTEM,
        user_message="".join(sections),
        response_model=Verdict,
    )
    verdict.tier_used = tier
    if total_tokens is not None:
        verdict.cost_tokens = total_tokens
    if total_cost_usd is not None:
        verdict.cost_usd = total_cost_usd
    return verdict


def _extract_final_handoff(log: ConversationLog) -> StrategicHandoff | None:
    """Extract the most recent StrategicHandoff from a conversation log."""
    from epistemic_agents.loop import _apply_amendment

    handoff = None
    for entry in log.entries:
        if entry.entry_type == "handoff" and isinstance(
            entry.content, StrategicHandoff
        ):
            handoff = entry.content
        elif entry.entry_type == "amendment" and handoff:
            from epistemic_agents.schema import ThinkerAmendment

            if isinstance(entry.content, ThinkerAmendment):
                handoff = _apply_amendment(handoff, entry.content)
    return handoff
