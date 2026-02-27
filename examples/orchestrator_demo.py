#!/usr/bin/env python3
"""Demo: run the tiered orchestrator in deep mode with all available models.

Usage:
    python examples/orchestrator_demo.py

Set API keys for more providers:
    export GOOGLE_API_KEY=...    # Gemini
    export XAI_API_KEY=...       # Grok
    export DEEPSEEK_API_KEY=...  # DeepSeek
    export DASHSCOPE_API_KEY=... # QwQ
    export OPENAI_API_KEY=...    # GPT
"""

from __future__ import annotations

import json
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from epistemic_agents.config import get_available_providers
from epistemic_agents.feedback import FeedbackLog, collect_feedback
from epistemic_agents.ledger import BeliefLedger
from epistemic_agents.panel import ModelPanel
from epistemic_agents.orchestrator import Orchestrator, Tier
from epistemic_agents.tracker import UsageTracker

console = Console()

TASK = """\
I've built an "epistemic agents" system — a multi-model AI thinking tool. It works like this: \
multiple AI models (Claude Opus, Gemini, Grok, GPT, DeepSeek, QwQ) analyze the same task in \
parallel as a "panel," then they debate each other's positions directly, and finally Claude \
Opus synthesizes the full debate to find agreements, tensions, blind spots, and unique insights.

The question: Will this system still be genuinely useful 3 months from now (mid-2026)?

Consider:
1. PACE OF MODEL IMPROVEMENT — Models are getting dramatically better every few months. \
If a single model in mid-2026 can do what this multi-model panel does today, the system \
is obsolete. How fast is the "single model capability frontier" advancing vs. the value \
of multi-model diversity?

2. LIFESTYLE/WORKFLOW FIT — This is a personal thinking tool for one person (a developer/builder). \
Is the overhead of maintaining API keys, running multi-model panels, and reading long synthesis \
outputs worth it vs. just asking one model a good question? When does "more perspectives" \
become "more noise"?

3. STRUCTURAL DURABILITY — Are the core ideas (epistemic protocols, belief falsification, \
cross-model synthesis) durable even if the specific implementation rots? Or is this \
solving a temporary problem that better models will dissolve entirely?

4. OPPORTUNITY COST — Time spent maintaining and evolving this tool is time not spent \
building other things. Is the ROI positive compared to alternatives?

Be brutally honest. If this is a clever toy that won't matter in 3 months, say so. \
If there's a core that survives, identify exactly what it is and what should be cut."""


def main() -> None:
    providers = get_available_providers()
    provider_names = [f"{p.name} ({p.model_id})" for p in providers]

    console.print(
        Panel(
            Text(TASK),
            title="[bold]Task[/bold]",
            border_style="white",
        )
    )

    console.print(f"\n[bold]Panel members:[/bold] {', '.join(provider_names)}")
    if len(providers) < 2:
        console.print(
            "[yellow]Note: Only one provider configured. Deep mode needs "
            "multiple providers for meaningful panel debate.[/yellow]\n"
        )

    # Set up components
    panel = ModelPanel(providers)
    ledger = BeliefLedger()
    tracker = UsageTracker()

    orchestrator = Orchestrator(
        panel=panel,
        ledger=ledger,
        thinker_model="opus",
        executor_model="sonnet",
        verdict_model="sonnet",
        verbose=True,
    )

    # Estimate tokens
    estimate = tracker.estimate_session_tokens(
        TASK, "deep", model_count=len(providers), rounds=3
    )
    console.print(f"[dim]Estimated tokens: ~{estimate:,}[/dim]\n")

    console.print("[bold magenta]Running Orchestrator in DEEP mode[/bold magenta]\n")
    console.print(
        "[dim]Pipeline: Panel Debate (3 rounds) → Synthesis → Refutation → "
        "Re-synthesis → Thinker-Executor Loop → Verdict[/dim]\n"
    )

    start = time.time()
    result = orchestrator.run(TASK, tier=Tier.DEEP)
    total_elapsed = time.time() - start

    # Print results
    console.print(f"\n{'='*80}")
    console.print(f"[bold green]ORCHESTRATOR COMPLETE[/bold green] — {total_elapsed:.1f}s total\n")

    # Panel synthesis summary
    if result.panel_synthesis:
        syn = result.panel_synthesis
        console.print("[bold cyan]Panel Synthesis Summary[/bold cyan]")

        if syn.agreements:
            console.print(f"\n  [bold green]Agreements ({len(syn.agreements)}):[/bold green]")
            for ag in syn.agreements:
                providers_str = ", ".join(ag.supporting_providers)
                console.print(f"    [{ag.combined_confidence.value}] {ag.claim}")
                console.print(f"      Supported by: {providers_str}")

        if syn.tensions:
            console.print(f"\n  [bold yellow]Tensions ({len(syn.tensions)}):[/bold yellow]")
            for t in syn.tensions:
                console.print(f"    {t.claim}")
                for provider, stance in t.positions.items():
                    console.print(f"      {provider}: {stance[:100]}...")
                console.print(f"      Synthesis: {t.synthesis_notes[:150]}...")

        if syn.blind_spots:
            console.print(f"\n  [bold red]Blind Spots ({len(syn.blind_spots)}):[/bold red]")
            for bs in syn.blind_spots:
                missed = ", ".join(bs.missed_by)
                console.print(f"    {bs.observation[:120]}...")
                console.print(f"      Caught by: {bs.identified_by} | Missed by: {missed}")

        if syn.unique_insights:
            console.print(f"\n  [bold magenta]Unique Insights ({len(syn.unique_insights)}):[/bold magenta]")
            for ui in syn.unique_insights:
                console.print(f"    [{ui.source_provider}] {ui.insight[:120]}...")

        console.print(f"\n  [bold cyan]Synthesized Strategy:[/bold cyan]")
        console.print(Panel(syn.synthesized_strategy, border_style="cyan"))

        console.print(f"\n  [bold]Meta-Confidence:[/bold] {syn.meta_confidence[:200]}...")

    # Thinker-executor loop summary
    if result.conversation_log:
        log = result.conversation_log
        console.print(
            f"\n[bold blue]Thinker-Executor Loop:[/bold blue] "
            f"{log.round_trips} rounds, converged: {log.converged}"
        )

    # Handoff summary
    if result.handoff:
        h = result.handoff
        console.print(f"\n[bold]Final Strategic Handoff:[/bold]")
        console.print(f"  Intent: {h.intent[:200]}...")
        console.print(f"  Beliefs: {len(h.beliefs)}")
        for b in h.beliefs:
            console.print(f"    [{b.confidence.value}] {b.id}: {b.claim[:100]}...")
        if h.meta_reasoning:
            console.print(f"  Meta-reasoning: {h.meta_reasoning[:200]}...")

    # THE VERDICT
    if result.verdict:
        v = result.verdict
        console.print(f"\n{'='*80}")
        console.print("[bold white on blue] VERDICT [/bold white on blue]\n")
        console.print(f"  [bold]Decision Point:[/bold] {v.decision_point}")
        console.print(f"\n  [bold green]Recommendation:[/bold green] {v.recommendation}")
        console.print(f"\n  [bold]Confidence:[/bold] {v.confidence.value}")
        console.print(f"  [bold red]Key Risk:[/bold red] {v.key_risk}")
        if v.dissent:
            console.print(f"  [bold yellow]Dissent:[/bold yellow] {v.dissent}")
        console.print(f"  [dim]Tier: {v.tier_used} | Tokens: {v.cost_tokens or 'N/A'}[/dim]")

    # Save full output
    output = {
        "tier": result.tier.value,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if result.verdict:
        output["verdict"] = result.verdict.model_dump(mode="json")
    if result.panel_synthesis:
        output["panel_synthesis"] = result.panel_synthesis.model_dump(mode="json")
    if result.handoff:
        output["handoff"] = result.handoff.model_dump(mode="json")
    if result.conversation_log:
        output["conversation_log"] = {
            "round_trips": result.conversation_log.round_trips,
            "converged": result.conversation_log.converged,
            "entries": len(result.conversation_log.entries),
        }

    output_path = "examples/orchestrator_demo_log.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"\n[dim]Full output saved to {output_path}[/dim]")
    console.print(f"[dim]Total time: {total_elapsed:.1f}s[/dim]")

    # Collect feedback
    fb = collect_feedback(
        task=TASK,
        tier="deep",
        model_count=len(providers),
        duration=total_elapsed,
    )
    if fb:
        fbl = FeedbackLog()
        fbl.add(fb)
        console.print(f"[dim]Feedback saved. {fbl.summary()}[/dim]")


if __name__ == "__main__":
    main()
