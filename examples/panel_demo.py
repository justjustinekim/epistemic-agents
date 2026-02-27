#!/usr/bin/env python3
"""Demo: multi-model debate on a strategic task.

Models analyze independently, then respond to each other's positions directly,
then a synthesizer distills the full debate into a structured output.

Usage:
    python examples/panel_demo.py

Set API keys for more providers (Claude is always available):
    export GOOGLE_API_KEY=...    # Gemini
    export XAI_API_KEY=...       # Grok
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
from epistemic_agents.panel import ModelPanel
from epistemic_agents.synthesizer import Synthesizer
from epistemic_agents.schema import PanelSynthesis, ProviderPosition

console = Console()

TASK = """\
I've built an "epistemic agents" system — a multi-model AI thinking tool. It works like this: \
multiple AI models (Claude Opus, Gemini, Grok, GPT) analyze the same task in parallel as a \
"panel," then they debate each other's positions directly, and finally Claude Opus synthesizes \
the full debate to find agreements, tensions, blind spots, and unique insights.

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


def print_round(round_num: int, positions: list[ProviderPosition]) -> None:
    """Print a round's positions as they come in."""
    label = "Initial Analysis" if round_num == 1 else f"Debate Round {round_num - 1}"
    console.print(f"\n  [bold]{label}[/bold] — {len(positions)} responses")
    for pos in positions:
        border = "blue" if round_num == 1 else "yellow"
        console.print(
            Panel(
                pos.raw_analysis,
                title=f"[bold]{pos.provider_name}[/bold] ({pos.model_id})",
                border_style=border,
            )
        )
        console.print()


def print_synthesis(synthesis: PanelSynthesis) -> None:
    """Print the structured synthesis."""
    if synthesis.agreements:
        console.print("\n[bold green]Agreements[/bold green]")
        for ag in synthesis.agreements:
            providers = ", ".join(ag.supporting_providers)
            console.print(f"  [{ag.combined_confidence.value}] {ag.claim}")
            console.print(f"    Supported by: {providers}")

    if synthesis.tensions:
        console.print("\n[bold yellow]Tensions[/bold yellow]")
        for t in synthesis.tensions:
            console.print(f"  {t.claim}")
            for provider, stance in t.positions.items():
                console.print(f"    {provider}: {stance}")
            console.print(f"    Synthesis: {t.synthesis_notes}")

    if synthesis.blind_spots:
        console.print("\n[bold red]Blind Spots[/bold red]")
        for bs in synthesis.blind_spots:
            missed = ", ".join(bs.missed_by)
            console.print(f"  {bs.observation}")
            console.print(f"    Caught by: {bs.identified_by} | Missed by: {missed}")

    if synthesis.unique_insights:
        console.print("\n[bold magenta]Unique Insights[/bold magenta]")
        for ui in synthesis.unique_insights:
            console.print(f"  [{ui.source_provider}] {ui.insight}")
            console.print(f"    Relevance: {ui.relevance}")

    console.print("\n[bold cyan]Synthesized Strategy[/bold cyan]")
    console.print(Panel(synthesis.synthesized_strategy, border_style="cyan"))

    console.print(f"\n[bold]Meta-Confidence:[/bold] {synthesis.meta_confidence}")


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
            "[yellow]Note: Only one provider configured. Add API keys for "
            "GOOGLE_API_KEY, XAI_API_KEY, or OPENAI_API_KEY to get diverse "
            "perspectives from multiple model families.[/yellow]\n"
        )

    # Phase 1: Multi-round debate
    console.print("\n[bold]Phase 1: Multi-Model Debate[/bold]")
    panel = ModelPanel(providers)
    start = time.time()
    rounds = panel.debate(task=TASK, rounds=3, on_round=print_round)
    elapsed = time.time() - start
    total_responses = sum(len(r) for r in rounds)
    console.print(
        f"\n  [dim]Debate complete: {len(rounds)} rounds, "
        f"{total_responses} total responses in {elapsed:.1f}s[/dim]"
    )

    if not any(rounds):
        console.print("[red]No providers returned results. Exiting.[/red]")
        sys.exit(1)

    # Phase 2: Initial synthesis
    console.print("\n[bold]Phase 2: Cross-Model Synthesis[/bold]")
    synthesizer = Synthesizer()
    start = time.time()
    synthesis = synthesizer.synthesize_debate(TASK, rounds)
    elapsed = time.time() - start
    console.print(f"  Initial synthesis in {elapsed:.1f}s\n")

    print_synthesis(synthesis)

    # Phase 3: Panel refutation — consultants challenge the synthesis
    console.print("\n[bold]Phase 3: Panel Refutation[/bold]")
    start = time.time()
    refutations = panel.refute(TASK, rounds, synthesis)
    elapsed = time.time() - start
    console.print(f"  {len(refutations)} refutations in {elapsed:.1f}s")
    for pos in refutations:
        console.print(
            Panel(
                pos.raw_analysis,
                title=f"[bold]{pos.provider_name}[/bold] ({pos.model_id}) — Refutation",
                border_style="red",
            )
        )
        console.print()

    # Phase 4: Re-synthesis incorporating refutations
    console.print("\n[bold]Phase 4: Final Synthesis (post-refutation)[/bold]")
    start = time.time()
    final_synthesis = synthesizer.resynthesize(TASK, rounds, synthesis, refutations)
    elapsed = time.time() - start
    console.print(f"  Final synthesis in {elapsed:.1f}s\n")

    print_synthesis(final_synthesis)

    # Save full output
    output_path = "examples/panel_demo_log.json"
    with open(output_path, "w") as f:
        json.dump(final_synthesis.model_dump(mode="json"), f, indent=2, default=str)
    console.print(f"\n[dim]Full output saved to {output_path}[/dim]")

    # Collect feedback
    total_time = time.time() - start
    fb = collect_feedback(
        task=TASK,
        tier="deep",
        model_count=len(providers),
        duration=total_time,
    )
    if fb:
        log = FeedbackLog()
        log.add(fb)
        console.print(f"[dim]Feedback saved. {log.summary()}[/dim]")


if __name__ == "__main__":
    main()
