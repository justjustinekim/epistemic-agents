#!/usr/bin/env python3
"""Demonstrate virtual panelists wrapping a single provider with cognitive roles.

Creates 4 virtual panelists from a ClaudeProvider, each with a different
adversarial cognitive role, and runs a panel debate.
"""

import sys
import time

from epistemic_agents.providers.claude import ClaudeProvider
from epistemic_agents.providers.virtual import create_virtual_panelists
from epistemic_agents.panel import ModelPanel
from epistemic_agents.synthesizer import Synthesizer


def main() -> None:
    task = (
        "Should a startup with 5 engineers build their own auth system "
        "or use a managed service like Auth0/Clerk? They have a tight "
        "6-month runway and their product requires enterprise SSO."
    )

    print(f"Task: {task}\n")
    print("Creating virtual panelists from Claude...\n")

    # Create a Claude provider and wrap it with all 4 cognitive roles
    base_provider = ClaudeProvider()
    panelists = create_virtual_panelists(base_provider)

    for p in panelists:
        print(f"  - {p.name} ({p.model_id})")

    # Create panel and run debate
    panel = ModelPanel(panelists, extract_beliefs=True, extraction_model="haiku")

    print(f"\nStarting 2-round debate with {len(panelists)} virtual panelists...\n")
    t0 = time.time()

    def on_round(round_num: int, positions: list) -> None:
        label = "Initial Analysis" if round_num == 1 else f"Debate Round {round_num - 1}"
        names = [p.provider_name for p in positions]
        print(f"  {label}: {len(positions)} responses from {', '.join(names)}")
        for pos in positions:
            print(f"    {pos.provider_name}: {len(pos.beliefs)} beliefs extracted")

    rounds = panel.debate(task, rounds=2, on_round=on_round)
    elapsed = time.time() - t0
    print(f"\nDebate complete in {elapsed:.1f}s")

    # Synthesize
    print("\nSynthesizing...")
    synth = Synthesizer(model="opus")
    synthesis = synth.synthesize_debate(task, rounds)

    print(f"\nAgreements: {len(synthesis.agreements)}")
    for ag in synthesis.agreements:
        print(f"  - [{ag.combined_confidence.value}] {ag.claim}")

    print(f"\nTensions: {len(synthesis.tensions)}")
    for t in synthesis.tensions:
        print(f"  - {t.claim}: {t.synthesis_notes[:80]}...")

    print(f"\nStrategy: {synthesis.synthesized_strategy[:200]}...")
    print(f"\nMeta-Confidence: {synthesis.meta_confidence[:200]}...")


if __name__ == "__main__":
    main()
