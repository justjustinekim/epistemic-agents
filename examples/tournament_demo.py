#!/usr/bin/env python3
"""Demonstrate the tournament system comparing all 3 analysis tiers.

Runs quick, standard, and deep tiers on the same task and compares
whether the deeper analysis changes the verdict.
"""

import sys
import time

from epistemic_agents.config import get_available_providers
from epistemic_agents.orchestrator import Orchestrator
from epistemic_agents.panel import ModelPanel
from epistemic_agents.tournament import TournamentLog, run_tournament


def main() -> None:
    task = (
        "Should we migrate our monolithic Django application to microservices? "
        "We have 15 engineers, 500k monthly active users, and our deploy times "
        "have grown to 45 minutes."
    )

    print(f"Task: {task}\n")

    # Set up orchestrator with available providers
    providers = get_available_providers()
    panel = ModelPanel(providers) if len(providers) > 1 else None

    orch = Orchestrator(
        panel=panel,
        verbose=True,
    )

    # Run tournament
    print("Running tournament across all 3 tiers...\n")
    t0 = time.time()
    result = run_tournament(orch, task)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Tournament complete in {elapsed:.1f}s\n")

    for tier, verdict in result.results.items():
        print(f"  {tier.upper()}: {verdict[:120]}...")

    print(f"\n  Verdict changed across tiers: {result.verdict_changed}")
    print(f"  Confidence changed: {result.confidence_changed}")
    if result.deep_added_value:
        print(f"  Deep tier added: {result.deep_added_value[:200]}")

    # Persist result
    log = TournamentLog()
    log.record(result)
    print(f"\n{log.value_of_depth_summary()}")


if __name__ == "__main__":
    main()
