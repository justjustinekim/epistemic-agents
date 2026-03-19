#!/usr/bin/env python3
"""Run empirical eval: panel vs adversarial vs persona on real tasks.

Usage:
    uv run python examples/run_empirical_eval.py
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from epistemic_agents.config import get_available_providers
from epistemic_agents.empirical_eval import run_empirical_eval

console = Console()

# Real tasks that exercise different reasoning demands
EVAL_TASKS = [
    "Should a startup use microservices or a monolith for their MVP? They have 3 engineers, expect 10x growth in 12 months, and need to ship in 6 weeks.",
    "Is it worth migrating a production PostgreSQL database to a distributed NewSQL system (CockroachDB/TiDB) for a SaaS product serving 50K users with 99.9% uptime requirements?",
    "A trading bot has a 15% win rate but winners pay 8:1. Is this strategy profitable long-term? What position sizing should be used? When should it be killed?",
    "Should an AI coding assistant use RAG over the codebase, fine-tuning on the codebase, or long-context window stuffing? The codebase is 500K lines of Python.",
    "Is it safe to use LLM-generated code in production without human review for low-risk internal tools? What guardrails are needed?",
]


def main() -> None:
    console.print(Panel("[bold]Empirical Evaluation[/bold]\nPanel vs Adversarial vs Persona"))

    providers = get_available_providers()
    available = [p for p in providers if p.available]
    console.print(f"{len(available)} providers available\n")
    console.print(f"Running {len(EVAL_TASKS)} tasks x 3 approaches = {len(EVAL_TASKS) * 3} evaluations\n")

    report = run_empirical_eval(EVAL_TASKS, available)
    console.print(report.format_report())
    console.print("\n[green]Results saved to results/empirical_eval.json[/green]")


if __name__ == "__main__":
    main()
