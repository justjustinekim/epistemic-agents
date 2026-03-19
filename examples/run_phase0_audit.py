#!/usr/bin/env python3
"""Phase 0: Run consensus audit across all available providers.

Measures how often majority consensus is wrong using calibration tasks
with known ground truth. Outputs the data needed to set C1 vote-locking
thresholds and decide on C2 viability.

Usage:
    uv run python examples/run_phase0_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from epistemic_agents.config import get_available_providers, provider_credit_status
from epistemic_agents.calibration_games import CALIBRATION_TASKS
from epistemic_agents.consensus_audit import run_consensus_audit

console = Console()


def main() -> None:
    console.print(Panel("[bold]Phase 0: Consensus Audit[/bold]\nMeasuring majority error frequency across calibration tasks"))
    console.print()

    # Show provider status
    console.print(provider_credit_status())
    console.print()

    providers = get_available_providers()
    available = [p for p in providers if p.available]
    console.print(f"[green]{len(available)}[/green] providers available: {[p.name for p in available]}")
    console.print(f"[blue]{len(CALIBRATION_TASKS)}[/blue] calibration tasks with known ground truth")
    console.print()

    if len(available) < 3:
        console.print("[red]Need at least 3 providers for meaningful consensus analysis.[/red]")
        sys.exit(1)

    start = time.time()
    console.print("[bold]Running calibration games...[/bold]")
    console.print()

    report = run_consensus_audit(available, CALIBRATION_TASKS)

    elapsed = time.time() - start
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]\n")

    # Print the full report
    console.print(report.format_report())

    # Save to JSON for downstream consumption
    output_path = Path("results/phase0_consensus_audit.json")
    output_path.parent.mkdir(exist_ok=True)

    audit_data = {
        "n_providers": report.n_providers,
        "n_tasks": report.n_tasks,
        "provider_accuracy": report.provider_accuracy,
        "provider_brier": report.provider_brier,
        "threshold_stats": {
            str(k): {"correct": v[0], "total_locked": v[1], "accuracy": v[2]}
            for k, v in report.threshold_stats.items()
        },
        "majority_error_rate": report.majority_error_rate,
        "unanimous_error_rate": report.unanimous_error_rate,
        "optimal_threshold": report.optimal_threshold,
        "per_task": [
            {
                "question": r.task.question,
                "ground_truth": r.task.ground_truth,
                "category": r.task.category,
                "difficulty": r.task.difficulty,
                "majority_answer": r.majority_answer,
                "majority_correct": r.majority_correct,
                "agreement_count": r.agreement_count,
                "consensus_confidence": r.consensus_confidence,
                "providers": [
                    {
                        "name": pr.provider_name,
                        "confidence": pr.predicted_confidence,
                        "correct": pr.correct,
                        "brier": pr.brier_score,
                    }
                    for pr in r.provider_results
                ],
            }
            for r in report.results
        ],
        "elapsed_seconds": elapsed,
    }

    output_path.write_text(json.dumps(audit_data, indent=2))
    console.print(f"\n[green]Report saved to {output_path}[/green]")

    # Print actionable summary
    console.print()
    console.print(Panel(
        f"[bold]C1 Threshold Recommendation:[/bold] >= {report.optimal_threshold} providers\n"
        f"[bold]Majority error rate:[/bold] {report.majority_error_rate:.1%}\n"
        f"[bold]Unanimous error rate:[/bold] {report.unanimous_error_rate:.1%}\n\n"
        "Use these numbers to calibrate vote-then-debate lock thresholds.",
        title="Phase 0 Findings",
    ))


if __name__ == "__main__":
    main()
