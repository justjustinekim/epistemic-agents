#!/usr/bin/env python3
"""Analyze debate checkpoints and results for consensus patterns.

Extracts provider positions from all available debate data and measures:
- Agreement rates at different thresholds
- Pairwise provider correlation
- Belief distribution per provider
- Where minority positions persisted

Usage:
    uv run python examples/analyze_checkpoints.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from epistemic_agents.schema import ProviderPosition
from epistemic_agents.agreement_detector import detect_agreements, detect_tensions
from epistemic_agents.rag import _tokenize, _jaccard_similarity

console = Console()


def load_positions_from_file(path: Path) -> list[tuple[str, list[ProviderPosition]]]:
    """Load provider positions from a debate file. Returns list of (source_label, positions)."""
    data = json.loads(path.read_text())
    results: list[tuple[str, list[ProviderPosition]]] = []
    name = path.stem

    # Checkpoint format: {task, phase, rounds: [{provider_name, beliefs, ...}, ...], ...}
    if "rounds" in data and isinstance(data["rounds"], list):
        rounds = data["rounds"]
        # Rounds can be a flat list of positions or a list of lists (one per round)
        if rounds and isinstance(rounds[0], dict) and "provider_name" in rounds[0]:
            # Flat list of positions (single round stored)
            positions = _parse_positions(rounds)
            if positions:
                results.append((f"{name}:positions", positions))
        elif rounds and isinstance(rounds[0], list):
            # List of rounds, each containing positions
            for i, round_data in enumerate(rounds):
                positions = _parse_positions(round_data)
                if positions:
                    results.append((f"{name}:round{i + 1}", positions))

    # Result/demo format: {panel_synthesis: {provider_positions: [...]}}
    if "panel_synthesis" in data:
        synth = data["panel_synthesis"]
        if "provider_positions" in synth:
            positions = _parse_positions(synth["provider_positions"])
            if positions:
                results.append((f"{name}:synthesis", positions))

    # Simple format: {agreements, tensions, provider_positions at top level}
    if "provider_positions" in data and "panel_synthesis" not in data:
        positions = _parse_positions(data["provider_positions"])
        if positions:
            results.append((f"{name}:positions", positions))

    return results


def _parse_positions(raw_positions: list[dict]) -> list[ProviderPosition]:
    """Parse raw dicts into ProviderPosition objects."""
    positions: list[ProviderPosition] = []
    for raw in raw_positions:
        if not isinstance(raw, dict):
            continue
        if "provider_name" not in raw or "beliefs" not in raw:
            continue
        try:
            pos = ProviderPosition.model_validate(raw)
            if pos.beliefs:
                positions.append(pos)
        except Exception:
            continue
    return positions


def analyze_positions(label: str, positions: list[ProviderPosition]) -> dict:
    """Analyze a set of positions for consensus patterns."""
    n_providers = len(positions)
    beliefs_per_provider = {p.provider_name: len(p.beliefs) for p in positions}
    total_beliefs = sum(beliefs_per_provider.values())

    # Detect agreements and tensions
    agreements = detect_agreements(positions, similarity_threshold=0.4)
    tensions = detect_tensions(positions)

    # Agreement by threshold
    threshold_counts: dict[int, int] = {}
    for thresh in range(2, n_providers + 1):
        threshold_counts[thresh] = sum(
            1 for a in agreements if len(a.supporting_providers) >= thresh
        )

    # Pairwise similarity matrix
    pair_agreements: dict[str, int] = {}
    pair_total: dict[str, int] = {}
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pa = positions[i]
            pb = positions[j]
            key = "|".join(sorted([pa.provider_name, pb.provider_name]))
            agreed = 0
            compared = 0
            for ba in pa.beliefs:
                tokens_a = _tokenize(ba.claim)
                for bb in pb.beliefs:
                    tokens_b = _tokenize(bb.claim)
                    sim = _jaccard_similarity(tokens_a, tokens_b)
                    if sim > 0.3:
                        compared += 1
                        gap = abs(ba.effective_score - bb.effective_score)
                        if gap < 0.2:
                            agreed += 1
            pair_agreements[key] = pair_agreements.get(key, 0) + agreed
            pair_total[key] = pair_total.get(key, 0) + max(compared, 1)

    pair_rates = {
        k: pair_agreements[k] / pair_total[k] for k in pair_total if pair_total[k] > 0
    }

    return {
        "label": label,
        "n_providers": n_providers,
        "beliefs_per_provider": beliefs_per_provider,
        "total_beliefs": total_beliefs,
        "n_agreements": len(agreements),
        "n_tensions": len(tensions),
        "threshold_counts": threshold_counts,
        "pair_rates": pair_rates,
        "agreements": agreements,
        "tensions": tensions,
    }


def main() -> None:
    console.print("[bold]Debate Checkpoint & Result Analysis[/bold]\n")

    # Collect all files
    files: list[Path] = []
    checkpoint_dir = Path(".epistemic_checkpoints")
    results_dir = Path("results")
    examples_dir = Path("examples")

    if checkpoint_dir.exists():
        files.extend(sorted(checkpoint_dir.glob("*.json")))
    if results_dir.exists():
        files.extend(sorted(results_dir.glob("*.json")))
    for name in ["orchestrator_demo_log.json", "evaluate_improvements_log.json"]:
        p = examples_dir / name
        if p.exists():
            files.append(p)

    console.print(f"Found {len(files)} debate files\n")

    all_analyses: list[dict] = []

    for f in files:
        try:
            sources = load_positions_from_file(f)
            for label, positions in sources:
                if len(positions) >= 2:
                    analysis = analyze_positions(label, positions)
                    all_analyses.append(analysis)
        except Exception as e:
            console.print(f"[red]Error processing {f.name}: {e}[/red]")

    if not all_analyses:
        console.print("[red]No analyzable debate data found.[/red]")
        return

    # Summary table
    table = Table(title="Consensus Patterns Across Debates")
    table.add_column("Source", style="cyan")
    table.add_column("Providers", justify="right")
    table.add_column("Beliefs", justify="right")
    table.add_column("Agreements", justify="right")
    table.add_column("Tensions", justify="right")
    table.add_column(">=2 agree", justify="right")
    table.add_column(">=4 agree", justify="right")
    table.add_column(">=6 agree", justify="right")
    table.add_column("Unanimous", justify="right")

    for a in all_analyses:
        n = a["n_providers"]
        tc = a["threshold_counts"]
        table.add_row(
            a["label"][:45],
            str(n),
            str(a["total_beliefs"]),
            str(a["n_agreements"]),
            str(a["n_tensions"]),
            str(tc.get(2, 0)),
            str(tc.get(4, 0)),
            str(tc.get(6, 0)),
            str(tc.get(n, 0)),
        )

    console.print(table)

    # Aggregate pairwise correlation
    console.print("\n[bold]Pairwise Provider Agreement Rates (aggregated)[/bold]")
    agg_pair_agree: dict[str, float] = {}
    agg_pair_count: dict[str, int] = {}

    for a in all_analyses:
        for pair, rate in a["pair_rates"].items():
            agg_pair_agree[pair] = agg_pair_agree.get(pair, 0.0) + rate
            agg_pair_count[pair] = agg_pair_count.get(pair, 0) + 1

    pair_table = Table(title="Pairwise Agreement")
    pair_table.add_column("Provider Pair", style="cyan")
    pair_table.add_column("Avg Agreement", justify="right")
    pair_table.add_column("Samples", justify="right")

    sorted_pairs = sorted(
        agg_pair_agree.keys(),
        key=lambda k: agg_pair_agree[k] / agg_pair_count[k],
        reverse=True,
    )
    for pair in sorted_pairs[:20]:
        avg = agg_pair_agree[pair] / agg_pair_count[pair]
        pair_table.add_row(
            pair.replace("|", " <-> "),
            f"{avg:.1%}",
            str(agg_pair_count[pair]),
        )

    console.print(pair_table)

    # Aggregate threshold analysis
    console.print("\n[bold]Agreement Threshold Analysis (all debates combined)[/bold]")
    total_agreements = sum(a["n_agreements"] for a in all_analyses)
    for thresh in range(2, 8):
        count = sum(a["threshold_counts"].get(thresh, 0) for a in all_analyses)
        if total_agreements > 0:
            pct = count / total_agreements * 100
            console.print(
                f"  >= {thresh} providers: {count}/{total_agreements} agreements ({pct:.0f}%)"
            )

    # Per-provider belief count
    console.print("\n[bold]Per-Provider Belief Counts (all debates)[/bold]")
    provider_totals: dict[str, int] = {}
    provider_appearances: dict[str, int] = {}
    for a in all_analyses:
        for prov, count in a["beliefs_per_provider"].items():
            provider_totals[prov] = provider_totals.get(prov, 0) + count
            provider_appearances[prov] = provider_appearances.get(prov, 0) + 1

    for prov in sorted(provider_totals.keys()):
        avg = provider_totals[prov] / provider_appearances[prov]
        console.print(
            f"  {prov:20s}: {provider_totals[prov]:4d} beliefs across {provider_appearances[prov]:2d} rounds (avg {avg:.1f}/round)"
        )

    # Save to JSON
    output = {
        "n_sources": len(all_analyses),
        "sources": [
            {
                "label": a["label"],
                "n_providers": a["n_providers"],
                "total_beliefs": a["total_beliefs"],
                "n_agreements": a["n_agreements"],
                "n_tensions": a["n_tensions"],
                "threshold_counts": {
                    str(k): v for k, v in a["threshold_counts"].items()
                },
            }
            for a in all_analyses
        ],
        "aggregate_pairwise": {
            pair: agg_pair_agree[pair] / agg_pair_count[pair] for pair in sorted_pairs
        },
        "total_agreements": total_agreements,
    }

    out_path = Path("results/checkpoint_analysis.json")
    out_path.write_text(json.dumps(output, indent=2))
    console.print(f"\n[green]Saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
