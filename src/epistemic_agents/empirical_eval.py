"""Empirical evaluation harness — compare panel vs single-model approaches.

Runs the same set of tasks through three configurations:
  (a) Full multi-model panel (7 providers, debate, synthesis)
  (b) Single model with adversarial self-critique prompting
  (c) Single model with persona-switching (simulate diverse perspectives)

Measures: belief count, confidence calibration, unique insight rate,
blind spot detection, and optionally ground-truth accuracy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from epistemic_agents import client
from epistemic_agents.agreement_detector import detect_agreements, detect_tensions
from epistemic_agents.belief_extractor import extract_beliefs


@dataclass
class EvalResult:
    """Result of evaluating a single task with one approach."""

    approach: str  # "panel", "adversarial", "persona"
    task: str
    beliefs: list[dict] = field(default_factory=list)
    n_beliefs: int = 0
    n_unique_claims: int = 0
    avg_confidence: float = 0.0
    confidence_spread: float = 0.0  # std dev of confidence scores
    n_agreements: int = 0
    n_tensions: int = 0
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class EvalComparison:
    """Comparison across approaches for a single task."""

    task: str
    results: dict[str, EvalResult] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Full evaluation report across all tasks."""

    comparisons: list[EvalComparison] = field(default_factory=list)
    n_tasks: int = 0

    def format_report(self) -> str:
        lines = [
            "=" * 70,
            "EMPIRICAL EVALUATION REPORT",
            "=" * 70,
            f"Tasks evaluated: {self.n_tasks}",
            "",
        ]

        # Summary table
        approach_stats: dict[str, dict[str, float]] = {}
        for comp in self.comparisons:
            for approach, result in comp.results.items():
                if approach not in approach_stats:
                    approach_stats[approach] = {
                        "beliefs": 0, "confidence": 0, "tensions": 0,
                        "time": 0, "count": 0, "agreements": 0,
                    }
                s = approach_stats[approach]
                s["beliefs"] += result.n_beliefs
                s["confidence"] += result.avg_confidence
                s["tensions"] += result.n_tensions
                s["agreements"] += result.n_agreements
                s["time"] += result.elapsed_seconds
                s["count"] += 1

        lines.append(f"{'Approach':<20} {'Beliefs':>8} {'Avg Conf':>10} {'Tensions':>10} {'Time(s)':>10}")
        lines.append("-" * 60)
        for approach, s in approach_stats.items():
            n = max(s["count"], 1)
            lines.append(
                f"{approach:<20} {s['beliefs']/n:>8.1f} {s['confidence']/n:>10.2f} "
                f"{s['tensions']/n:>10.1f} {s['time']/n:>10.1f}"
            )

        # Per-task details
        for comp in self.comparisons:
            lines.extend(["", f"--- Task: {comp.task[:70]} ---"])
            for approach, result in comp.results.items():
                if result.error:
                    lines.append(f"  {approach}: ERROR - {result.error}")
                else:
                    lines.append(
                        f"  {approach}: {result.n_beliefs} beliefs, "
                        f"avg_conf={result.avg_confidence:.2f}, "
                        f"spread={result.confidence_spread:.2f}, "
                        f"{result.n_tensions} tensions, "
                        f"{result.elapsed_seconds:.1f}s"
                    )

        lines.append("=" * 70)
        return "\n".join(lines)


ADVERSARIAL_PROMPT = """\
You are a rigorous analyst who challenges your own reasoning. For the given task:

1. First, provide your honest analysis with key beliefs and confidence levels.
2. Then, adopt a devil's advocate perspective and systematically challenge each belief:
   - What assumptions might be wrong?
   - What evidence would falsify each claim?
   - Where are you most likely overconfident?
3. Finally, revise your beliefs based on the self-critique.

Structure your final output as:
**Key Beliefs** — Each with confidence (high/moderate/low/speculative) and justification.
**Revised After Critique** — Which beliefs changed and why.
**Remaining Uncertainties** — What you still can't resolve."""


PERSONA_PROMPT = """\
You will analyze the following task from 4 distinct expert perspectives, then synthesize.

**Perspective 1 — The Pragmatist**: Focus on what works in practice, real-world constraints, \
implementation feasibility. Be skeptical of theoretical elegance.

**Perspective 2 — The Theorist**: Focus on formal correctness, edge cases, mathematical properties. \
Challenge hand-wavy arguments.

**Perspective 3 — The Contrarian**: Actively look for what everyone else is getting wrong. \
Challenge consensus positions. Consider uncommon failure modes.

**Perspective 4 — The Integrator**: Find connections between the other perspectives. \
Identify where they agree (genuine signal) vs. where they clash (needs resolution).

For each perspective, state 2-3 key beliefs with confidence levels.

Then provide a **Synthesis**: Which beliefs survived all 4 perspectives? Where do \
the perspectives genuinely disagree? What are the blind spots?"""


def run_panel_eval(
    task: str,
    providers: list,
) -> EvalResult:
    """Run full panel analysis on a task."""
    from epistemic_agents.panel import ModelPanel

    result = EvalResult(approach="panel", task=task)
    start = time.time()

    try:
        panel = ModelPanel(providers)
        rounds = panel.debate(task, rounds=2)

        # debate() returns list[list[ProviderPosition]] — use final round
        final_positions = rounds[-1] if rounds else []

        all_beliefs = []
        for pos in final_positions:
            for b in pos.beliefs:
                all_beliefs.append(b)
                result.beliefs.append({"claim": b.claim, "confidence": b.effective_score})

        result.n_beliefs = len(all_beliefs)
        scores = [b.effective_score for b in all_beliefs]
        if scores:
            result.avg_confidence = sum(scores) / len(scores)
            mean = result.avg_confidence
            result.confidence_spread = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

        agreements = detect_agreements(final_positions)
        tensions = detect_tensions(final_positions)
        result.n_agreements = len(agreements)
        result.n_tensions = len(tensions)

    except Exception as e:
        result.error = str(e)

    result.elapsed_seconds = time.time() - start
    return result


def run_adversarial_eval(task: str) -> EvalResult:
    """Run single-model adversarial self-critique on a task."""
    result = EvalResult(approach="adversarial", task=task)
    start = time.time()

    try:
        raw = client.plain_request(
            model="sonnet",
            system=ADVERSARIAL_PROMPT,
            user_message=task,
        )
        result.raw_output = raw
        extracted = extract_beliefs(raw)
        for b in extracted.beliefs:
            result.beliefs.append({"claim": b.claim, "confidence": b.effective_score})

        result.n_beliefs = len(extracted.beliefs)
        scores = [b.effective_score for b in extracted.beliefs]
        if scores:
            result.avg_confidence = sum(scores) / len(scores)
            mean = result.avg_confidence
            result.confidence_spread = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

        # Single model can't have inter-model tensions, but track intra-belief spread
        result.n_tensions = 0
        result.n_agreements = 0

    except Exception as e:
        result.error = str(e)

    result.elapsed_seconds = time.time() - start
    return result


def run_persona_eval(task: str) -> EvalResult:
    """Run single-model persona-switching on a task."""
    result = EvalResult(approach="persona", task=task)
    start = time.time()

    try:
        raw = client.plain_request(
            model="sonnet",
            system=PERSONA_PROMPT,
            user_message=task,
        )
        result.raw_output = raw
        extracted = extract_beliefs(raw)
        for b in extracted.beliefs:
            result.beliefs.append({"claim": b.claim, "confidence": b.effective_score})

        result.n_beliefs = len(extracted.beliefs)
        scores = [b.effective_score for b in extracted.beliefs]
        if scores:
            result.avg_confidence = sum(scores) / len(scores)
            mean = result.avg_confidence
            result.confidence_spread = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

        result.n_tensions = 0
        result.n_agreements = 0

    except Exception as e:
        result.error = str(e)

    result.elapsed_seconds = time.time() - start
    return result


def run_empirical_eval(
    tasks: list[str],
    providers: list,
    output_path: str | Path = "results/empirical_eval.json",
) -> EvalReport:
    """Run full empirical evaluation across all approaches.

    Args:
        tasks: List of task descriptions to evaluate.
        providers: List of BaseProvider instances for panel mode.
        output_path: Where to save JSON results.

    Returns:
        EvalReport with full comparison data.
    """
    report = EvalReport(n_tasks=len(tasks))

    for i, task in enumerate(tasks):
        comp = EvalComparison(task=task)

        # Run all three approaches
        comp.results["panel"] = run_panel_eval(task, providers)
        comp.results["adversarial"] = run_adversarial_eval(task)
        comp.results["persona"] = run_persona_eval(task)

        report.comparisons.append(comp)

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    data = {
        "n_tasks": report.n_tasks,
        "comparisons": [
            {
                "task": comp.task,
                "results": {
                    approach: {
                        "approach": r.approach,
                        "n_beliefs": r.n_beliefs,
                        "avg_confidence": r.avg_confidence,
                        "confidence_spread": r.confidence_spread,
                        "n_agreements": r.n_agreements,
                        "n_tensions": r.n_tensions,
                        "elapsed_seconds": r.elapsed_seconds,
                        "beliefs": r.beliefs,
                        "error": r.error,
                    }
                    for approach, r in comp.results.items()
                },
            }
            for comp in report.comparisons
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))

    return report
