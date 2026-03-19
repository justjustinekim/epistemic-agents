"""Phase 0: Consensus audit — measure how often majority consensus is wrong.

Runs calibration tasks with known ground truth across all available providers,
then computes consensus accuracy at different agreement thresholds. This data
resolves the C1 vote-locking threshold debate and determines whether C2
(MAD-conformist freeze) has any empirical basis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from epistemic_agents.calibration_games import (
    CALIBRATION_TASKS,
    CalibrationGameResult,
    CalibrationTask,
    run_calibration_game,
)


@dataclass
class ConsensusResult:
    """Result of majority consensus on a single calibration task."""

    task: CalibrationTask
    provider_results: list[CalibrationGameResult]
    n_providers: int

    # Majority vote outcome
    majority_answer: bool | None = None  # True/False/None (no majority)
    majority_correct: bool | None = None
    agreement_count: int = 0  # How many providers agreed with majority
    consensus_confidence: float = 0.5

    # Per-threshold correctness
    threshold_correct: dict[int, bool | None] = field(default_factory=dict)
    # threshold_correct[k] = True/False/None means:
    #   at threshold k (>=k providers agree), was consensus correct?
    #   None means threshold not met (no lock would occur)


@dataclass
class ConsensusAuditReport:
    """Full audit report across all calibration tasks."""

    results: list[ConsensusResult]
    n_providers: int
    n_tasks: int

    # Per-provider accuracy
    provider_accuracy: dict[str, float] = field(default_factory=dict)
    provider_brier: dict[str, float] = field(default_factory=dict)

    # Consensus accuracy by threshold
    # threshold -> (correct_count, total_locked, accuracy)
    threshold_stats: dict[int, tuple[int, int, float]] = field(default_factory=dict)

    # Key metrics for C1 threshold decision
    majority_error_rate: float = 0.0  # How often simple majority is wrong
    unanimous_error_rate: float = 0.0  # How often unanimous consensus is wrong
    optimal_threshold: int = (
        0  # Threshold that maximizes accuracy while locking >50% of tasks
    )

    def format_report(self) -> str:
        """Human-readable audit report."""
        lines = [
            "=" * 60,
            "PHASE 0: CONSENSUS AUDIT REPORT",
            "=" * 60,
            f"Providers: {self.n_providers}",
            f"Tasks: {self.n_tasks}",
            "",
            "--- Per-Provider Accuracy ---",
        ]

        for name in sorted(self.provider_accuracy.keys()):
            acc = self.provider_accuracy[name]
            brier = self.provider_brier.get(name, 0.0)
            lines.append(f"  {name:20s}: {acc:.0%} correct, Brier={brier:.4f}")

        lines.extend(["", "--- Consensus by Threshold ---"])
        lines.append(
            f"  {'Threshold':>12s}  {'Locked':>8s}  {'Correct':>8s}  {'Accuracy':>10s}  {'Error Rate':>12s}"
        )
        lines.append("  " + "-" * 56)

        for thresh in sorted(self.threshold_stats.keys()):
            correct, total, acc = self.threshold_stats[thresh]
            err = 1.0 - acc if total > 0 else float("nan")
            locked_pct = f"{total}/{self.n_tasks}"
            lines.append(
                f"  >= {thresh} providers  {locked_pct:>8s}  {correct:>8d}  {acc:>9.0%}  {err:>11.1%}"
            )

        lines.extend(
            [
                "",
                "--- Key Findings ---",
                f"  Majority error rate (>={math.ceil(self.n_providers / 2)} agree): {self.majority_error_rate:.1%}",
                f"  Unanimous error rate ({self.n_providers}/{self.n_providers} agree): {self.unanimous_error_rate:.1%}",
                f"  Recommended C1 lock threshold: >= {self.optimal_threshold} providers",
            ]
        )

        # Per-task breakdown
        lines.extend(["", "--- Per-Task Breakdown ---"])
        for r in self.results:
            status = (
                "CORRECT"
                if r.majority_correct
                else "WRONG"
                if r.majority_correct is False
                else "NO MAJORITY"
            )
            gt = "TRUE" if r.task.ground_truth else "FALSE"
            lines.append(f"  [{r.task.difficulty:6s}] {r.task.question[:60]:60s}")
            lines.append(
                f"           Ground truth: {gt}, Majority: {status}, Agreement: {r.agreement_count}/{r.n_providers}"
            )
            for pr in r.provider_results:
                mark = "v" if pr.correct else "X"
                lines.append(
                    f"             [{mark}] {pr.provider_name}: conf={pr.predicted_confidence:.2f}, brier={pr.brier_score:.4f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


def run_consensus_audit(
    providers: list,
    tasks: list[CalibrationTask] | None = None,
) -> ConsensusAuditReport:
    """Run all calibration tasks across all providers, measure consensus accuracy.

    Args:
        providers: List of BaseProvider instances.
        tasks: Optional task list (defaults to CALIBRATION_TASKS).

    Returns:
        ConsensusAuditReport with full breakdown.
    """
    tasks = tasks or CALIBRATION_TASKS
    results: list[ConsensusResult] = []

    for i, task in enumerate(tasks):
        game_results = run_calibration_game(providers, task_index=i)
        if not game_results:
            continue

        cr = _evaluate_consensus(task, game_results, len(providers))
        results.append(cr)

    return _build_report(results, len(providers), len(tasks))


def _evaluate_consensus(
    task: CalibrationTask,
    game_results: list[CalibrationGameResult],
    n_providers: int,
) -> ConsensusResult:
    """Evaluate consensus on a single task."""
    cr = ConsensusResult(
        task=task,
        provider_results=game_results,
        n_providers=n_providers,
    )

    # Determine what each provider answered based on correctness + ground truth
    said_true = []
    said_false = []
    for r in game_results:
        if r.correct:
            # Got it right → they agreed with ground truth
            if task.ground_truth:
                said_true.append(r)
            else:
                said_false.append(r)
        else:
            # Got it wrong → they disagreed with ground truth
            if task.ground_truth:
                said_false.append(r)
            else:
                said_true.append(r)

    n_true = len(said_true)
    n_false = len(said_false)

    if n_true > n_false:
        cr.majority_answer = True
        cr.agreement_count = n_true
        cr.majority_correct = task.ground_truth is True
    elif n_false > n_true:
        cr.majority_answer = False
        cr.agreement_count = n_false
        cr.majority_correct = task.ground_truth is False
    else:
        cr.majority_answer = None
        cr.agreement_count = max(n_true, n_false)
        cr.majority_correct = None

    # Consensus confidence (average of majority-side confidences)
    majority_side = said_true if n_true >= n_false else said_false
    if majority_side:
        cr.consensus_confidence = sum(
            r.predicted_confidence for r in majority_side
        ) / len(majority_side)

    # Per-threshold correctness
    for thresh in range(2, n_providers + 1):
        if cr.agreement_count >= thresh:
            cr.threshold_correct[thresh] = cr.majority_correct
        else:
            cr.threshold_correct[thresh] = None  # Threshold not met

    return cr


def _build_report(
    results: list[ConsensusResult],
    n_providers: int,
    n_tasks: int,
) -> ConsensusAuditReport:
    """Build the full audit report from individual results."""
    report = ConsensusAuditReport(
        results=results,
        n_providers=n_providers,
        n_tasks=n_tasks,
    )

    # Per-provider accuracy and Brier
    provider_correct: dict[str, int] = {}
    provider_total: dict[str, int] = {}
    provider_brier_sum: dict[str, float] = {}

    for cr in results:
        for pr in cr.provider_results:
            provider_correct.setdefault(pr.provider_name, 0)
            provider_total.setdefault(pr.provider_name, 0)
            provider_brier_sum.setdefault(pr.provider_name, 0.0)
            provider_total[pr.provider_name] += 1
            if pr.correct:
                provider_correct[pr.provider_name] += 1
            provider_brier_sum[pr.provider_name] += pr.brier_score

    for name in provider_total:
        total = provider_total[name]
        report.provider_accuracy[name] = (
            provider_correct[name] / total if total > 0 else 0.0
        )
        report.provider_brier[name] = (
            provider_brier_sum[name] / total if total > 0 else 0.0
        )

    # Threshold stats
    for thresh in range(2, n_providers + 1):
        correct = 0
        total_locked = 0
        for cr in results:
            val = cr.threshold_correct.get(thresh)
            if val is not None:
                total_locked += 1
                if val:
                    correct += 1
        acc = correct / total_locked if total_locked > 0 else 0.0
        report.threshold_stats[thresh] = (correct, total_locked, acc)

    # Key metrics
    majority_thresh = math.ceil(n_providers / 2)
    if majority_thresh in report.threshold_stats:
        _, total, acc = report.threshold_stats[majority_thresh]
        report.majority_error_rate = 1.0 - acc if total > 0 else 0.0

    if n_providers in report.threshold_stats:
        _, total, acc = report.threshold_stats[n_providers]
        report.unanimous_error_rate = 1.0 - acc if total > 0 else 0.0

    # Find optimal threshold: maximize accuracy while locking > 50% of tasks
    best_thresh = majority_thresh
    best_score = 0.0
    for thresh in sorted(report.threshold_stats.keys()):
        correct, total_locked, acc = report.threshold_stats[thresh]
        coverage = total_locked / n_tasks if n_tasks > 0 else 0.0
        if coverage >= 0.3:  # Lock at least 30% of tasks
            # Score: accuracy * sqrt(coverage) — reward accuracy but penalize low coverage
            score = acc * math.sqrt(coverage)
            if score >= best_score:
                best_score = score
                best_thresh = thresh
    report.optimal_threshold = best_thresh

    return report
