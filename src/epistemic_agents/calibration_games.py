"""Calibration games — synthetic tasks with known ground truth for testing model accuracy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from epistemic_agents.schema import Belief, ConfidenceLevel

if TYPE_CHECKING:
    from epistemic_agents.providers.base import BaseProvider


@dataclass
class CalibrationTask:
    """A task with known ground truth for calibration testing."""

    question: str
    ground_truth: bool  # True if the claim in the question is correct
    category: str
    difficulty: str  # "easy", "medium", "hard"


@dataclass
class CalibrationGameResult:
    """Result of one provider's performance on a calibration task."""

    provider_name: str
    task: CalibrationTask
    predicted_confidence: float  # Provider's confidence the claim is true
    correct: bool  # Whether the provider got it right
    brier_score: float  # (confidence - outcome)^2


# Predefined tasks with known ground truth
CALIBRATION_TASKS: list[CalibrationTask] = [
    CalibrationTask(
        question="Is the time complexity of binary search O(log n)?",
        ground_truth=True,
        category="algorithms",
        difficulty="easy",
    ),
    CalibrationTask(
        question="Is Python's built-in sort algorithm Timsort, which is stable?",
        ground_truth=True,
        category="languages",
        difficulty="easy",
    ),
    CalibrationTask(
        question="Does TCP use a 3-way handshake for connection establishment?",
        ground_truth=True,
        category="networking",
        difficulty="easy",
    ),
    CalibrationTask(
        question="Is the space complexity of merge sort O(1)?",
        ground_truth=False,  # It's O(n)
        category="algorithms",
        difficulty="medium",
    ),
    CalibrationTask(
        question="Can a SQL query with a LEFT JOIN return more rows than the left table?",
        ground_truth=True,  # Yes, if there are multiple matches in the right table
        category="databases",
        difficulty="medium",
    ),
    CalibrationTask(
        question="Is JavaScript single-threaded in all runtime environments?",
        ground_truth=False,  # Worker threads exist in Node.js
        category="languages",
        difficulty="medium",
    ),
    CalibrationTask(
        question="Does HTTPS prevent man-in-the-middle attacks in all cases?",
        ground_truth=False,  # Certificate pinning issues, compromised CAs, etc.
        category="security",
        difficulty="hard",
    ),
    CalibrationTask(
        question="Is the CAP theorem about choosing exactly 2 of 3 properties?",
        ground_truth=False,  # It's more nuanced — about tradeoffs during partitions
        category="distributed_systems",
        difficulty="hard",
    ),
    CalibrationTask(
        question="Can a neural network with a single hidden layer approximate any continuous function?",
        ground_truth=True,  # Universal approximation theorem
        category="ml",
        difficulty="hard",
    ),
    CalibrationTask(
        question="Is Redis single-threaded for command processing?",
        ground_truth=True,  # Main event loop is single-threaded (as of Redis 6, I/O threads exist but commands are single-threaded)
        category="databases",
        difficulty="medium",
    ),
]


CALIBRATION_PROMPT = """\
You are evaluating a technical claim. Provide your analysis as a structured response.

For the following question, determine:
1. Whether the claim is TRUE or FALSE
2. Your confidence level (high/moderate/low/speculative)
3. Your justification

Question: {question}

Structure your response as:
- **Answer**: TRUE or FALSE
- **Confidence**: high/moderate/low/speculative
- **Justification**: Your reasoning
"""


def run_calibration_game(
    providers: list[BaseProvider],
    task_index: int = 0,
) -> list[CalibrationGameResult]:
    """Run a calibration game: each provider analyzes a task, beliefs are extracted.

    Args:
        providers: List of providers to test.
        task_index: Index into CALIBRATION_TASKS (0-based).

    Returns:
        List of results, one per provider.
    """
    if task_index >= len(CALIBRATION_TASKS):
        raise ValueError(f"Task index {task_index} out of range (max {len(CALIBRATION_TASKS) - 1})")

    task = CALIBRATION_TASKS[task_index]
    prompt = CALIBRATION_PROMPT.format(question=task.question)
    results: list[CalibrationGameResult] = []

    for provider in providers:
        if not provider.available:
            continue

        try:
            raw = provider.analyze(task.question, prompt)
            confidence = _extract_confidence(raw)
            outcome = 1.0 if task.ground_truth else 0.0
            brier = (confidence - outcome) ** 2

            results.append(CalibrationGameResult(
                provider_name=provider.name,
                task=task,
                predicted_confidence=confidence,
                correct=_check_correct(raw, task.ground_truth),
                brier_score=brier,
            ))
        except Exception:
            pass

    return results


def _extract_confidence(raw: str) -> float:
    """Extract confidence score from raw analysis text."""
    raw_lower = raw.lower()
    if "high" in raw_lower:
        return 0.9
    elif "moderate" in raw_lower:
        return 0.7
    elif "low" in raw_lower:
        return 0.4
    elif "speculative" in raw_lower:
        return 0.2
    return 0.5  # Default


def _check_correct(raw: str, ground_truth: bool) -> bool:
    """Check if the provider's answer matches ground truth."""
    raw_lower = raw.lower()
    said_true = "true" in raw_lower and "false" not in raw_lower
    said_false = "false" in raw_lower and "true" not in raw_lower

    if ground_truth:
        return said_true
    else:
        return said_false
