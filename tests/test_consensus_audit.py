"""Tests for consensus_audit module."""

from epistemic_agents.calibration_games import CalibrationTask, CalibrationGameResult
from epistemic_agents.consensus_audit import (
    ConsensusResult,
    ConsensusAuditReport,
    _evaluate_consensus,
    _build_report,
)


def _make_result(
    name: str, confidence: float, correct: bool, ground_truth: bool
) -> CalibrationGameResult:
    task = CalibrationTask(
        question="test", ground_truth=ground_truth, category="test", difficulty="easy"
    )
    outcome = 1.0 if ground_truth else 0.0
    brier = (confidence - outcome) ** 2
    return CalibrationGameResult(
        provider_name=name,
        task=task,
        predicted_confidence=confidence,
        correct=correct,
        brier_score=brier,
    )


class TestEvaluateConsensus:
    def test_unanimous_correct(self):
        """All providers agree and are correct."""
        task = CalibrationTask(
            question="test", ground_truth=True, category="test", difficulty="easy"
        )
        results = [
            _make_result("a", 0.9, True, True),
            _make_result("b", 0.9, True, True),
            _make_result("c", 0.9, True, True),
        ]
        cr = _evaluate_consensus(task, results, 3)
        assert cr.majority_answer is True
        assert cr.majority_correct is True
        assert cr.agreement_count == 3

    def test_unanimous_wrong(self):
        """All providers agree but are wrong."""
        task = CalibrationTask(
            question="test", ground_truth=False, category="test", difficulty="hard"
        )
        results = [
            _make_result("a", 0.9, False, False),
            _make_result("b", 0.9, False, False),
            _make_result("c", 0.7, False, False),
        ]
        cr = _evaluate_consensus(task, results, 3)
        assert cr.majority_correct is False
        assert cr.agreement_count == 3

    def test_majority_correct_minority_wrong(self):
        """Majority is correct, minority dissents."""
        task = CalibrationTask(
            question="test", ground_truth=True, category="test", difficulty="medium"
        )
        results = [
            _make_result("a", 0.9, True, True),
            _make_result("b", 0.9, True, True),
            _make_result("c", 0.4, False, True),
        ]
        cr = _evaluate_consensus(task, results, 3)
        assert cr.majority_answer is True
        assert cr.majority_correct is True
        assert cr.agreement_count == 2

    def test_majority_wrong_minority_correct(self):
        """Majority is wrong — the critical case for threshold calibration."""
        task = CalibrationTask(
            question="test", ground_truth=True, category="test", difficulty="hard"
        )
        results = [
            _make_result("a", 0.4, False, True),
            _make_result("b", 0.4, False, True),
            _make_result("c", 0.9, True, True),
        ]
        cr = _evaluate_consensus(task, results, 3)
        assert cr.majority_answer is False
        assert cr.majority_correct is False
        assert cr.agreement_count == 2

    def test_tie_no_majority(self):
        """Equal votes → no majority."""
        task = CalibrationTask(
            question="test", ground_truth=True, category="test", difficulty="easy"
        )
        results = [
            _make_result("a", 0.9, True, True),
            _make_result("b", 0.4, False, True),
        ]
        cr = _evaluate_consensus(task, results, 2)
        assert cr.majority_answer is None
        assert cr.majority_correct is None

    def test_threshold_correct_populated(self):
        """threshold_correct maps thresholds to correctness."""
        task = CalibrationTask(
            question="test", ground_truth=True, category="test", difficulty="easy"
        )
        results = [
            _make_result("a", 0.9, True, True),
            _make_result("b", 0.9, True, True),
            _make_result("c", 0.9, True, True),
            _make_result("d", 0.4, False, True),
            _make_result("e", 0.4, False, True),
        ]
        cr = _evaluate_consensus(task, results, 5)
        # 3 said True, 2 said False → agreement_count = 3
        assert cr.threshold_correct[2] is True  # >=2, 3 agree → locked
        assert cr.threshold_correct[3] is True  # >=3, 3 agree → locked
        assert cr.threshold_correct[4] is None  # >=4, only 3 → not locked
        assert cr.threshold_correct[5] is None  # >=5, only 3 → not locked


class TestBuildReport:
    def test_report_stats(self):
        """Report correctly aggregates per-provider and threshold stats."""
        task1 = CalibrationTask(
            question="t1", ground_truth=True, category="test", difficulty="easy"
        )
        task2 = CalibrationTask(
            question="t2", ground_truth=False, category="test", difficulty="hard"
        )

        cr1 = ConsensusResult(
            task=task1,
            provider_results=[
                _make_result("a", 0.9, True, True),
                _make_result("b", 0.9, True, True),
                _make_result("c", 0.9, True, True),
            ],
            n_providers=3,
            majority_answer=True,
            majority_correct=True,
            agreement_count=3,
            threshold_correct={2: True, 3: True},
        )

        cr2 = ConsensusResult(
            task=task2,
            provider_results=[
                _make_result("a", 0.9, False, False),
                _make_result("b", 0.9, False, False),
                _make_result("c", 0.4, True, False),
            ],
            n_providers=3,
            majority_answer=True,
            majority_correct=False,
            agreement_count=2,
            threshold_correct={2: False, 3: None},
        )

        report = _build_report([cr1, cr2], n_providers=3, n_tasks=2)

        assert report.n_providers == 3
        assert report.n_tasks == 2
        # Provider a: 1 correct out of 2
        assert report.provider_accuracy["a"] == 0.5
        # Threshold 2: both locked, 1 correct, 1 wrong → 50%
        assert report.threshold_stats[2] == (1, 2, 0.5)
        # Threshold 3: only task1 locked → 100%
        assert report.threshold_stats[3] == (1, 1, 1.0)

    def test_format_report_runs(self):
        """format_report doesn't crash on empty results."""
        report = ConsensusAuditReport(results=[], n_providers=3, n_tasks=0)
        text = report.format_report()
        assert "CONSENSUS AUDIT REPORT" in text

    def test_optimal_threshold_favors_accuracy_with_coverage(self):
        """Optimal threshold balances accuracy and coverage."""
        task = CalibrationTask(
            question="test", ground_truth=True, category="test", difficulty="easy"
        )

        # 5 tasks, all unanimous correct
        results = []
        for _ in range(5):
            cr = ConsensusResult(
                task=task,
                provider_results=[_make_result("a", 0.9, True, True)] * 5,
                n_providers=5,
                majority_answer=True,
                majority_correct=True,
                agreement_count=5,
                threshold_correct={2: True, 3: True, 4: True, 5: True},
            )
            results.append(cr)

        report = _build_report(results, n_providers=5, n_tasks=5)
        # All thresholds have 100% accuracy, optimal should be highest with coverage
        assert report.optimal_threshold >= 2
        assert report.majority_error_rate == 0.0
        assert report.unanimous_error_rate == 0.0
