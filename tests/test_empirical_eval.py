"""Tests for empirical_eval module."""

from epistemic_agents.empirical_eval import EvalResult, EvalComparison, EvalReport


def test_eval_result_defaults():
    r = EvalResult(approach="test", task="t")
    assert r.n_beliefs == 0
    assert r.error is None


def test_eval_report_format_empty():
    report = EvalReport(n_tasks=0)
    text = report.format_report()
    assert "EMPIRICAL EVALUATION" in text


def test_eval_report_format_with_data():
    r1 = EvalResult(approach="panel", task="t", n_beliefs=5, avg_confidence=0.8, elapsed_seconds=10.0)
    r2 = EvalResult(approach="adversarial", task="t", n_beliefs=3, avg_confidence=0.7, elapsed_seconds=2.0)
    comp = EvalComparison(task="t", results={"panel": r1, "adversarial": r2})
    report = EvalReport(comparisons=[comp], n_tasks=1)
    text = report.format_report()
    assert "panel" in text
    assert "adversarial" in text


def test_eval_result_with_error():
    r = EvalResult(approach="panel", task="t", error="API timeout")
    comp = EvalComparison(task="t", results={"panel": r})
    report = EvalReport(comparisons=[comp], n_tasks=1)
    text = report.format_report()
    assert "ERROR" in text
