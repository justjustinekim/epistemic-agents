"""Tests for tournament system."""

import tempfile
from pathlib import Path

from epistemic_agents.tournament import TournamentLog, TournamentResult


def test_tournament_result_construction():
    result = TournamentResult(
        task="Test task",
        results={"quick": "Use A", "standard": "Use B", "deep": "Use B"},
        verdict_changed=True,
        confidence_changed=False,
        deep_added_value="Deep found edge case",
    )
    assert result.verdict_changed is True
    assert result.deep_added_value == "Deep found edge case"


def test_tournament_log_persistence():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tournaments.json"

        log1 = TournamentLog(path=path)
        log1.record(TournamentResult(
            task="Task 1",
            results={"quick": "A", "standard": "B", "deep": "B"},
            verdict_changed=True,
        ))

        log2 = TournamentLog(path=path)
        assert len(log2.results) == 1
        assert log2.results[0].task == "Task 1"


def test_tournament_log_value_of_depth():
    with tempfile.TemporaryDirectory() as d:
        log = TournamentLog(path=Path(d) / "t.json")

        log.record(TournamentResult(
            task="Task 1",
            results={"quick": "A", "standard": "A", "deep": "B"},
            verdict_changed=True,
            deep_added_value="Found edge case",
        ))
        log.record(TournamentResult(
            task="Task 2",
            results={"quick": "A", "standard": "A", "deep": "A"},
            verdict_changed=False,
        ))

        summary = log.value_of_depth_summary()
        assert "2 runs" in summary
        assert "1/2" in summary  # verdict changed in 1 of 2


def test_tournament_log_empty():
    with tempfile.TemporaryDirectory() as d:
        log = TournamentLog(path=Path(d) / "t.json")
        assert "No tournament data" in log.value_of_depth_summary()
