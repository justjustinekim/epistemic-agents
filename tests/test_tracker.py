"""Tests for cost and contribution tracking."""

import json
import tempfile
from pathlib import Path

from epistemic_agents.tracker import (
    ProviderContribution,
    ProviderUsage,
    SessionStats,
    UsageTracker,
)
from epistemic_agents.schema import (
    AgreementPoint,
    BlindSpot,
    ConfidenceLevel,
    PanelSynthesis,
    ProviderPosition,
    TensionPoint,
    UniqueInsight,
)


def test_provider_usage_total_tokens():
    u = ProviderUsage(
        provider_name="claude",
        model_id="opus",
        input_tokens=1000,
        output_tokens=500,
    )
    assert u.total_tokens == 1500


def test_record_usage():
    tracker = UsageTracker(path=Path(tempfile.mktemp(suffix=".json")))
    usage = tracker.record_usage("claude", "opus", input_tokens=1000, output_tokens=500)
    assert usage.estimated_cost_usd > 0
    assert usage.total_tokens == 1500


def test_finalize_session():
    path = Path(tempfile.mktemp(suffix=".json"))
    tracker = UsageTracker(path=path)
    tracker.record_usage("claude", "opus", input_tokens=1000, output_tokens=500)
    tracker.record_usage("gemini", "gemini-2.0-flash", input_tokens=1000, output_tokens=500)

    stats = tracker.finalize_session(
        task="Test task",
        tier="standard",
        elapsed=10.0,
        model_count=2,
    )
    assert stats.total_tokens == 3000
    assert stats.estimated_cost_usd > 0
    assert stats.tier == "standard"
    assert len(stats.provider_usage) == 2

    # Verify persistence
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 1


def test_persistence():
    path = Path(tempfile.mktemp(suffix=".json"))
    t1 = UsageTracker(path=path)
    t1.record_usage("claude", "opus", input_tokens=100, output_tokens=50)
    t1.finalize_session(task="Session 1", tier="quick")

    # Load from same path
    t2 = UsageTracker(path=path)
    assert len(t2.sessions) == 1
    assert t2.sessions[0].task_summary == "Session 1"


def test_estimate_session_tokens():
    tracker = UsageTracker(path=Path(tempfile.mktemp(suffix=".json")))
    quick = tracker.estimate_session_tokens("Simple question", "quick", model_count=1)
    standard = tracker.estimate_session_tokens("Build auth system", "standard", model_count=1)
    deep = tracker.estimate_session_tokens("Evaluate strategy", "deep", model_count=4, rounds=3)

    assert quick < standard < deep


def test_extract_contributions():
    tracker = UsageTracker(path=Path(tempfile.mktemp(suffix=".json")))

    synthesis = PanelSynthesis(
        task="test",
        provider_positions=[
            ProviderPosition(provider_name="claude", model_id="opus", raw_analysis="..."),
            ProviderPosition(provider_name="gemini", model_id="flash", raw_analysis="..."),
        ],
        agreements=[
            AgreementPoint(
                claim="X is true",
                supporting_providers=["claude", "gemini"],
                combined_confidence=ConfidenceLevel.HIGH,
            ),
        ],
        tensions=[
            TensionPoint(
                claim="Y approach",
                positions={"claude": "for", "gemini": "against"},
                synthesis_notes="...",
            ),
        ],
        blind_spots=[
            BlindSpot(
                observation="Missed Z",
                identified_by="gemini",
                missed_by=["claude"],
            ),
        ],
        unique_insights=[
            UniqueInsight(
                insight="Novel idea",
                source_provider="claude",
                relevance="Important",
            ),
        ],
        synthesized_strategy="Claude suggested X, gemini suggested Y",
        meta_confidence="moderate",
    )

    contribs = tracker.extract_contributions(synthesis)
    assert len(contribs) == 2

    by_name = {c.provider_name: c for c in contribs}
    assert by_name["claude"].unique_insights == 1  # from unique_insights
    assert by_name["gemini"].unique_insights == 1  # from blind_spots
    assert by_name["claude"].agreements_participated == 1
    assert by_name["gemini"].agreements_participated == 1
    assert by_name["claude"].tensions_involved == 1
    assert by_name["claude"].was_referenced_in_synthesis is True
    assert by_name["gemini"].was_referenced_in_synthesis is True


def test_cumulative_summary():
    path = Path(tempfile.mktemp(suffix=".json"))
    tracker = UsageTracker(path=path)

    # No data
    assert tracker.cumulative_summary() == "No usage data yet."

    # Add a session
    tracker.record_usage("claude", "opus", input_tokens=5000, output_tokens=2000)
    tracker.finalize_session(
        task="Test",
        tier="standard",
        model_count=1,
        contributions=[
            ProviderContribution(
                provider_name="claude",
                model_id="opus",
                unique_insights=3,
            ),
        ],
    )

    summary = tracker.cumulative_summary()
    assert "1 sessions" in summary
    assert "claude: 3" in summary


def test_free_tier_zero_cost():
    tracker = UsageTracker(path=Path(tempfile.mktemp(suffix=".json")))
    usage = tracker.record_usage("gemini", "gemini-2.0-flash", input_tokens=10000, output_tokens=5000)
    assert usage.estimated_cost_usd == 0.0

    usage2 = tracker.record_usage("qwq", "qwq-plus", input_tokens=10000, output_tokens=5000)
    assert usage2.estimated_cost_usd == 0.0
