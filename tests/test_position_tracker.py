"""Tests for position tracking across debate rounds."""

from epistemic_agents.position_tracker import (
    StanceShift,
    format_position_summary,
    track_positions,
)
from epistemic_agents.schema import Belief, ConfidenceLevel, ProviderPosition


def _pos(name: str, beliefs: list[Belief]) -> ProviderPosition:
    return ProviderPosition(
        provider_name=name,
        model_id=f"{name}-model",
        beliefs=beliefs,
        raw_analysis="test",
    )


def test_track_positions_detects_strengthening():
    round1 = [
        _pos("claude", [
            Belief(
                id="c-b1",
                claim="Redis caching improves latency for API calls",
                confidence=ConfidenceLevel.LOW,
                justification="test",
            ),
        ]),
    ]
    round2 = [
        _pos("claude", [
            Belief(
                id="c-b1",
                claim="Redis caching improves latency for API calls",
                confidence=ConfidenceLevel.HIGH,
                justification="test",
            ),
        ]),
    ]
    shifts = track_positions([round1, round2])
    assert len(shifts) >= 1
    assert shifts[0].direction == "strengthened"
    assert shifts[0].provider_name == "claude"


def test_track_positions_detects_weakening():
    round1 = [
        _pos("gemini", [
            Belief(
                id="g-b1",
                claim="GraphQL better than REST for this API",
                confidence=ConfidenceLevel.HIGH,
                justification="test",
            ),
        ]),
    ]
    round2 = [
        _pos("gemini", [
            Belief(
                id="g-b1",
                claim="GraphQL better than REST for this API",
                confidence=ConfidenceLevel.LOW,
                justification="test",
            ),
        ]),
    ]
    shifts = track_positions([round1, round2])
    assert len(shifts) >= 1
    assert shifts[0].direction == "weakened"


def test_track_positions_no_change():
    round1 = [
        _pos("claude", [
            Belief(
                id="c-b1",
                claim="Use Redis for caching",
                confidence=ConfidenceLevel.HIGH,
                justification="test",
            ),
        ]),
    ]
    round2 = [
        _pos("claude", [
            Belief(
                id="c-b1",
                claim="Use Redis for caching",
                confidence=ConfidenceLevel.HIGH,
                justification="test",
            ),
        ]),
    ]
    shifts = track_positions([round1, round2])
    assert len(shifts) == 0


def test_track_positions_single_round():
    round1 = [
        _pos("claude", [
            Belief(id="c-b1", claim="Test", confidence=ConfidenceLevel.HIGH, justification="test"),
        ]),
    ]
    shifts = track_positions([round1])
    assert shifts == []


def test_track_positions_empty():
    shifts = track_positions([])
    assert shifts == []


def test_format_position_summary_empty():
    assert format_position_summary([]) == ""


def test_format_position_summary_with_shifts():
    shifts = [
        StanceShift(
            provider_name="claude",
            claim="Redis is best",
            old_confidence_score=0.4,
            new_confidence_score=0.9,
            direction="strengthened",
        ),
    ]
    text = format_position_summary(shifts)
    assert "POSITION SHIFTS" in text
    assert "claude" in text
    assert "strengthened" in text
