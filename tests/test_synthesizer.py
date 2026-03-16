"""Tests for the synthesizer — single, debate, and re-synthesis modes."""

from __future__ import annotations

from unittest.mock import patch

from epistemic_agents.schema import (
    AgreementPoint,
    Belief,
    ConfidenceLevel,
    PanelSynthesis,
    ProviderPosition,
)
from epistemic_agents.synthesizer import Synthesizer


def _synthesis() -> PanelSynthesis:
    return PanelSynthesis(
        task="Test task",
        provider_positions=[],
        agreements=[
            AgreementPoint(
                claim="Redis is good",
                supporting_providers=["claude", "gemini"],
                combined_confidence=ConfidenceLevel.HIGH,
            ),
        ],
        synthesized_strategy="Use Redis for caching",
        meta_confidence="High confidence in this approach",
    )


def _positions() -> list[ProviderPosition]:
    return [
        ProviderPosition(
            provider_name="claude",
            model_id="opus",
            beliefs=[
                Belief(id="c-b1", claim="Redis is fast", confidence=ConfidenceLevel.HIGH, justification="test"),
            ],
            raw_analysis="Redis is the best caching solution for this use case.",
        ),
        ProviderPosition(
            provider_name="gemini",
            model_id="flash",
            beliefs=[
                Belief(id="g-b1", claim="Redis is fast", confidence=ConfidenceLevel.MODERATE, justification="test"),
            ],
            raw_analysis="Redis could work but Memcached is simpler.",
        ),
    ]


@patch("epistemic_agents.synthesizer.client")
def test_synthesize_single_round(mock_client):
    mock_client.structured_request.return_value = _synthesis()

    synth = Synthesizer(model="opus")
    result = synth.synthesize("Test task", _positions())

    assert result.task == "Test task"
    assert len(result.agreements) >= 1
    mock_client.structured_request.assert_called_once()


@patch("epistemic_agents.synthesizer.client")
def test_synthesize_debate_multi_round(mock_client):
    mock_client.structured_request.return_value = _synthesis()

    synth = Synthesizer(model="opus")
    rounds = [_positions(), _positions()]  # 2 rounds
    result = synth.synthesize_debate("Test task", rounds)

    assert result.task == "Test task"
    # Should have been called with debate context
    call_args = mock_client.structured_request.call_args
    user_msg = call_args.kwargs.get("user_message", call_args[1].get("user_message", ""))
    assert "Initial Analysis" in user_msg


@patch("epistemic_agents.synthesizer.client")
def test_resynthesize_with_refutations(mock_client):
    mock_client.structured_request.return_value = _synthesis()

    synth = Synthesizer(model="opus")
    refutations = [
        ProviderPosition(
            provider_name="claude",
            model_id="opus",
            beliefs=[],
            raw_analysis="The synthesis misrepresented my position.",
        ),
    ]
    result = synth.resynthesize(
        "Test task", [_positions()], _synthesis(), refutations
    )

    assert result.task == "Test task"
    call_args = mock_client.structured_request.call_args
    user_msg = call_args.kwargs.get("user_message", call_args[1].get("user_message", ""))
    assert "refutation" in user_msg.lower() or "Refutation" in user_msg


@patch("epistemic_agents.synthesizer.client")
def test_programmatic_detection_injected(mock_client):
    """Test that programmatic agreements/tensions are injected into synthesis context."""
    mock_client.structured_request.return_value = _synthesis()

    positions = _positions()
    synth = Synthesizer(model="opus")
    result = synth.synthesize("Test task", positions)

    # The programmatic analysis should be in the user message
    call_args = mock_client.structured_request.call_args
    user_msg = call_args.kwargs.get("user_message", call_args[1].get("user_message", ""))
    # With beliefs populated, programmatic analysis should be injected
    assert "PROGRAMMATIC ANALYSIS" in user_msg or len(positions[0].beliefs) == 0


@patch("epistemic_agents.synthesizer.client")
def test_synthesize_n_samples_picks_longest(mock_client):
    """Test that n_samples>1 picks the synthesis with longest strategy."""
    short = _synthesis()
    short.synthesized_strategy = "Short"
    long = _synthesis()
    long.synthesized_strategy = "This is a much longer and more comprehensive strategy"
    mock_client.structured_request.side_effect = [short, long, short]

    synth = Synthesizer(model="opus")
    result = synth._run_synthesis("Test task", "msg", _positions(), n_samples=3)
    assert result.synthesized_strategy == long.synthesized_strategy
    assert mock_client.structured_request.call_count == 3


@patch("epistemic_agents.synthesizer.client")
def test_synthesize_single_sample(mock_client):
    """Test that n_samples=1 calls structured_request exactly once."""
    mock_client.structured_request.return_value = _synthesis()

    synth = Synthesizer(model="opus")
    result = synth._run_synthesis("Test task", "msg", _positions(), n_samples=1)
    assert mock_client.structured_request.call_count == 1


@patch("epistemic_agents.synthesizer.client")
def test_synthesize_debate_default_n_samples(mock_client):
    """Test that synthesize_debate uses n_samples=1 by default."""
    mock_client.structured_request.return_value = _synthesis()

    synth = Synthesizer(model="opus")
    rounds = [_positions()]
    result = synth.synthesize_debate("Test task", rounds)
    assert mock_client.structured_request.call_count == 1


@patch("epistemic_agents.synthesizer.client")
def test_synthesis_without_beliefs(mock_client):
    """Test synthesis works fine when no beliefs are populated."""
    mock_client.structured_request.return_value = _synthesis()

    positions = [
        ProviderPosition(
            provider_name="claude",
            model_id="opus",
            beliefs=[],  # Empty beliefs
            raw_analysis="Some analysis",
        ),
    ]
    synth = Synthesizer(model="opus")
    result = synth.synthesize("Test task", positions)

    assert result is not None
    # Should not have programmatic analysis injected
    call_args = mock_client.structured_request.call_args
    user_msg = call_args.kwargs.get("user_message", call_args[1].get("user_message", ""))
    assert "PROGRAMMATIC ANALYSIS" not in user_msg
