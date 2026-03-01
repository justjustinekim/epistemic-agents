"""Tests for belief extraction from raw model analyses."""

from unittest.mock import patch, MagicMock

from epistemic_agents.belief_extractor import (
    ExtractedBeliefs,
    extract_beliefs,
)
from epistemic_agents.schema import Belief, BeliefGrounding, ConfidenceLevel


def _mock_extraction_response() -> ExtractedBeliefs:
    return ExtractedBeliefs(
        beliefs=[
            Belief(
                id="claude-b1",
                claim="Redis is the best cache",
                confidence=ConfidenceLevel.HIGH,
                justification="High throughput",
                falsification_conditions=["If serverless"],
                key_assumptions=["Persistent servers"],
                grounding=BeliefGrounding.SINGLE_MODEL,
            ),
            Belief(
                id="claude-b2",
                claim="60s TTL is appropriate",
                confidence=ConfidenceLevel.MODERATE,
                justification="Data updates every few minutes",
                grounding=BeliefGrounding.SINGLE_MODEL,
            ),
        ]
    )


@patch("epistemic_agents.belief_extractor.client")
def test_extract_beliefs_basic(mock_client):
    mock_client.structured_request.return_value = _mock_extraction_response()

    beliefs = extract_beliefs("Some analysis text", "claude")
    assert len(beliefs) == 2
    assert beliefs[0].id == "claude-b1"
    assert beliefs[0].grounding == BeliefGrounding.SINGLE_MODEL
    mock_client.structured_request.assert_called_once()


@patch("epistemic_agents.belief_extractor.client")
def test_extract_beliefs_grounding_forced(mock_client):
    # Even if the model returns wrong grounding, we override
    response = _mock_extraction_response()
    response.beliefs[0].grounding = BeliefGrounding.EMPIRICAL
    mock_client.structured_request.return_value = response

    beliefs = extract_beliefs("Analysis", "test-model")
    assert all(b.grounding == BeliefGrounding.SINGLE_MODEL for b in beliefs)


@patch("epistemic_agents.belief_extractor.client")
def test_extract_beliefs_id_prefixing(mock_client):
    response = ExtractedBeliefs(
        beliefs=[
            Belief(
                id="b1",  # Missing prefix
                claim="Test",
                confidence=ConfidenceLevel.HIGH,
                justification="Test",
            ),
        ]
    )
    mock_client.structured_request.return_value = response

    beliefs = extract_beliefs("Analysis", "gemini")
    assert beliefs[0].id == "gemini-b1"


@patch("epistemic_agents.belief_extractor.client")
def test_extract_beliefs_already_prefixed(mock_client):
    response = ExtractedBeliefs(
        beliefs=[
            Belief(
                id="gemini-b1",  # Already prefixed
                claim="Test",
                confidence=ConfidenceLevel.HIGH,
                justification="Test",
            ),
        ]
    )
    mock_client.structured_request.return_value = response

    beliefs = extract_beliefs("Analysis", "gemini")
    assert beliefs[0].id == "gemini-b1"


@patch("epistemic_agents.belief_extractor.client")
def test_extract_beliefs_custom_model(mock_client):
    mock_client.structured_request.return_value = _mock_extraction_response()

    extract_beliefs("Analysis", "claude", model="sonnet")
    call_kwargs = mock_client.structured_request.call_args
    assert call_kwargs.kwargs.get("model") == "sonnet" or call_kwargs[1].get("model") == "sonnet"
