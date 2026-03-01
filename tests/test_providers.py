"""Tests for provider construction, config loading, and panel schema models."""

import json
import os
from unittest.mock import patch

from epistemic_agents.schema import (
    AgreementPoint,
    BlindSpot,
    ConfidenceLevel,
    PanelSynthesis,
    ProviderPosition,
    TensionPoint,
    UniqueInsight,
)
from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.providers.claude import ClaudeProvider
from epistemic_agents.providers.openai_compat import OpenAICompatProvider
from epistemic_agents.providers.gemini import GeminiProvider
from epistemic_agents.config import get_available_providers
from epistemic_agents.panel import ModelPanel


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def test_claude_provider_construction():
    p = ClaudeProvider()
    assert p.name == "claude"
    assert p.model_id == "opus"
    assert p.available is True


def test_claude_provider_custom_model():
    p = ClaudeProvider(model_id="sonnet")
    assert p.model_id == "sonnet"


def test_grok_factory():
    p = OpenAICompatProvider.grok(api_key="test-key")
    assert p.name == "grok"
    assert p.model_id == "grok-3"
    assert p.available is True


def test_deepseek_factory():
    p = OpenAICompatProvider.deepseek(api_key="test-key")
    assert p.name == "deepseek"
    assert p.model_id == "deepseek-reasoner"
    assert p.available is True


def test_qwq_factory():
    p = OpenAICompatProvider.qwq(api_key="test-key")
    assert p.name == "qwq"
    assert p.model_id == "qwq-plus"
    assert p.available is True


def test_gpt_factory():
    p = OpenAICompatProvider.gpt(api_key="test-key")
    assert p.name == "gpt"
    assert p.model_id == "gpt-4o-mini"
    assert p.available is True


def test_perplexity_factory():
    p = OpenAICompatProvider.perplexity(api_key="test-key")
    assert p.name == "perplexity"
    assert p.model_id == "sonar-pro"
    assert p.available is True


def test_openai_compat_unavailable_without_key():
    p = OpenAICompatProvider(api_key="", base_url="https://example.com", model_id="x", name="test")
    assert p.available is False


def test_gemini_provider_construction():
    p = GeminiProvider(api_key="test-key")
    assert p.name == "gemini"
    assert p.model_id == "gemini-2.5-flash"
    assert p.available is True


def test_gemini_unavailable_without_key():
    p = GeminiProvider(api_key="")
    assert p.available is False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_config_claude_always_available():
    with patch.dict(os.environ, {}, clear=True):
        providers = get_available_providers()
        assert len(providers) >= 1
        assert providers[0].name == "claude"


def test_config_all_providers():
    env = {
        "GOOGLE_API_KEY": "goog-key",
        "XAI_API_KEY": "xai-key",
        "DEEPSEEK_API_KEY": "ds-key",
        "DASHSCOPE_API_KEY": "dash-key",
        "OPENAI_API_KEY": "oai-key",
        "PERPLEXITY_API_KEY": "pplx-key",
    }
    with patch.dict(os.environ, env, clear=True):
        providers = get_available_providers()
        names = [p.name for p in providers]
        assert "claude" in names
        assert "gemini" in names
        assert "grok" in names
        assert "deepseek" in names
        assert "qwq" in names
        assert "gpt" in names
        assert "perplexity" in names
        assert len(providers) == 7


def test_config_partial_providers():
    env = {"GOOGLE_API_KEY": "goog-key"}
    with patch.dict(os.environ, env, clear=True):
        providers = get_available_providers()
        names = [p.name for p in providers]
        assert "claude" in names
        assert "gemini" in names
        assert "grok" not in names
        assert "gpt" not in names


# ---------------------------------------------------------------------------
# Panel schema models
# ---------------------------------------------------------------------------


def test_provider_position():
    pos = ProviderPosition(
        provider_name="claude",
        model_id="opus",
        raw_analysis="This is a deep analysis.",
    )
    assert pos.provider_name == "claude"
    assert pos.beliefs == []
    assert pos.raw_analysis == "This is a deep analysis."


def test_agreement_point():
    ag = AgreementPoint(
        claim="PLG is the best path for a 3-person startup",
        supporting_providers=["claude", "gemini", "gpt"],
        combined_confidence=ConfidenceLevel.HIGH,
    )
    assert len(ag.supporting_providers) == 3
    assert ag.combined_confidence == ConfidenceLevel.HIGH


def test_tension_point():
    t = TensionPoint(
        claim="Whether to go enterprise or PLG",
        positions={
            "claude": "PLG is better given team size",
            "gpt": "Enterprise gives higher ACV needed for runway",
        },
        synthesis_notes="Both valid — depends on team's sales ability",
    )
    assert len(t.positions) == 2


def test_blind_spot():
    bs = BlindSpot(
        observation="Open source creates acquisition leverage even if revenue is delayed",
        identified_by="gemini",
        missed_by=["claude", "gpt"],
    )
    assert bs.identified_by == "gemini"
    assert len(bs.missed_by) == 2


def test_unique_insight():
    ui = UniqueInsight(
        insight="Combine PLG with strategic open-source to get both community and revenue",
        source_provider="grok",
        relevance="Addresses the tension between growth and revenue directly",
    )
    assert ui.source_provider == "grok"


def test_panel_synthesis_construction():
    synthesis = PanelSynthesis(
        task="Pick a go-to-market strategy",
        provider_positions=[
            ProviderPosition(
                provider_name="claude",
                model_id="opus",
                raw_analysis="Analysis from Claude...",
            ),
        ],
        agreements=[
            AgreementPoint(
                claim="Team size constrains enterprise path",
                supporting_providers=["claude", "gemini"],
                combined_confidence=ConfidenceLevel.HIGH,
            ),
        ],
        tensions=[
            TensionPoint(
                claim="Revenue timeline",
                positions={"claude": "6 months", "gpt": "12 months"},
                synthesis_notes="Depends on pricing model",
            ),
        ],
        synthesized_strategy="Go PLG with enterprise upsell path.",
        meta_confidence="Moderate — limited by single-provider panel in this test.",
    )
    assert len(synthesis.provider_positions) == 1
    assert len(synthesis.agreements) == 1
    assert len(synthesis.tensions) == 1


def test_panel_synthesis_serialization():
    synthesis = PanelSynthesis(
        task="Test task",
        provider_positions=[],
        synthesized_strategy="Strategy here",
        meta_confidence="High",
    )
    json_str = synthesis.model_dump_json(indent=2)
    parsed = json.loads(json_str)
    roundtripped = PanelSynthesis.model_validate(parsed)
    assert roundtripped.task == "Test task"
    assert roundtripped.synthesized_strategy == "Strategy here"


def test_panel_synthesis_json_schema():
    schema = PanelSynthesis.model_json_schema()
    assert "properties" in schema
    assert "task" in schema["properties"]
    assert "synthesized_strategy" in schema["properties"]
    assert "agreements" in schema["properties"]


# ---------------------------------------------------------------------------
# ModelPanel construction
# ---------------------------------------------------------------------------


def test_panel_filters_unavailable_providers():
    available = ClaudeProvider()
    unavailable = GeminiProvider(api_key="")
    panel = ModelPanel([available, unavailable])
    assert len(panel.providers) == 1
    assert panel.providers[0].name == "claude"


def test_panel_rejects_empty_providers():
    try:
        ModelPanel([GeminiProvider(api_key="")])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
