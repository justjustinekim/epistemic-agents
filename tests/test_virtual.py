"""Tests for virtual panelists — adversarial cognitive roles."""

import pytest

from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.providers.virtual import (
    ROLE_PROMPTS,
    VirtualPanelist,
    create_virtual_panelists,
)
from epistemic_agents.panel import ModelPanel


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """Records the system prompt it receives."""

    def __init__(self, name: str = "fake", model_id: str = "fake-1"):
        self.name = name
        self.model_id = model_id
        self.last_system_prompt: str | None = None
        self.last_task: str | None = None

    def analyze(self, task: str, system_prompt: str) -> str:
        self.last_task = task
        self.last_system_prompt = system_prompt
        return f"[{self.name}] analysis of: {task}"

    @property
    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# VirtualPanelist
# ---------------------------------------------------------------------------


def test_prompt_override():
    """VirtualPanelist should replace the system prompt with the role's prompt."""
    fake = FakeProvider()
    vp = VirtualPanelist(fake, "devil-advocate")

    vp.analyze("test task", "original panel prompt")

    assert fake.last_system_prompt is not None
    # Should contain the role prompt
    assert "Devil's Advocate" in fake.last_system_prompt
    # Should also include the original panel context
    assert "original panel prompt" in fake.last_system_prompt


def test_name_override():
    """VirtualPanelist.name should be the role name, not the provider name."""
    fake = FakeProvider(name="claude")
    vp = VirtualPanelist(fake, "steelman")
    assert vp.name == "steelman"
    assert vp.model_id == "fake-1"


def test_unknown_role_raises_value_error():
    """Unknown roles should raise ValueError."""
    fake = FakeProvider()
    with pytest.raises(ValueError, match="Unknown role"):
        VirtualPanelist(fake, "nonexistent-role")


def test_available_delegates_to_provider():
    """available should delegate to the wrapped provider."""
    fake = FakeProvider()
    vp = VirtualPanelist(fake, "naive-outsider")
    assert vp.available is True


def test_all_roles_have_prompts():
    """All four roles should be defined in ROLE_PROMPTS."""
    expected = {"devil-advocate", "steelman", "assumption-hunter", "naive-outsider"}
    assert set(ROLE_PROMPTS.keys()) == expected


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_creates_all_roles():
    """create_virtual_panelists should create one panelist per role by default."""
    fake = FakeProvider()
    panelists = create_virtual_panelists(fake)
    assert len(panelists) == 4
    names = {p.name for p in panelists}
    assert names == {"devil-advocate", "steelman", "assumption-hunter", "naive-outsider"}


def test_factory_subset_roles():
    """Factory with explicit roles creates only those roles."""
    fake = FakeProvider()
    panelists = create_virtual_panelists(fake, roles=["steelman", "naive-outsider"])
    assert len(panelists) == 2
    names = {p.name for p in panelists}
    assert names == {"steelman", "naive-outsider"}


# ---------------------------------------------------------------------------
# Panel integration
# ---------------------------------------------------------------------------


def test_panel_accepts_virtual_panelists():
    """ModelPanel should accept VirtualPanelists like any other provider."""
    fake = FakeProvider()
    panelists = create_virtual_panelists(fake, roles=["devil-advocate", "steelman"])
    panel = ModelPanel(panelists)
    assert len(panel.providers) == 2

    positions = panel.run("test question")
    assert len(positions) == 2
    names = {p.provider_name for p in positions}
    assert names == {"devil-advocate", "steelman"}


def test_panel_mixed_real_and_virtual():
    """Panel should work with a mix of real and virtual providers."""
    real = FakeProvider(name="claude", model_id="opus")
    virtual = create_virtual_panelists(
        FakeProvider(name="claude-virt", model_id="opus"),
        roles=["devil-advocate"],
    )
    panel = ModelPanel([real] + virtual)
    assert len(panel.providers) == 2
