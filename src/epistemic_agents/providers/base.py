"""Abstract base class for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from epistemic_agents.client import CallUsage
    from epistemic_agents.schema import PanelResponse


class BaseProvider(ABC):
    """A model provider that can analyze tasks."""

    name: str
    model_id: str
    _last_usage: CallUsage | None = None

    @abstractmethod
    def analyze(self, task: str, system_prompt: str) -> str:
        """Send task to model, return raw text analysis."""

    @property
    def supports_structured_output(self) -> bool:
        """Whether this provider can return structured PanelResponse directly."""
        return False

    def structured_analyze(self, task: str, system_prompt: str) -> PanelResponse:
        """Return structured PanelResponse. Only available if supports_structured_output is True."""
        raise NotImplementedError(
            f"{self.name} does not support structured output"
        )

    @property
    def available(self) -> bool:
        """Whether this provider is configured (has API key etc)."""
        return True

    @property
    def last_usage(self) -> CallUsage | None:
        """Token usage from the most recent analyze() call."""
        return self._last_usage
