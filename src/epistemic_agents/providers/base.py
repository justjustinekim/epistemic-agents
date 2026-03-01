"""Abstract base class for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from epistemic_agents.client import CallUsage


class BaseProvider(ABC):
    """A model provider that can analyze tasks."""

    name: str
    model_id: str
    _last_usage: CallUsage | None = None

    @abstractmethod
    def analyze(self, task: str, system_prompt: str) -> str:
        """Send task to model, return raw text analysis."""

    @property
    def available(self) -> bool:
        """Whether this provider is configured (has API key etc)."""
        return True

    @property
    def last_usage(self) -> CallUsage | None:
        """Token usage from the most recent analyze() call."""
        return self._last_usage
