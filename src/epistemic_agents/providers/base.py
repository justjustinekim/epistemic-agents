"""Abstract base class for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """A model provider that can analyze tasks."""

    name: str
    model_id: str

    @abstractmethod
    def analyze(self, task: str, system_prompt: str) -> str:
        """Send task to model, return raw text analysis."""

    @property
    def available(self) -> bool:
        """Whether this provider is configured (has API key etc)."""
        return True
