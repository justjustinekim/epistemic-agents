"""Claude provider — wraps existing client.py CLI calls."""

from __future__ import annotations

from epistemic_agents.providers.base import BaseProvider
from epistemic_agents import client


class ClaudeProvider(BaseProvider):
    """Claude via the CLI. Always available (uses Max plan)."""

    def __init__(self, model_id: str = "opus") -> None:
        self.name = "claude"
        self.model_id = model_id

    def analyze(self, task: str, system_prompt: str) -> str:
        return client.plain_request(
            model=self.model_id,
            system=system_prompt,
            user_message=task,
        )

    @property
    def available(self) -> bool:
        return True
