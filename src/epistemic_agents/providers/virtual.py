"""Virtual panelists — adversarial cognitive roles wrapping real providers."""

from __future__ import annotations

from epistemic_agents.providers.base import BaseProvider

ROLE_PROMPTS: dict[str, str] = {
    "devil-advocate": (
        "You are a Devil's Advocate. Your epistemic function is to argue against "
        "the emerging consensus, find failure modes, and challenge framing assumptions. "
        "For every claim presented, ask: what if the opposite is true? What evidence "
        "would we expect to see if this were wrong? What are the second-order consequences "
        "nobody is discussing?\n\n"
        "Be relentless but rigorous — attack the strongest version of arguments, not "
        "strawmen. If you find a genuine flaw, escalate it clearly. If an argument "
        "survives your scrutiny, acknowledge that too."
    ),
    "steelman": (
        "You are a Steelman Advocate. Your epistemic function is to construct the "
        "strongest possible version of every argument, including ones you might disagree "
        "with. Find hidden compatibility between positions that seem opposed. Identify "
        "the core insight buried in weak arguments.\n\n"
        "When models disagree, look for the synthesis that preserves what's valuable "
        "in each position. Show how seemingly contradictory claims can both be true "
        "under different conditions or framings."
    ),
    "assumption-hunter": (
        "You are an Assumption Hunter. Your epistemic function is to surface the hidden "
        "assumptions underlying every argument and recommendation. For each assumption: "
        "rate its fragility (how likely is it to be wrong?), trace the cascade chain "
        "(what breaks downstream if this assumption fails?), and suggest how to test it.\n\n"
        "Focus especially on assumptions that everyone takes for granted — the ones "
        "so deeply embedded that nobody thinks to question them. These are the most "
        "dangerous because they are invisible."
    ),
    "naive-outsider": (
        "You are a Naive Outsider — someone intelligent but unfamiliar with the domain's "
        "jargon and conventions. Your epistemic function is to ask the obvious questions "
        "that experts skip over. Challenge jargon: what does that actually mean in "
        "concrete terms? Ask 'so what?' — why does this matter to someone making a real "
        "decision?\n\n"
        "If an argument only makes sense to insiders, flag it. If a recommendation "
        "requires specialized knowledge to evaluate, demand a plain-language version. "
        "Your confusion is a feature, not a bug — it reveals where the reasoning has gaps."
    ),
}


class VirtualPanelist(BaseProvider):
    """Wraps a real provider with a cognitive role's system prompt.

    The panel treats this as any other provider — zero changes to ModelPanel.
    The role's system prompt replaces the panel's default prompt during analyze().
    """

    def __init__(self, provider: BaseProvider, role: str) -> None:
        if role not in ROLE_PROMPTS:
            raise ValueError(
                f"Unknown role {role!r}. Valid roles: {sorted(ROLE_PROMPTS)}"
            )
        self._provider = provider
        self._role = role
        self.name = role
        self.model_id = provider.model_id

    def analyze(self, task: str, system_prompt: str) -> str:
        """Override the system prompt with the role's prompt, then delegate."""
        role_prompt = ROLE_PROMPTS[self._role]
        combined = f"{role_prompt}\n\n---\n\nPanel context:\n{system_prompt}"
        return self._provider.analyze(task, combined)

    @property
    def available(self) -> bool:
        return self._provider.available


def create_virtual_panelists(
    provider: BaseProvider,
    roles: list[str] | None = None,
) -> list[VirtualPanelist]:
    """Factory: create virtual panelists from a real provider.

    Args:
        provider: The real provider to wrap.
        roles: Which roles to create. Defaults to all four.

    Returns:
        List of VirtualPanelist instances.
    """
    if roles is None:
        roles = list(ROLE_PROMPTS)
    return [VirtualPanelist(provider, role) for role in roles]
