"""ModelPanel — parallel multi-model orchestration with debate."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.schema import PanelSynthesis, ProviderPosition

PANEL_SYSTEM_PROMPT = """\
You are participating in a multi-model analysis panel. Your job is to provide a deep, \
honest analysis of the given task.

Be specific and substantive. State your key beliefs explicitly, along with your \
confidence level (high / moderate / low / speculative) and what assumptions you're making. \
For each belief, explain what would prove it wrong.

Structure your analysis as:
1. **Key Beliefs** — Your core conclusions with confidence levels and assumptions
2. **Strategic Approach** — Your recommended path forward with concrete steps
3. **Risks & Uncertainties** — What could go wrong, what you're unsure about
4. **Assumptions** — What you're taking for granted that might not hold

Do NOT hedge everything — take clear positions where you have conviction. \
Mark speculative ideas clearly as speculative."""

DEBATE_SYSTEM_PROMPT = """\
You are in a multi-model debate. Other AI models have analyzed the same task. \
You've seen their positions. Now respond directly.

Your job:
- **Challenge** — Where do you disagree? Be specific. Don't be polite about it.
- **Build** — Where did another model say something genuinely good that you missed? \
  Acknowledge it and build on it.
- **Defend** — If your original position was challenged, defend it with new arguments \
  or concede if they're right.
- **Sharpen** — What's the crux of the disagreement? What's the one question that, \
  if answered, would resolve the debate?

Do NOT repeat your original analysis. Respond to the SPECIFIC claims other models made. \
Reference them by name. This is a conversation, not parallel monologues."""

REFUTATION_SYSTEM_PROMPT = """\
You are reviewing a synthesis produced by a separate AI after a multi-model debate \
you participated in. The synthesizer claims to have distilled your debate into \
agreements, tensions, blind spots, and a final strategy.

Your job is to CHALLENGE the synthesis:
- **Misrepresentation** — Did the synthesizer mischaracterize your position or \
  another model's position? Be specific about what was wrong.
- **False consensus** — Did the synthesizer claim agreement where real disagreement \
  still exists? Did it smooth over tensions you think are unresolved?
- **Dismissed too quickly** — Did the synthesizer reject a position or insight \
  that actually had merit? Defend it with new arguments.
- **Missing from synthesis** — What important point from the debate was left out \
  of the final strategy entirely?
- **Synthesizer bias** — Is the synthesizer favoring its own model family's \
  positions? Is it being too diplomatic when it should take a harder stance?

If the synthesis is genuinely good and fair, say so — but be specific about why. \
Don't rubber-stamp it. This is your chance to correct the record before the \
final output is produced.

Be direct and adversarial. The point of this round is accountability."""


class ModelPanel:
    """Run multiple models in parallel on the same task, with optional debate."""

    def __init__(self, providers: list[BaseProvider]) -> None:
        self.providers = [p for p in providers if p.available]
        if not self.providers:
            raise ValueError("No available providers configured")

    def run(self, task: str, system_prompt: str | None = None) -> list[ProviderPosition]:
        """Query all providers in parallel and return their positions."""
        prompt = system_prompt or PANEL_SYSTEM_PROMPT
        return self._parallel_query(task, prompt)

    def debate(
        self,
        task: str,
        rounds: int = 2,
        on_round: callable | None = None,
        initial_system_prompt: str | None = None,
    ) -> list[list[ProviderPosition]]:
        """Multi-round debate where models respond to each other's analyses.

        Round 1: Independent analysis (parallel).
        Round 2+: Each model sees all previous analyses and responds (parallel).

        Args:
            task: The task to analyze.
            rounds: Total rounds (including initial analysis). Minimum 2.
            on_round: Optional callback(round_num, positions) called after each round.
            initial_system_prompt: Optional system prompt override for Round 1 only.
                Useful for injecting RAG context into the initial analysis.

        Returns:
            List of position lists, one per round.
        """
        rounds = max(2, rounds)
        all_rounds: list[list[ProviderPosition]] = []

        # Round 1: Independent analysis
        positions = self.run(task, system_prompt=initial_system_prompt)
        all_rounds.append(positions)
        if on_round:
            on_round(1, positions)

        # Subsequent rounds: models respond to each other
        for round_num in range(2, rounds + 1):
            debate_prompt = self._build_debate_context(task, all_rounds)
            positions = self._parallel_query(debate_prompt, DEBATE_SYSTEM_PROMPT)
            all_rounds.append(positions)
            if on_round:
                on_round(round_num, positions)

        return all_rounds

    def refute(
        self,
        task: str,
        rounds: list[list[ProviderPosition]],
        synthesis: PanelSynthesis,
    ) -> list[ProviderPosition]:
        """Let the panel challenge the synthesizer's conclusions.

        Each model sees the full debate transcript AND the synthesis output,
        and can challenge misrepresentations, false consensus, or dismissed insights.
        """
        sections = [f"# Original Task\n{task}\n"]

        # Include debate transcript
        for round_num, positions in enumerate(rounds, 1):
            label = (
                "Initial Analysis" if round_num == 1 else f"Debate Round {round_num - 1}"
            )
            sections.append(f"---\n# {label}\n")
            for pos in positions:
                sections.append(
                    f"## {pos.provider_name} ({pos.model_id}):\n"
                    f"{pos.raw_analysis}\n"
                )

        # Include the synthesis
        sections.append("---\n# SYNTHESIZER'S OUTPUT\n")
        sections.append(f"## Agreements\n")
        for ag in synthesis.agreements:
            sections.append(
                f"- [{ag.combined_confidence.value}] {ag.claim} "
                f"(by: {', '.join(ag.supporting_providers)})\n"
            )
        sections.append(f"\n## Tensions\n")
        for t in synthesis.tensions:
            sections.append(f"- {t.claim}\n")
            for prov, stance in t.positions.items():
                sections.append(f"  {prov}: {stance}\n")
            sections.append(f"  Synthesis: {t.synthesis_notes}\n")
        sections.append(f"\n## Blind Spots\n")
        for bs in synthesis.blind_spots:
            sections.append(
                f"- {bs.observation} (caught by {bs.identified_by})\n"
            )
        sections.append(f"\n## Synthesized Strategy\n{synthesis.synthesized_strategy}\n")
        sections.append(f"\n## Meta-Confidence\n{synthesis.meta_confidence}\n")

        sections.append(
            "---\n"
            "Review this synthesis critically. Challenge misrepresentations, "
            "false consensus, dismissed insights, or synthesizer bias. "
            "Defend positions that were unfairly rejected."
        )

        refutation_prompt = "\n".join(sections)
        return self._parallel_query(refutation_prompt, REFUTATION_SYSTEM_PROMPT)

    def _parallel_query(
        self,
        task: str,
        system_prompt: str,
    ) -> list[ProviderPosition]:
        """Query all providers in parallel."""
        positions: list[ProviderPosition] = []

        with ThreadPoolExecutor(max_workers=len(self.providers)) as pool:
            futures = {
                pool.submit(self._query_provider, provider, task, system_prompt): provider
                for provider in self.providers
            }

            for future in as_completed(futures):
                provider = futures[future]
                try:
                    position = future.result()
                    positions.append(position)
                except Exception as exc:
                    print(
                        f"[panel] {provider.name} ({provider.model_id}) failed: {exc}",
                        file=sys.stderr,
                    )

        return positions

    @staticmethod
    def _build_debate_context(
        task: str,
        all_rounds: list[list[ProviderPosition]],
    ) -> str:
        """Build the debate context showing all previous rounds."""
        sections = [f"# Original Task\n{task}\n"]

        for round_num, positions in enumerate(all_rounds, 1):
            label = "Initial Analysis" if round_num == 1 else f"Response Round {round_num - 1}"
            sections.append(f"---\n# {label}\n")
            for pos in positions:
                sections.append(
                    f"## {pos.provider_name} ({pos.model_id}):\n"
                    f"{pos.raw_analysis}\n"
                )

        sections.append(
            "---\n"
            "Now respond to the above. Challenge what you disagree with, "
            "build on what's good, defend or update your positions."
        )
        return "\n".join(sections)

    @staticmethod
    def _query_provider(
        provider: BaseProvider,
        task: str,
        system_prompt: str,
    ) -> ProviderPosition:
        raw = provider.analyze(task, system_prompt)
        return ProviderPosition(
            provider_name=provider.name,
            model_id=provider.model_id,
            beliefs=[],
            raw_analysis=raw,
        )
