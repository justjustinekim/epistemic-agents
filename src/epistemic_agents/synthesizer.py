"""Synthesizer — cross-model synthesis via Claude Opus."""

from __future__ import annotations

from epistemic_agents import client
from epistemic_agents.schema import PanelSynthesis, ProviderPosition

SYNTHESIS_SYSTEM_PROMPT = """\
You are the synthesis engine of a multi-model analysis panel. Multiple AI models \
(from different families — Claude, Gemini, GPT, Grok) have debated the same task \
across multiple rounds. Your job is to produce a deep cross-model synthesis.

Instructions:
1. **Agreements** — Find genuine points of convergence (not just surface-level \
   similarity in wording). Where multiple models independently reach the same conclusion, \
   that signal is stronger. Pay attention to agreements that DEEPENED through debate.
2. **Tensions** — Identify real disagreements (not just different emphasis or wording). \
   For each tension, note how the debate evolved — did models move toward each other, \
   or did positions harden? Which position is more compelling after the full exchange?
3. **Blind Spots** — What did one model catch that others missed entirely? Did the \
   debate surface anything that no model raised in the initial round?
4. **Unique Insights** — Novel framings, connections, or ideas that only one model \
   contributed. Were any unique insights validated or challenged by others in debate?
5. **Synthesized Strategy** — Produce a final strategy that incorporates the best of \
   all perspectives. Don't just average — make hard choices where models disagree. \
   The debate should have sharpened where the real cruxes are.
6. **Meta-Confidence** — How confident are you in the synthesis? Where is it weakest? \
   Did the debate resolve key uncertainties or expose new ones?

Be honest about uncertainty. If models disagree on something important, don't paper \
over it — surface the tension and explain your reasoning for the position you take."""

RESYNTHESIS_SYSTEM_PROMPT = """\
You are the synthesis engine. You previously produced a synthesis of a multi-model \
debate. The panel models have now reviewed your synthesis and submitted refutations — \
challenges to your conclusions, defenses of positions you dismissed, and corrections \
to misrepresentations.

Review the refutations carefully and produce a REVISED synthesis:
- If a model convincingly argues you misrepresented their position, correct it.
- If a model defends a position you dismissed and their new arguments have merit, \
  reconsider. You don't have to change your mind, but engage with the argument.
- If multiple models independently flag the same issue with your synthesis, take \
  that seriously — it's likely a real problem.
- If refutations are weak or just restating original positions without new arguments, \
  hold your ground and explain why.

This is the FINAL output. Make it count. The goal is the most accurate, fair, and \
useful synthesis possible — not to please the panel models."""


class Synthesizer:
    """Synthesize multiple model analyses into a unified PanelSynthesis."""

    def __init__(self, model: str = "opus") -> None:
        self.model = model

    def synthesize(
        self,
        task: str,
        positions: list[ProviderPosition],
    ) -> PanelSynthesis:
        """Synthesize a single round of positions."""
        sections = [f"# Task\n{task}\n"]
        for pos in positions:
            sections.append(
                f"## Analysis from {pos.provider_name} ({pos.model_id})\n"
                f"{pos.raw_analysis}\n"
            )
        sections.append(
            "---\n"
            "Now synthesize the above analyses. Identify agreements, tensions, "
            "blind spots, and unique insights. Produce a final synthesized strategy."
        )
        return self._run_synthesis(task, "\n".join(sections), positions)

    def synthesize_debate(
        self,
        task: str,
        rounds: list[list[ProviderPosition]],
    ) -> PanelSynthesis:
        """Synthesize a full multi-round debate transcript."""
        sections = [f"# Task\n{task}\n"]

        for round_num, positions in enumerate(rounds, 1):
            label = "Initial Analysis" if round_num == 1 else f"Debate Round {round_num - 1}"
            sections.append(f"---\n# {label}\n")
            for pos in positions:
                sections.append(
                    f"## {pos.provider_name} ({pos.model_id}):\n"
                    f"{pos.raw_analysis}\n"
                )

        sections.append(
            "---\n"
            "Synthesize the FULL debate above — initial positions AND how they evolved "
            "through direct exchange. Where did models change their minds? Where did "
            "positions harden? What emerged from the conversation that wasn't in any "
            "initial analysis?"
        )

        # Flatten all positions for the synthesis result
        all_positions = [pos for round_positions in rounds for pos in round_positions]
        # Use only the final round's positions as the canonical provider_positions
        final_positions = rounds[-1] if rounds else []

        return self._run_synthesis(task, "\n".join(sections), final_positions)

    def resynthesize(
        self,
        task: str,
        rounds: list[list[ProviderPosition]],
        original_synthesis: PanelSynthesis,
        refutations: list[ProviderPosition],
    ) -> PanelSynthesis:
        """Re-synthesize after panel refutations of the initial synthesis."""
        sections = [f"# Task\n{task}\n"]

        # Include debate summary (condensed)
        sections.append("# Debate Summary\n")
        for round_num, positions in enumerate(rounds, 1):
            label = (
                "Initial Analysis" if round_num == 1 else f"Debate Round {round_num - 1}"
            )
            sections.append(f"## {label}\n")
            for pos in positions:
                sections.append(
                    f"### {pos.provider_name} ({pos.model_id}):\n"
                    f"{pos.raw_analysis}\n"
                )

        # Include original synthesis
        sections.append("---\n# Your Previous Synthesis\n")
        sections.append(f"Strategy: {original_synthesis.synthesized_strategy}\n")
        sections.append(
            f"Meta-Confidence: {original_synthesis.meta_confidence}\n"
        )

        # Include refutations
        sections.append("---\n# Panel Refutations\n")
        for pos in refutations:
            sections.append(
                f"## {pos.provider_name} ({pos.model_id}):\n"
                f"{pos.raw_analysis}\n"
            )

        sections.append(
            "---\n"
            "The panel has challenged your synthesis. Review their refutations "
            "and produce a REVISED final synthesis. Correct errors, engage with "
            "strong counterarguments, but hold your ground where refutations are weak."
        )

        final_positions = rounds[-1] if rounds else []
        return self._run_synthesis(
            task,
            "\n".join(sections),
            final_positions,
            system=RESYNTHESIS_SYSTEM_PROMPT,
        )

    def _run_synthesis(
        self,
        task: str,
        user_message: str,
        positions: list[ProviderPosition],
        system: str = SYNTHESIS_SYSTEM_PROMPT,
    ) -> PanelSynthesis:
        synthesis = client.structured_request(
            model=self.model,
            system=system,
            user_message=user_message,
            response_model=PanelSynthesis,
        )
        synthesis.provider_positions = positions
        return synthesis
