"""ModelPanel — parallel multi-model orchestration with debate."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from epistemic_agents.prediction_market import PairwiseTracker
from epistemic_agents.providers.base import BaseProvider
from epistemic_agents.schema import PanelSynthesis, ProviderPosition

# Per-provider timeout for parallel queries (seconds).
# Individual provider calls (API or CLI) have their own timeouts;
# this is the outer bound on waiting for all futures.
_PROVIDER_TIMEOUT = int(os.environ.get("PANEL_PROVIDER_TIMEOUT", "600"))

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

# C16: Adversarial persona lenses injected into debate rounds to decorrelate reasoning.
# Each provider gets a different lens per round, rotating through all lenses.
DEBATE_PERSONAS: list[str] = [
    (
        "LENS: You are the Empiricist. Demand concrete evidence for every claim. "
        "Reject arguments from authority or analogy unless backed by data. "
        "Ask: what experiment or observation would settle this?"
    ),
    (
        "LENS: You are the Systems Thinker. Focus on second-order effects, feedback loops, "
        "and emergent behavior. What interactions between components does everyone else ignore? "
        "Where do local optima create global failures?"
    ),
    (
        "LENS: You are the Historian. What precedents exist for this situation? "
        "Where have similar approaches succeeded or failed before? "
        "Challenge novelty claims — most 'new' problems have old solutions."
    ),
    (
        "LENS: You are the Adversary. Assume the current consensus is wrong and construct "
        "the strongest possible case against it. What would a smart critic say? "
        "Find the weakest link in the argument chain."
    ),
    (
        "LENS: You are the Pragmatist. Cut through theoretical elegance — what actually works? "
        "What are the real-world constraints everyone is ignoring? "
        "Simplify: what's the minimum viable approach?"
    ),
    (
        "LENS: You are the Edge Case Hunter. Find the scenarios where the consensus breaks down. "
        "What boundary conditions haven't been tested? Where does the model fail gracefully "
        "vs catastrophically?"
    ),
    (
        "LENS: You are the Bayesian Updater. What's the prior probability of each claim? "
        "How much should the evidence presented actually shift our beliefs? "
        "Flag where confidence exceeds what the evidence supports."
    ),
]


def _get_persona_for_round(provider_index: int, round_num: int) -> str:
    """Return a persona lens for a provider in a given round, rotating to avoid repeats."""
    n = len(DEBATE_PERSONAS)
    # Rotate: each round shifts the assignment so no provider gets the same lens twice
    idx = (provider_index + round_num - 2) % n
    return DEBATE_PERSONAS[idx]


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

    def __init__(
        self,
        providers: list[BaseProvider],
        extract_beliefs: bool = True,
        extraction_model: str = "haiku",
        pairwise_tracker: PairwiseTracker | None = None,
    ) -> None:
        self.providers = [p for p in providers if p.available]
        if not self.providers:
            raise ValueError("No available providers configured")
        self._extract_beliefs = extract_beliefs
        self._extraction_model = extraction_model
        self.pairwise_tracker = pairwise_tracker or PairwiseTracker()

    def run(
        self, task: str, system_prompt: str | None = None
    ) -> list[ProviderPosition]:
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
        """Multi-round debate with vote-then-debate optimization.

        Round 1: Independent analysis (parallel).
        After Round 1: majority_vote() locks high-consensus beliefs.
        Round 2+: Only contested beliefs are debated.

        Args:
            task: The task to analyze.
            rounds: Total rounds (including initial analysis). Minimum 2.
            on_round: Optional callback(round_num, positions) called after each round.
            initial_system_prompt: Optional system prompt override for Round 1 only.

        Returns:
            List of position lists, one per round.
        """
        rounds = max(2, rounds)
        all_rounds: list[list[ProviderPosition]] = []

        # Round 1: Independent analysis
        positions = self.run(task, system_prompt=initial_system_prompt)
        all_rounds.append(positions)
        self.pairwise_tracker.record_round(positions)
        if on_round:
            on_round(1, positions)

        # Vote-then-debate: lock high-consensus beliefs after Round 1
        self._locked_agreements = []
        self._contested_beliefs = []
        try:
            from epistemic_agents.agreement_detector import majority_vote

            n_eff = self.pairwise_tracker.n_eff(len(self.providers))
            locked, contested = majority_vote(positions, n_eff)
            self._locked_agreements = locked
            self._contested_beliefs = contested
        except Exception:
            pass

        # Track sycophantic providers for prompt injection
        sycophantic_providers: set[str] = set()

        # Subsequent rounds: models respond to each other
        for round_num in range(2, rounds + 1):
            # Detect sycophancy from previous rounds
            if len(all_rounds) >= 2:
                try:
                    from epistemic_agents.position_tracker import (
                        track_positions,
                        detect_sycophancy,
                    )

                    shifts = track_positions(all_rounds)
                    shifts = detect_sycophancy(shifts, all_rounds)
                    sycophantic_providers = {
                        s.provider_name for s in shifts if s.is_sycophantic
                    }
                except Exception:
                    pass

            # Use targeted prompting if beliefs are populated
            has_beliefs = any(
                pos.beliefs for round_positions in all_rounds for pos in round_positions
            )
            if has_beliefs:
                positions = self._targeted_parallel_query(
                    task,
                    all_rounds,
                    sycophantic_providers=sycophantic_providers,
                    round_num=round_num,
                )
            else:
                debate_prompt = self._build_debate_context(task, all_rounds)
                positions = self._parallel_query(debate_prompt, DEBATE_SYSTEM_PROMPT)
            all_rounds.append(positions)
            self.pairwise_tracker.record_round(positions)
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
                "Initial Analysis"
                if round_num == 1
                else f"Debate Round {round_num - 1}"
            )
            sections.append(f"---\n# {label}\n")
            for pos in positions:
                sections.append(
                    f"## {pos.provider_name} ({pos.model_id}):\n{pos.raw_analysis}\n"
                )

        # Include the synthesis
        sections.append("---\n# SYNTHESIZER'S OUTPUT\n")
        sections.append("## Agreements\n")
        for ag in synthesis.agreements:
            sections.append(
                f"- [{ag.combined_confidence.value}] {ag.claim} "
                f"(by: {', '.join(ag.supporting_providers)})\n"
            )
        sections.append("\n## Tensions\n")
        for t in synthesis.tensions:
            sections.append(f"- {t.claim}\n")
            for prov, stance in t.positions.items():
                sections.append(f"  {prov}: {stance}\n")
            sections.append(f"  Synthesis: {t.synthesis_notes}\n")
        sections.append("\n## Blind Spots\n")
        for bs in synthesis.blind_spots:
            sections.append(f"- {bs.observation} (caught by {bs.identified_by})\n")
        sections.append(
            f"\n## Synthesized Strategy\n{synthesis.synthesized_strategy}\n"
        )
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
        """Query all providers in parallel with timeout protection."""
        positions: list[ProviderPosition] = []

        with ThreadPoolExecutor(max_workers=len(self.providers)) as pool:
            futures = {
                pool.submit(
                    self._query_provider,
                    provider,
                    task,
                    system_prompt,
                    self._extract_beliefs,
                    self._extraction_model,
                ): provider
                for provider in self.providers
            }

            try:
                for future in as_completed(futures, timeout=_PROVIDER_TIMEOUT):
                    provider = futures[future]
                    try:
                        position = future.result()
                        positions.append(position)
                    except Exception as exc:
                        print(
                            f"[panel] {provider.name} ({provider.model_id}) failed: {exc}",
                            file=sys.stderr,
                        )
            except TimeoutError:
                timed_out = [p.name for f, p in futures.items() if not f.done()]
                print(
                    f"[panel] Timed out waiting for providers: {', '.join(timed_out)}",
                    file=sys.stderr,
                )
                for f in futures:
                    if not f.done():
                        f.cancel()

        if not positions:
            print(
                "[panel] WARNING: All providers failed — no positions returned",
                file=sys.stderr,
            )

        return positions

    def _targeted_parallel_query(
        self,
        task: str,
        all_rounds: list[list[ProviderPosition]],
        sycophantic_providers: set[str] | None = None,
        round_num: int = 2,
    ) -> list[ProviderPosition]:
        """Query each provider with a targeted prompt specific to them.

        C16: Each provider gets a rotating adversarial persona lens to
        decorrelate reasoning paths across the panel.
        """
        positions: list[ProviderPosition] = []
        sycophantic_providers = sycophantic_providers or set()

        with ThreadPoolExecutor(max_workers=len(self.providers)) as pool:
            futures = {}
            for i, provider in enumerate(self.providers):
                targeted_prompt = self._build_targeted_debate_context(
                    task, all_rounds, provider.name
                )
                # C16: Inject rotating persona lens
                persona = _get_persona_for_round(i, round_num)
                system_prompt = f"{persona}\n\n{DEBATE_SYSTEM_PROMPT}"

                # Inject anti-sycophancy warning for flagged providers
                if provider.name in sycophantic_providers:
                    targeted_prompt += (
                        "\n\nWARNING: Your previous response appeared to adopt "
                        "another model's position without substantive new reasoning. "
                        "Any position change MUST include a NEW argument not "
                        "previously stated by any participant."
                    )
                futures[
                    pool.submit(
                        self._query_provider,
                        provider,
                        targeted_prompt,
                        system_prompt,
                        self._extract_beliefs,
                        self._extraction_model,
                    )
                ] = provider

            try:
                for future in as_completed(futures, timeout=_PROVIDER_TIMEOUT):
                    provider = futures[future]
                    try:
                        position = future.result()
                        positions.append(position)
                    except Exception as exc:
                        print(
                            f"[panel] {provider.name} ({provider.model_id}) failed: {exc}",
                            file=sys.stderr,
                        )
            except TimeoutError:
                timed_out = [p.name for f, p in futures.items() if not f.done()]
                print(
                    f"[panel] Timed out waiting for providers: {', '.join(timed_out)}",
                    file=sys.stderr,
                )
                for f in futures:
                    if not f.done():
                        f.cancel()

        return positions

    @staticmethod
    def _build_targeted_debate_context(
        task: str,
        all_rounds: list[list[ProviderPosition]],
        target_provider: str,
    ) -> str:
        """Build debate context targeted for a specific provider.

        For rounds 2+, uses state deltas instead of full text for efficiency.
        """

        sections = [f"# Original Task\n{task}\n"]

        # Find target's latest position
        target_pos = None
        for round_positions in reversed(all_rounds):
            for pos in round_positions:
                if pos.provider_name == target_provider:
                    target_pos = pos
                    break
            if target_pos:
                break

        if target_pos:
            sections.append(
                f"---\n# YOUR PREVIOUS POSITION ({target_provider}):\n"
                f"{target_pos.raw_analysis[:2000]}\n"
            )

        # Use state deltas for rounds 2+ when beliefs are available
        has_beliefs = any(
            pos.beliefs for round_positions in all_rounds for pos in round_positions
        )
        if has_beliefs and len(all_rounds) >= 2:
            try:
                from epistemic_agents.position_tracker import compute_deltas

                deltas = compute_deltas(all_rounds)
                if deltas:
                    sections.append(f"---\n# STATE CHANGES:\n{deltas}\n")
            except Exception:
                pass

        # Find counterarguments from other models (use latest round only for efficiency)
        latest_round = all_rounds[-1] if all_rounds else []
        other_latest = [
            pos for pos in latest_round if pos.provider_name != target_provider
        ]

        sections.append("---\n# OTHER MODELS' LATEST POSITIONS:\n")
        for pos in other_latest:
            sections.append(
                f"## {pos.provider_name} ({pos.model_id}):\n"
                f"{pos.raw_analysis[:500]}...\n"
            )

        sections.append(
            "---\n"
            "Address the counterarguments above. Where do you disagree? "
            "Where have other models changed your mind? "
            "What's the crux of remaining disagreements?"
        )
        return "\n".join(sections)

    @staticmethod
    def _build_debate_context(
        task: str,
        all_rounds: list[list[ProviderPosition]],
    ) -> str:
        """Build the debate context showing all previous rounds."""
        sections = [f"# Original Task\n{task}\n"]

        for round_num, positions in enumerate(all_rounds, 1):
            label = (
                "Initial Analysis"
                if round_num == 1
                else f"Response Round {round_num - 1}"
            )
            sections.append(f"---\n# {label}\n")
            for pos in positions:
                sections.append(
                    f"## {pos.provider_name} ({pos.model_id}):\n{pos.raw_analysis}\n"
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
        extract_beliefs: bool = False,
        extraction_model: str = "haiku",
    ) -> ProviderPosition:
        # Try structured output path first if provider supports it
        if extract_beliefs and provider.supports_structured_output:
            try:
                panel_response = provider.structured_analyze(task, system_prompt)
                return ProviderPosition(
                    provider_name=provider.name,
                    model_id=provider.model_id,
                    beliefs=panel_response.beliefs,
                    raw_analysis=panel_response.raw_analysis,
                )
            except Exception as exc:
                print(
                    f"[panel] Structured output failed for {provider.name}, "
                    f"falling back to extraction: {exc}",
                    file=sys.stderr,
                )

        # Fallback: plain analyze + belief extraction
        raw = provider.analyze(task, system_prompt)
        beliefs: list = []

        if extract_beliefs:
            try:
                from epistemic_agents.belief_extractor import (
                    extract_beliefs as _extract,
                )

                beliefs = _extract(raw, provider.name, model=extraction_model)
            except Exception as exc:
                print(
                    f"[panel] Belief extraction failed for {provider.name}: {exc}",
                    file=sys.stderr,
                )

        return ProviderPosition(
            provider_name=provider.name,
            model_id=provider.model_id,
            beliefs=beliefs,
            raw_analysis=raw,
        )
