"""Tournament — compare all three analysis tiers on the same task."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class TournamentResult(BaseModel):
    """Result of running all 3 tiers on the same task."""

    task: str
    results: dict[str, str] = Field(
        description="Mapping of tier name to verdict summary"
    )
    verdict_changed: bool = Field(
        default=False,
        description="Whether the verdict changed across tiers",
    )
    confidence_changed: bool = Field(
        default=False,
        description="Whether confidence level changed across tiers",
    )
    deep_added_value: str = Field(
        default="",
        description="What deep tier added that other tiers missed",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TournamentLog:
    """Persist and analyze tournament results across sessions."""

    def __init__(self, path: str | Path = ".epistemic_tournaments.json"):
        self._path = Path(path)
        self._results: list[TournamentResult] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._results = [TournamentResult.model_validate(r) for r in data]

    def _save(self) -> None:
        data = [r.model_dump(mode="json") for r in self._results]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def record(self, result: TournamentResult) -> None:
        self._results.append(result)
        self._save()

    def value_of_depth_summary(self) -> str:
        """Summarize how often deeper tiers add value."""
        if not self._results:
            return "No tournament data yet."

        total = len(self._results)
        verdict_changes = sum(1 for r in self._results if r.verdict_changed)
        confidence_changes = sum(1 for r in self._results if r.confidence_changed)
        deep_valuable = sum(1 for r in self._results if r.deep_added_value)

        return (
            f"Tournament summary ({total} runs):\n"
            f"  Verdict changed across tiers: {verdict_changes}/{total} ({verdict_changes/total:.0%})\n"
            f"  Confidence changed: {confidence_changes}/{total} ({confidence_changes/total:.0%})\n"
            f"  Deep tier added unique value: {deep_valuable}/{total} ({deep_valuable/total:.0%})"
        )

    @property
    def results(self) -> list[TournamentResult]:
        return list(self._results)


def run_tournament(
    orchestrator: object,
    task: str,
) -> TournamentResult:
    """Run all 3 tiers on the same task and compare results.

    Args:
        orchestrator: An Orchestrator instance.
        task: The task to analyze.

    Returns:
        TournamentResult with comparison across tiers.
    """
    from epistemic_agents.orchestrator import Tier

    tier_results: dict[str, str] = {}
    verdicts: dict[str, object] = {}

    for tier in [Tier.QUICK, Tier.STANDARD, Tier.DEEP]:
        try:
            result = orchestrator.run(task, tier)  # type: ignore[union-attr]
            if result.verdict:
                tier_results[tier.value] = result.verdict.recommendation
                verdicts[tier.value] = result.verdict
            else:
                tier_results[tier.value] = "(no verdict)"
        except Exception as e:
            tier_results[tier.value] = f"(error: {e})"

    # Compare verdicts
    recommendations = list(tier_results.values())
    verdict_changed = len(set(recommendations)) > 1

    confidence_changed = False
    confidences = [
        v.confidence.value for v in verdicts.values()
        if hasattr(v, "confidence")
    ]
    if len(set(confidences)) > 1:
        confidence_changed = True

    # Check what deep tier added
    deep_added_value = ""
    if "deep" in tier_results and "standard" in tier_results:
        if tier_results["deep"] != tier_results["standard"]:
            deep_added_value = (
                f"Deep tier recommendation differs: {tier_results['deep'][:200]}"
            )

    return TournamentResult(
        task=task,
        results=tier_results,
        verdict_changed=verdict_changed,
        confidence_changed=confidence_changed,
        deep_added_value=deep_added_value,
    )
