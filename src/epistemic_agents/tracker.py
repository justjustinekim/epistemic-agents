"""Cost and contribution tracking with passive meta-learning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class ProviderUsage(BaseModel):
    """Token usage for a single provider call."""

    provider_name: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ProviderContribution(BaseModel):
    """Tracks a provider's contribution quality in a session."""

    provider_name: str
    model_id: str
    unique_insights: int = Field(
        default=0, description="Blind spots or unique insights attributed to this provider"
    )
    agreements_participated: int = Field(
        default=0, description="Number of agreement points this provider supported"
    )
    tensions_involved: int = Field(
        default=0, description="Number of tensions this provider was part of"
    )
    was_referenced_in_synthesis: bool = Field(
        default=False, description="Whether the final synthesis specifically cited this provider"
    )


class SessionStats(BaseModel):
    """Aggregated stats for a single session."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_summary: str = Field(default="", description="First 100 chars of task")
    tier: str = Field(default="unknown")
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    provider_usage: list[ProviderUsage] = Field(default_factory=list)
    provider_contributions: list[ProviderContribution] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    model_count: int = 0


class UsageTracker:
    """Tracks token usage and provider contributions across sessions."""

    # Approximate per-token costs (USD) — updated as pricing changes
    _COST_PER_1K: dict[str, tuple[float, float]] = {
        # (input_per_1k, output_per_1k)
        "opus": (0.015, 0.075),
        "sonnet": (0.003, 0.015),
        "haiku": (0.00025, 0.00125),
        "gemini-2.0-flash": (0.0, 0.0),  # Free tier
        "grok-3": (0.003, 0.015),
        "grok-3-mini": (0.0003, 0.0005),
        "deepseek-reasoner": (0.00055, 0.00219),
        "qwq-plus": (0.0, 0.0),  # Free tier (1M tokens)
        "gpt-4o-mini": (0.00015, 0.0006),
        "o3": (0.01, 0.04),
    }

    def __init__(self, path: str | Path = ".epistemic_usage.json"):
        self._path = Path(path)
        self._sessions: list[SessionStats] = []
        self._current_usage: list[ProviderUsage] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._sessions = [SessionStats.model_validate(s) for s in data]

    def _save(self) -> None:
        data = [s.model_dump(mode="json") for s in self._sessions]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def record_usage(
        self,
        provider_name: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ProviderUsage:
        """Record token usage for a single provider call."""
        costs = self._COST_PER_1K.get(model_id, (0.001, 0.005))
        cost = (input_tokens / 1000 * costs[0]) + (output_tokens / 1000 * costs[1])

        usage = ProviderUsage(
            provider_name=provider_name,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )
        self._current_usage.append(usage)
        return usage

    def estimate_session_tokens(
        self,
        task: str,
        tier: str,
        model_count: int,
        rounds: int = 1,
    ) -> int:
        """Rough estimate of tokens for a session (for display before running)."""
        base_input = len(task.split()) * 1.3  # rough word-to-token ratio
        if tier == "quick":
            return int(base_input + 2000)  # single call
        elif tier == "standard":
            return int((base_input + 4000) * 3)  # ~3 loop iterations
        else:  # deep
            per_model = base_input + 3000  # each model's analysis
            debate = per_model * model_count * rounds * 2  # debate rounds
            synthesis = 8000  # synthesis + refutation
            loop = (base_input + 4000) * 3  # thinker-executor
            return int(debate + synthesis + loop)

    def finalize_session(
        self,
        task: str,
        tier: str,
        elapsed: float = 0.0,
        model_count: int = 0,
        contributions: list[ProviderContribution] | None = None,
    ) -> SessionStats:
        """Finalize and persist the current session's stats."""
        total_tokens = sum(u.total_tokens for u in self._current_usage)
        total_cost = sum(u.estimated_cost_usd for u in self._current_usage)

        stats = SessionStats(
            task_summary=task[:100],
            tier=tier,
            total_tokens=total_tokens,
            estimated_cost_usd=total_cost,
            provider_usage=list(self._current_usage),
            provider_contributions=contributions or [],
            elapsed_seconds=elapsed,
            model_count=model_count,
        )

        self._sessions.append(stats)
        self._current_usage = []
        self._save()
        return stats

    def extract_contributions(
        self,
        synthesis: "PanelSynthesis",  # noqa: F821 — avoid circular import
    ) -> list[ProviderContribution]:
        """Extract provider contribution metrics from a panel synthesis."""
        from epistemic_agents.schema import PanelSynthesis

        provider_names: set[str] = set()
        for pos in synthesis.provider_positions:
            provider_names.add(pos.provider_name)

        contribs: dict[str, ProviderContribution] = {}
        for pos in synthesis.provider_positions:
            contribs[pos.provider_name] = ProviderContribution(
                provider_name=pos.provider_name,
                model_id=pos.model_id,
            )

        # Count unique insights
        for ui in synthesis.unique_insights:
            if ui.source_provider in contribs:
                contribs[ui.source_provider].unique_insights += 1

        # Count blind spots
        for bs in synthesis.blind_spots:
            if bs.identified_by in contribs:
                contribs[bs.identified_by].unique_insights += 1

        # Count agreement participation
        for ag in synthesis.agreements:
            for prov in ag.supporting_providers:
                if prov in contribs:
                    contribs[prov].agreements_participated += 1

        # Count tension involvement
        for t in synthesis.tensions:
            for prov in t.positions:
                if prov in contribs:
                    contribs[prov].tensions_involved += 1

        # Check if referenced in synthesis text
        strategy_lower = synthesis.synthesized_strategy.lower()
        for name, c in contribs.items():
            if name.lower() in strategy_lower:
                c.was_referenced_in_synthesis = True

        return list(contribs.values())

    def cumulative_summary(self) -> str:
        """Generate a summary of all sessions for meta-learning."""
        if not self._sessions:
            return "No usage data yet."

        total_sessions = len(self._sessions)
        total_tokens = sum(s.total_tokens for s in self._sessions)
        total_cost = sum(s.estimated_cost_usd for s in self._sessions)
        tier_counts = {}
        for s in self._sessions:
            tier_counts[s.tier] = tier_counts.get(s.tier, 0) + 1

        # Provider contribution leaderboard
        provider_insights: dict[str, int] = {}
        for s in self._sessions:
            for c in s.provider_contributions:
                name = c.provider_name
                provider_insights[name] = (
                    provider_insights.get(name, 0) + c.unique_insights
                )

        lines = [
            f"USAGE SUMMARY ({total_sessions} sessions):",
            f"  Total tokens: ~{total_tokens:,}",
            f"  Estimated cost: ${total_cost:.4f}",
            f"  Tiers: {', '.join(f'{k}: {v}' for k, v in tier_counts.items())}",
        ]

        if provider_insights:
            lines.append("  Provider unique insights:")
            for name, count in sorted(
                provider_insights.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"    {name}: {count}")

        return "\n".join(lines)

    @property
    def sessions(self) -> list[SessionStats]:
        return list(self._sessions)
