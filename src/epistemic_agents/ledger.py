"""BeliefLedger — cross-session belief tracking and calibration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from epistemic_agents.schema import (
    Belief,
    ConfidenceLevel,
    ConversationLog,
    ExecutorFeedback,
    StrategicHandoff,
)


class BeliefOutcome(str, Enum):
    """What happened to a belief after execution."""

    CONFIRMED = "confirmed"  # Evidence supported the belief
    FALSIFIED = "falsified"  # Evidence contradicted the belief
    REVISED = "revised"  # Belief was updated during the loop
    UNTESTED = "untested"  # Never got evidence either way


class BeliefRecord(BaseModel):
    """A single belief's outcome from one session."""

    belief_id: str
    claim: str
    confidence: ConfidenceLevel
    outcome: BeliefOutcome
    failure_reason: str = Field(
        default="",
        description="Why the belief failed, if falsified or revised",
    )
    task_summary: str = Field(
        default="",
        description="Brief description of the task this belief was part of",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationStats(BaseModel):
    """Aggregated accuracy stats by confidence level."""

    level: ConfidenceLevel
    total: int = 0
    confirmed: int = 0
    falsified: int = 0
    revised: int = 0
    untested: int = 0

    @property
    def tested(self) -> int:
        return self.confirmed + self.falsified + self.revised

    @property
    def accuracy(self) -> float | None:
        """Fraction of tested beliefs that were confirmed."""
        if self.tested == 0:
            return None
        return self.confirmed / self.tested


class BeliefLedger:
    """Tracks belief outcomes across sessions for calibration."""

    def __init__(self, path: str | Path = ".epistemic_ledger.json"):
        self._path = Path(path)
        self._records: list[BeliefRecord] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._records = [BeliefRecord.model_validate(r) for r in data]

    def _save(self) -> None:
        data = [r.model_dump(mode="json") for r in self._records]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def record_outcomes(self, log: ConversationLog) -> list[BeliefRecord]:
        """Extract belief outcomes from a completed conversation log and record them."""
        if not log.entries:
            return []

        # Get the initial handoff
        handoff = None
        for entry in log.entries:
            if entry.entry_type == "handoff" and isinstance(
                entry.content, StrategicHandoff
            ):
                handoff = entry.content
                break
        if not handoff:
            return []

        # Collect all challenged belief IDs from executor feedback
        challenged_ids: dict[str, str] = {}  # belief_id -> evidence
        revised_ids: set[str] = set()

        for entry in log.entries:
            if entry.entry_type == "feedback" and isinstance(
                entry.content, ExecutorFeedback
            ):
                for cb in entry.content.challenged_beliefs:
                    challenged_ids[cb.belief_id] = cb.evidence
            if entry.entry_type == "amendment":
                from epistemic_agents.schema import ThinkerAmendment

                if isinstance(entry.content, ThinkerAmendment):
                    for b in entry.content.updated_beliefs:
                        revised_ids.add(b.id)

        # Create records for each original belief
        task_summary = log.task[:100]
        new_records: list[BeliefRecord] = []
        for belief in handoff.beliefs:
            if belief.id in challenged_ids:
                outcome = BeliefOutcome.FALSIFIED
                reason = challenged_ids[belief.id]
            elif belief.id in revised_ids:
                outcome = BeliefOutcome.REVISED
                reason = "Revised during thinker-executor loop"
            elif log.converged:
                outcome = BeliefOutcome.CONFIRMED
                reason = ""
            else:
                outcome = BeliefOutcome.UNTESTED
                reason = ""

            record = BeliefRecord(
                belief_id=belief.id,
                claim=belief.claim,
                confidence=belief.confidence,
                outcome=outcome,
                failure_reason=reason,
                task_summary=task_summary,
            )
            new_records.append(record)

        self._records.extend(new_records)
        self._save()
        return new_records

    def calibration_report(self) -> list[CalibrationStats]:
        """Compute accuracy stats grouped by confidence level."""
        stats: dict[ConfidenceLevel, CalibrationStats] = {}
        for level in ConfidenceLevel:
            stats[level] = CalibrationStats(level=level)

        for record in self._records:
            s = stats[record.confidence]
            s.total += 1
            if record.outcome == BeliefOutcome.CONFIRMED:
                s.confirmed += 1
            elif record.outcome == BeliefOutcome.FALSIFIED:
                s.falsified += 1
            elif record.outcome == BeliefOutcome.REVISED:
                s.revised += 1
            else:
                s.untested += 1

        return [s for s in stats.values() if s.total > 0]

    def calibration_context(self) -> str:
        """Generate a calibration summary for injection into the thinker's prompt."""
        report = self.calibration_report()
        if not report:
            return ""

        total = sum(s.total for s in report)
        if total < 5:
            return ""  # Not enough data to be meaningful

        lines = [
            f"CALIBRATION DATA (from {total} beliefs across previous sessions):"
        ]
        for s in report:
            if s.tested > 0:
                acc = s.accuracy
                acc_str = f"{acc:.0%}" if acc is not None else "N/A"
                lines.append(
                    f"  {s.level.value.upper()}: {s.total} beliefs, "
                    f"{s.confirmed} confirmed, {s.falsified} falsified, "
                    f"{s.revised} revised (accuracy: {acc_str})"
                )

        # Add specific warnings for miscalibration
        for s in report:
            if s.accuracy is not None and s.level == ConfidenceLevel.HIGH and s.accuracy < 0.8:
                lines.append(
                    f"  WARNING: Your HIGH confidence beliefs have only been "
                    f"confirmed {s.accuracy:.0%} of the time. Consider downgrading "
                    f"some HIGH beliefs to MODERATE."
                )
            if (
                s.accuracy is not None
                and s.level == ConfidenceLevel.SPECULATIVE
                and s.accuracy > 0.7
            ):
                lines.append(
                    f"  NOTE: Your SPECULATIVE beliefs have been confirmed "
                    f"{s.accuracy:.0%} of the time. You may be under-confident "
                    f"on some claims."
                )

        return "\n".join(lines)

    @property
    def records(self) -> list[BeliefRecord]:
        return list(self._records)
