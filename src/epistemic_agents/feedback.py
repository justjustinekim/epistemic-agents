"""Post-session feedback collection and logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class SessionFeedback(BaseModel):
    """User feedback after a session."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_summary: str = Field(description="First 200 chars of the task")
    useful: str = Field(description="yes / partially / no")
    comment: str = Field(default="", description="What would have made it better")
    tier_used: str = Field(default="unknown", description="quick / standard / deep")
    model_count: int = Field(default=1, description="Number of models used")
    duration_seconds: float = Field(default=0.0)


class FeedbackLog:
    """Persists user feedback across sessions."""

    def __init__(self, path: str | Path = ".epistemic_feedback.json"):
        self._path = Path(path)
        self._entries: list[SessionFeedback] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._entries = [SessionFeedback.model_validate(e) for e in data]

    def _save(self) -> None:
        data = [e.model_dump(mode="json") for e in self._entries]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def add(self, feedback: SessionFeedback) -> None:
        self._entries.append(feedback)
        self._save()

    @property
    def entries(self) -> list[SessionFeedback]:
        return list(self._entries)

    def summary(self) -> str:
        """Quick summary of feedback trends."""
        if not self._entries:
            return "No feedback collected yet."
        total = len(self._entries)
        yes = sum(1 for e in self._entries if e.useful == "yes")
        partial = sum(1 for e in self._entries if e.useful == "partially")
        no = sum(1 for e in self._entries if e.useful == "no")
        comments = [e.comment for e in self._entries if e.comment]
        lines = [
            f"Feedback: {total} sessions — {yes} useful, {partial} partially, {no} not useful"
        ]
        if comments:
            lines.append(f"Recent comments:")
            for c in comments[-3:]:
                lines.append(f"  - {c}")
        return "\n".join(lines)


def collect_feedback(
    task: str,
    tier: str = "unknown",
    model_count: int = 1,
    duration: float = 0.0,
) -> SessionFeedback | None:
    """Interactive feedback collection from the terminal.

    Returns the feedback, or None if the user skips.
    """
    print("\n--- Session Feedback ---")
    print("Was this output useful?")
    print("  1) yes")
    print("  2) partially")
    print("  3) no")
    print("  s) skip")

    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice in ("s", "skip", ""):
        return None

    useful_map = {"1": "yes", "yes": "yes", "2": "partially", "partially": "partially", "3": "no", "no": "no"}
    useful = useful_map.get(choice)
    if not useful:
        return None

    comment = ""
    if useful in ("partially", "no"):
        try:
            comment = input("What would have made it better? (enter to skip) > ").strip()
        except (EOFError, KeyboardInterrupt):
            pass

    return SessionFeedback(
        task_summary=task[:200],
        useful=useful,
        comment=comment,
        tier_used=tier,
        model_count=model_count,
        duration_seconds=duration,
    )
