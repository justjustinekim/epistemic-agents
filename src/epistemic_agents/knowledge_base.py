"""Knowledge base — persistent store of confirmed beliefs across sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from epistemic_agents.rag import _tokenize, _jaccard_similarity
from epistemic_agents.schema import Belief, VerificationMethod


class KnowledgeEntry(BaseModel):
    """A confirmed belief stored in the knowledge base."""

    belief: Belief
    task_origin: str = Field(description="The task that produced this belief")
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verification_method: VerificationMethod = Field(
        default=VerificationMethod.UNVERIFIED,
    )
    usage_count: int = Field(
        default=0,
        description="How many times this entry has been retrieved",
    )


class KnowledgeBase:
    """Persistent store of confirmed beliefs for cross-session retrieval."""

    def __init__(self, path: str | Path = ".epistemic_knowledge.json"):
        self._path = Path(path)
        self._entries: list[KnowledgeEntry] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._entries = [KnowledgeEntry.model_validate(e) for e in data]

    def _save(self) -> None:
        data = [e.model_dump(mode="json") for e in self._entries]
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def store(
        self,
        belief: Belief,
        task_origin: str,
        verification_method: VerificationMethod = VerificationMethod.UNVERIFIED,
    ) -> KnowledgeEntry:
        """Store a confirmed belief in the knowledge base."""
        entry = KnowledgeEntry(
            belief=belief,
            task_origin=task_origin,
            verification_method=verification_method,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
    ) -> list[KnowledgeEntry]:
        """Search for relevant knowledge entries by query similarity."""
        query_tokens = _tokenize(query)
        scored: list[tuple[float, KnowledgeEntry]] = []

        for entry in self._entries:
            entry_tokens = _tokenize(
                entry.belief.claim + " " + entry.task_origin
            )
            sim = _jaccard_similarity(query_tokens, entry_tokens)
            if sim >= min_similarity:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Increment usage counts for retrieved entries
        results = []
        for _, entry in scored[:top_k]:
            entry.usage_count += 1
            results.append(entry)

        if results:
            self._save()

        return results

    def build_context(self, task: str, top_k: int = 5) -> str:
        """Build a context string from relevant knowledge for a task."""
        entries = self.search(task, top_k=top_k)
        if not entries:
            return ""

        lines = ["KNOWLEDGE BASE (confirmed beliefs from previous sessions):"]
        for entry in entries:
            method = entry.verification_method.value
            lines.append(
                f"  - [{entry.belief.confidence.value}] {entry.belief.claim}"
                f" (from: {entry.task_origin[:60]}, verified: {method})"
            )
        return "\n".join(lines)

    @property
    def entries(self) -> list[KnowledgeEntry]:
        return list(self._entries)
