"""Epistemic memory — unified facade over knowledge base, ledger, and prediction market."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from epistemic_agents.rag import _tokenize, _jaccard_similarity

if TYPE_CHECKING:
    from epistemic_agents.knowledge_base import KnowledgeBase, KnowledgeEntry
    from epistemic_agents.ledger import BeliefLedger, BeliefRecord
    from epistemic_agents.prediction_market import PredictionMarket


@dataclass
class MemoryResult:
    """A single result from epistemic memory recall."""
    source: str  # "knowledge_base", "ledger", "prediction_market"
    content: str
    relevance: float
    raw: object = None  # The underlying KnowledgeEntry, BeliefRecord, etc.


class EpistemicMemory:
    """Unified query interface over all epistemic stores."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        ledger: BeliefLedger | None = None,
        prediction_market: PredictionMarket | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._ledger = ledger
        self._market = prediction_market

    def recall(self, query: str, top_k: int = 10) -> list[MemoryResult]:
        """Search all stores and return unified, ranked results."""
        results: list[MemoryResult] = []
        query_tokens = _tokenize(query)

        # Knowledge base
        if self._kb:
            for entry in self._kb.entries:
                tokens = _tokenize(entry.belief.claim + " " + entry.task_origin)
                sim = _jaccard_similarity(query_tokens, tokens)
                if sim > 0.05:
                    results.append(MemoryResult(
                        source="knowledge_base",
                        content=f"[{entry.belief.confidence.value}] {entry.belief.claim}",
                        relevance=sim,
                        raw=entry,
                    ))

        # Ledger
        if self._ledger:
            for record in self._ledger.records:
                tokens = _tokenize(record.claim + " " + record.task_summary)
                sim = _jaccard_similarity(query_tokens, tokens)
                if sim > 0.05:
                    results.append(MemoryResult(
                        source="ledger",
                        content=f"[{record.outcome.value}] {record.claim}",
                        relevance=sim,
                        raw=record,
                    ))

        # Sort by relevance, return top_k
        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:top_k]

    def build_context(self, query: str, top_k: int = 5) -> str:
        """Build a context string from recall results."""
        results = self.recall(query, top_k=top_k)
        if not results:
            return ""
        lines = ["EPISTEMIC MEMORY:"]
        for r in results:
            lines.append(f"  [{r.source}] {r.content}")
        return "\n".join(lines)
