"""Tests for epistemic memory facade (Change 5A)."""

import tempfile
from pathlib import Path

from epistemic_agents.memory import EpistemicMemory, MemoryResult
from epistemic_agents.knowledge_base import KnowledgeBase
from epistemic_agents.ledger import BeliefLedger, BeliefOutcome, BeliefRecord
from epistemic_agents.schema import Belief, ConfidenceLevel, VerificationMethod


def test_memory_empty_stores():
    mem = EpistemicMemory()
    results = mem.recall("anything")
    assert results == []


def test_memory_recall_from_kb():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        kb.store(
            belief=Belief(id="b1", claim="Redis caching improves latency", confidence=ConfidenceLevel.HIGH, justification="test"),
            task_origin="caching strategy evaluation",
        )
        mem = EpistemicMemory(knowledge_base=kb)
        results = mem.recall("Redis caching performance")
        assert len(results) >= 1
        assert results[0].source == "knowledge_base"


def test_memory_recall_from_ledger():
    with tempfile.TemporaryDirectory() as d:
        ledger = BeliefLedger(path=Path(d) / "ledger.json")
        ledger._records.append(BeliefRecord(
            belief_id="b1",
            claim="Docker containers reduce deployment time",
            confidence=ConfidenceLevel.MODERATE,
            outcome=BeliefOutcome.CONFIRMED,
            task_summary="deployment optimization",
        ))
        ledger._save()
        mem = EpistemicMemory(ledger=ledger)
        results = mem.recall("Docker deployment containers")
        assert len(results) >= 1
        assert results[0].source == "ledger"


def test_memory_build_context():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        kb.store(
            belief=Belief(id="b1", claim="Redis caching improves latency", confidence=ConfidenceLevel.HIGH, justification="test"),
            task_origin="caching evaluation",
        )
        mem = EpistemicMemory(knowledge_base=kb)
        ctx = mem.build_context("Redis caching")
        assert "EPISTEMIC MEMORY" in ctx
        assert "knowledge_base" in ctx


def test_memory_build_context_empty():
    mem = EpistemicMemory()
    ctx = mem.build_context("anything")
    assert ctx == ""


def test_memory_results_sorted_by_relevance():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        kb.store(
            belief=Belief(id="b1", claim="Redis caching improves latency significantly", confidence=ConfidenceLevel.HIGH, justification="test"),
            task_origin="Redis caching evaluation",
        )
        kb.store(
            belief=Belief(id="b2", claim="Python typing helps code quality", confidence=ConfidenceLevel.MODERATE, justification="test"),
            task_origin="code quality review",
        )
        mem = EpistemicMemory(knowledge_base=kb)
        results = mem.recall("Redis caching latency")
        # The Redis entry should be more relevant
        assert len(results) >= 1
        assert "Redis" in results[0].content
