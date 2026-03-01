"""Tests for persistent knowledge base."""

import tempfile
from pathlib import Path

from epistemic_agents.knowledge_base import KnowledgeBase, KnowledgeEntry
from epistemic_agents.schema import Belief, ConfidenceLevel, VerificationMethod


def _make_belief(id: str, claim: str) -> Belief:
    return Belief(
        id=id,
        claim=claim,
        confidence=ConfidenceLevel.HIGH,
        justification="Test",
    )


def test_store_and_retrieve():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")

        belief = _make_belief("b1", "Redis improves API latency")
        kb.store(belief, task_origin="Optimize API performance")

        results = kb.search("API latency Redis")
        assert len(results) == 1
        assert results[0].belief.id == "b1"


def test_search_relevance():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")

        kb.store(
            _make_belief("b1", "Redis improves API latency"),
            task_origin="API optimization",
        )
        kb.store(
            _make_belief("b2", "React hooks simplify component state"),
            task_origin="Frontend refactoring",
        )

        results = kb.search("API latency optimization")
        assert len(results) >= 1
        assert results[0].belief.id == "b1"  # More relevant


def test_search_no_match():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        kb.store(
            _make_belief("b1", "Redis caching"),
            task_origin="Backend work",
        )
        results = kb.search("quantum computing physics")
        assert len(results) == 0


def test_build_context():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        kb.store(
            _make_belief("b1", "Redis improves latency"),
            task_origin="API optimization",
            verification_method=VerificationMethod.CODE_EXECUTION,
        )
        context = kb.build_context("Optimize API response times")
        assert "KNOWLEDGE BASE" in context
        assert "Redis" in context
        assert "code_execution" in context


def test_build_context_empty():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        assert kb.build_context("anything") == ""


def test_usage_count_incremented():
    with tempfile.TemporaryDirectory() as d:
        kb = KnowledgeBase(path=Path(d) / "kb.json")
        kb.store(
            _make_belief("b1", "Redis caching improves performance"),
            task_origin="API work",
        )
        kb.search("Redis caching performance")
        kb.search("Redis caching performance")
        assert kb.entries[0].usage_count == 2


def test_persistence():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "kb.json"

        kb1 = KnowledgeBase(path=path)
        kb1.store(_make_belief("b1", "Test claim"), task_origin="Test")

        kb2 = KnowledgeBase(path=path)
        assert len(kb2.entries) == 1
        assert kb2.entries[0].belief.id == "b1"
