#!/usr/bin/env python3
"""Demonstrate the persistent KnowledgeBase storing and retrieving beliefs.

Shows how confirmed beliefs from one session can be stored and retrieved
in future sessions, building institutional memory.
"""

import tempfile
from pathlib import Path

from epistemic_agents.knowledge_base import KnowledgeBase
from epistemic_agents.schema import Belief, ConfidenceLevel, VerificationMethod


def main() -> None:
    # Use a temporary directory for demo (in production, use a persistent path)
    with tempfile.TemporaryDirectory() as d:
        kb_path = Path(d) / "knowledge.json"

        # === Session 1: Store confirmed beliefs ===
        print("=== Session 1: Storing confirmed beliefs ===\n")
        kb = KnowledgeBase(path=kb_path)

        # Store some beliefs from a previous analysis
        beliefs_to_store = [
            (
                Belief(
                    id="perf-1",
                    claim="Redis p95 latency is under 1ms for simple GET operations",
                    confidence=ConfidenceLevel.HIGH,
                    justification="Benchmarked with redis-benchmark on production hardware",
                    falsification_conditions=["If network latency dominates"],
                ),
                "Optimize API response times for product catalog",
                VerificationMethod.CODE_EXECUTION,
            ),
            (
                Belief(
                    id="arch-1",
                    claim="Connection pooling reduces PostgreSQL latency by 40-60%",
                    confidence=ConfidenceLevel.HIGH,
                    justification="Measured before/after with pgbouncer",
                ),
                "Database optimization for user service",
                VerificationMethod.EXECUTOR_CHALLENGE,
            ),
            (
                Belief(
                    id="arch-2",
                    claim="Microservices add 10-50ms per service hop due to serialization",
                    confidence=ConfidenceLevel.MODERATE,
                    justification="Observed in distributed tracing data",
                ),
                "Architecture evaluation for payment system",
                VerificationMethod.MODEL_CONSENSUS,
            ),
        ]

        for belief, task, method in beliefs_to_store:
            entry = kb.store(belief, task_origin=task, verification_method=method)
            print(f"  Stored: [{belief.confidence.value}] {belief.claim}")
            print(f"    From: {task}")
            print(f"    Verified by: {method.value}\n")

        print(f"Knowledge base now has {len(kb.entries)} entries.\n")

        # === Session 2: Retrieve relevant knowledge ===
        print("=== Session 2: Retrieving relevant knowledge ===\n")
        kb2 = KnowledgeBase(path=kb_path)

        # Search for relevant knowledge
        query = "How to improve API latency for our product endpoints?"
        print(f"Query: {query}\n")

        results = kb2.search(query, top_k=3)
        print(f"Found {len(results)} relevant entries:\n")
        for entry in results:
            print(f"  [{entry.belief.confidence.value}] {entry.belief.claim}")
            print(f"    From: {entry.task_origin}")
            print(f"    Verified: {entry.verification_method.value}")
            print(f"    Used {entry.usage_count} time(s)\n")

        # Build RAG context
        context = kb2.build_context("Optimize API response times")
        print(f"RAG context for thinker:\n{context}")


if __name__ == "__main__":
    main()
