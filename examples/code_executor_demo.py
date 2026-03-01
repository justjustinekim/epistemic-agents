#!/usr/bin/env python3
"""Demonstrate the CodeExecutorProvider validating claims empirically.

Uses the code executor to test specific factual claims by generating
and running Python code, providing empirical grounding for beliefs.
"""

import sys
import time

from epistemic_agents.providers.code_executor import CodeExecutorProvider
from epistemic_agents.panel import ModelPanel, PANEL_SYSTEM_PROMPT


def main() -> None:
    task = (
        "Evaluate the performance characteristics of Python's built-in sort "
        "versus a manual quicksort implementation for lists of 10,000 integers. "
        "Is Python's Timsort consistently faster? How does the gap change with "
        "nearly-sorted input?"
    )

    print(f"Task: {task}\n")

    # Create a code executor provider
    executor = CodeExecutorProvider()
    if not executor.available:
        print("CodeExecutorProvider requires Claude CLI. Skipping.", file=sys.stderr)
        return

    print("Running code executor analysis...\n")
    t0 = time.time()

    result = executor.analyze(task, PANEL_SYSTEM_PROMPT)
    elapsed = time.time() - t0

    print(f"Analysis complete in {elapsed:.1f}s")
    print(f"\nResult:\n{result[:2000]}")

    # Show it can be used in a panel alongside other providers
    print("\n\nTo use in a panel, include with other providers:")
    print("  panel = ModelPanel([claude, gemini, executor], extract_beliefs=True)")
    print("  rounds = panel.debate(task, rounds=2)")


if __name__ == "__main__":
    main()
