"""Run both the epistemic and naive versions and compare."""

import subprocess
import sys
import os


def main() -> None:
    examples_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("RUNNING EPISTEMIC VERSION (structured feedback loop)")
    print("=" * 60)
    subprocess.run(
        [sys.executable, os.path.join(examples_dir, "research_task.py")],
        check=True,
    )

    print("\n" * 3)
    print("=" * 60)
    print("RUNNING NAIVE VERSION (plain text, no feedback loop)")
    print("=" * 60)
    subprocess.run(
        [sys.executable, os.path.join(examples_dir, "naive_comparison.py")],
        check=True,
    )

    print("\n" * 2)
    print("=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(
        "Check the output above to compare:\n"
        "  - Did the epistemic version's thinker revise its strategy after feedback?\n"
        "  - Did the structured protocol surface specific belief challenges?\n"
        "  - Did the naive version's thinker ever learn about its wrong assumptions?\n"
        "\nLog files:\n"
        f"  Epistemic: {os.path.join(examples_dir, 'research_task_log.json')}\n"
        f"  Naive:     {os.path.join(examples_dir, 'naive_comparison_log.txt')}"
    )


if __name__ == "__main__":
    main()
