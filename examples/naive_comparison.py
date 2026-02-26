"""Demo: Same task as research_task.py but with naive text handoff (no protocol).

Sends the task to a thinker model as plain text, then passes the plain text
response to an executor model. No structured schema, no feedback loop.
Compare the output quality against the epistemic version.
"""

import os
import sys

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epistemic_agents.client import plain_request

console = Console()

TASK = """\
Design the best approach to implement a caching layer for a REST API that \
serves product catalog data at approximately 10,000 requests per second. \
The API currently has no caching and response times average 200ms. \
The goal is to reduce p95 latency to under 50ms.
"""

EXECUTOR_CONTEXT = """\
Important context the strategic thinker did not have access to:

1. The product catalog updates frequently — prices change every 30 seconds for \
   flash sales, and inventory counts update in real-time. A standard TTL-based \
   cache would serve stale data constantly.

2. The infrastructure runs on a serverless platform (AWS Lambda) with no persistent \
   in-memory state between invocations. Redis is available but adds 5ms network latency \
   per hop, and the team has a strict policy against adding new infrastructure dependencies.

3. The existing database is already read-optimized with materialized views. \
   The 200ms latency is primarily from network hops through an API gateway, \
   not from slow queries. The database itself responds in ~15ms.

4. 80% of traffic is for the top 50 products (highly skewed access pattern), \
   but the remaining 20% spans 500,000+ products (long tail).
"""

THINKER_PROMPT = "You are a strategic thinker. Analyze this task deeply and provide your recommended approach."
EXECUTOR_PROMPT = """\
You are an executor. A strategic thinker has provided the following strategy. \
Your job is to evaluate it against the ground truth context you have and provide \
your assessment. Be critical — point out any flaws or incorrect assumptions."""


def main() -> None:
    # Step 1: Thinker produces plain text strategy
    console.print(Panel("NAIVE THINKER: Analyzing task (plain text)...", style="bold blue"))

    thinker_response = plain_request(
        model="opus",
        system=THINKER_PROMPT,
        user_message=TASK,
    )

    console.print(Panel(thinker_response, title="Thinker Strategy", border_style="blue"))

    # Step 2: Executor receives plain text + ground truth, responds with plain text
    console.print(Panel("NAIVE EXECUTOR: Evaluating strategy...", style="bold green"))

    executor_message = (
        f"Strategy from thinker:\n\n{thinker_response}\n\n"
        f"--- GROUND TRUTH CONTEXT ---\n\n{EXECUTOR_CONTEXT}\n\n"
        "Evaluate this strategy against the ground truth. "
        "What's wrong? What would you change?"
    )

    executor_response = plain_request(
        model="sonnet",
        system=EXECUTOR_PROMPT,
        user_message=executor_message,
    )

    console.print(Panel(executor_response, title="Executor Assessment", border_style="green"))

    # Note: No feedback loop — the thinker never sees the executor's critique.
    console.print(
        Panel(
            "DONE — No feedback loop. The thinker never sees the executor's critique.\n"
            "Compare this output with the epistemic version (research_task.py) to see\n"
            "whether structured feedback produces a better final strategy.",
            style="bold yellow",
        )
    )

    # Save outputs
    log_path = os.path.join(os.path.dirname(__file__), "naive_comparison_log.txt")
    with open(log_path, "w") as f:
        f.write("=== THINKER STRATEGY (PLAIN TEXT) ===\n\n")
        f.write(thinker_response)
        f.write("\n\n=== EXECUTOR ASSESSMENT (PLAIN TEXT) ===\n\n")
        f.write(executor_response)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
