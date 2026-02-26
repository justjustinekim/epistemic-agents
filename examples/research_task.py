"""Demo: Epistemic feedback loop on a research/design task.

The thinker analyzes a caching strategy task. The executor has additional
"ground truth" context that contradicts some of the thinker's assumptions,
triggering the feedback loop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epistemic_agents.executor import Executor
from epistemic_agents.loop import EpistemicLoop
from epistemic_agents.thinker import Thinker

TASK = """\
Design the best approach to implement a caching layer for a REST API that \
serves product catalog data at approximately 10,000 requests per second. \
The API currently has no caching and response times average 200ms. \
The goal is to reduce p95 latency to under 50ms.
"""

# This is "ground truth" that the executor knows but the thinker does not.
# It's designed to challenge assumptions the thinker is likely to make.
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


def main() -> None:
    thinker = Thinker()
    executor = Executor(additional_context=EXECUTOR_CONTEXT)
    loop = EpistemicLoop(thinker=thinker, executor=executor, max_rounds=3)

    log = loop.run(TASK)

    print(f"\n{'='*60}")
    print(f"Completed in {log.round_trips} round(s). Converged: {log.converged}")
    print(f"Total entries: {len(log.entries)}")

    # Save the full log
    log_path = os.path.join(os.path.dirname(__file__), "research_task_log.json")
    with open(log_path, "w") as f:
        f.write(log.model_dump_json(indent=2))
    print(f"Full log saved to: {log_path}")


if __name__ == "__main__":
    main()
