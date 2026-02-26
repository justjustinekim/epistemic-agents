"""Demo: Custom task with epistemic feedback loop."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epistemic_agents.executor import Executor
from epistemic_agents.loop import EpistemicLoop
from epistemic_agents.thinker import Thinker

TASK = """\
Design a go-to-market strategy for a new AI-powered code review tool \
called "ReviewBot" that automatically reviews pull requests, suggests \
improvements, and catches bugs before human reviewers see them. \
The company has $2M in seed funding and needs to reach $1M ARR within \
18 months to raise a Series A. The founding team is 4 engineers with \
strong ML backgrounds but no prior startup experience.
"""

# Ground truth the executor knows that the thinker doesn't.
# Designed to challenge common GTM assumptions for dev tools.
EXECUTOR_CONTEXT = """\
Critical context the strategic thinker did not have access to:

1. The market just shifted dramatically: GitHub Copilot launched a built-in \
   code review feature 3 months ago, included free for all GitHub Enterprise \
   customers. Google also integrated AI review into their internal tools and \
   is rumored to be bringing it to Cloud Workstations. The "AI code review" \
   category is being commoditized by platform incumbents.

2. The team's only distribution channel is a personal blog with 2,000 \
   subscribers and a Twitter/X account with 800 followers. They have no \
   sales team, no developer relations person, no partnerships, and no \
   enterprise sales experience. Cold outbound to engineering leaders has a \
   0.5% response rate in their initial tests.

3. Early beta feedback (47 users over 2 months) reveals a surprise: users \
   don't value the bug-catching or code improvement suggestions (they say \
   Copilot already does this "well enough"). What users love is ReviewBot's \
   ability to enforce team-specific coding standards and architectural \
   patterns — something generic AI tools can't do because it requires \
   codebase-specific context. The top feature request is "teach ReviewBot \
   our team's conventions."

4. The $2M seed gives them roughly 14 months of runway at current burn rate \
   ($140K/month for 4 engineers + infrastructure). Marketing budget is \
   effectively $0 — all spend is engineering.

5. Enterprise sales cycles in their initial conversations average 4-6 months \
   from first contact to signed contract. At that pace, even if they close \
   every deal, they can't reach $1M ARR through enterprise sales alone within \
   18 months — the math doesn't work with a 4-person team and zero sales \
   infrastructure.
"""


def main() -> None:
    thinker = Thinker()
    executor = Executor(additional_context=EXECUTOR_CONTEXT)
    loop = EpistemicLoop(thinker=thinker, executor=executor, max_rounds=3)

    log = loop.run(TASK)

    print(f"\n{'='*60}")
    print(f"Completed in {log.round_trips} round(s). Converged: {log.converged}")
    print(f"Total entries: {len(log.entries)}")

    log_path = os.path.join(os.path.dirname(__file__), "custom_task_log.json")
    with open(log_path, "w") as f:
        f.write(log.model_dump_json(indent=2))
    print(f"Full log saved to: {log_path}")


if __name__ == "__main__":
    main()
