# Epistemic Agents

Structured epistemological communication between AI models. A thinker model does deep analysis and produces strategic handoffs with explicit beliefs, confidence levels, falsification conditions, and assumptions. An executor model acts on the strategy, challenges flawed premises with evidence, and triggers belief revision through a structured feedback loop.

## Why

Existing multi-agent frameworks handle orchestration plumbing (routing, tool use, message passing) but treat the communication between models as an afterthought — usually raw text or loose JSON. Nobody formalizes the **epistemological layer**: what does one model *believe*, how confident is it, what would prove it wrong, and how should beliefs update when new evidence arrives?

This project explores that gap. The core idea: if you force a thinker to state falsification conditions upfront, and force an executor to check those conditions against ground truth, you get meaningfully deeper analysis than naive handoff.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                     THINKER (Opus)                   │
│                                                     │
│  Analyzes task → produces StrategicHandoff:         │
│  • Intent (commander's intent — the "why")          │
│  • Beliefs (claims + confidence + justification     │
│    + falsification conditions + assumptions)        │
│  • Plan steps                                       │
│  • Decision boundaries (pre-committed responses)    │
│  • Open questions (explicit uncertainty)            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    EXECUTOR (Sonnet)                  │
│                                                     │
│  Acts on strategy → produces ExecutorFeedback:      │
│  • Observations                                     │
│  • Challenged beliefs (with evidence)               │
│  • Escalation type (contradiction / ambiguity /     │
│    discovery / assumption violation)                │
│  • Recommendation                                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼ (if escalation needed)
┌─────────────────────────────────────────────────────┐
│                  THINKER REVISES                      │
│                                                     │
│  Reviews feedback → produces ThinkerAmendment:      │
│  • Amendment type (revise / clarify / delegate /    │
│    abort)                                           │
│  • Updated beliefs incorporating new evidence       │
│  • Revised plan steps                               │
│  • Guidance for the executor                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼ (loop continues until
                          convergence or max rounds)
```

## Example: What the Feedback Loop Produces

Given a task like *"Design a caching layer for a REST API at 10K RPS"*, the system:

**Round 1 — Thinker** designs a multi-tier cache (L1 in-process + L2 Redis) with 6 beliefs, each with falsification conditions like *"If the application runs on serverless infrastructure, L1 provides no value."*

**Round 1 — Executor** (with ground truth context) challenges 5 of 6 beliefs:
- *"The 200ms latency is API Gateway hops, not DB queries — the DB responds in 15ms"*
- *"This is AWS Lambda — L1 in-process caching is architecturally impossible"*
- *"Prices change every 30 seconds during flash sales — your 5-minute TTL serves stale data"*

**Round 2 — Thinker revises fundamentally**: drops L1, redesigns around Redis-only with volatility-segmented caching (stable metadata vs. volatile pricing vs. real-time inventory), identifies API Gateway topology as the actual bottleneck.

**Round 3** — Executor catches second-order issues: *"A new Lambda subscriber IS new infrastructure under a strict policy"* and *"Inventory isn't occasionally stale — it's continuously, structurally stale during flash sales."* Thinker revises again.

Compare this to naive handoff (included in `examples/naive_comparison.py`), where the executor identifies the same flaws but the thinker never sees the critique and the final output is still the wrong strategy.

## Setup

Requires [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) authenticated with your account (Max/Pro plan or API key).

```bash
cd epistemic-agents
uv venv && uv pip install -e ".[dev]"
source .venv/bin/activate
```

## Usage

Run the epistemic feedback loop demo:
```bash
python examples/research_task.py
```

Run the naive comparison (same task, plain text, no feedback loop):
```bash
python examples/naive_comparison.py
```

Run both and compare:
```bash
python examples/run_all.py
```

Run tests:
```bash
pytest tests/ -v
```

## Project Structure

```
src/epistemic_agents/
├── schema.py      # The epistemic protocol — Pydantic models for beliefs,
│                  # handoffs, feedback, and amendments
├── thinker.py     # Thinker agent (Claude Opus) — deep analysis + revision
├── executor.py    # Executor agent (Claude Sonnet) — execution + challenge
├── loop.py        # Feedback loop controller — orchestration + convergence
└── client.py      # Claude CLI wrapper (uses your Max plan, no API key needed)
```

## The Protocol

The core contribution is the structured schema for model-to-model epistemic communication:

**Belief** — A claim with `confidence` (high/moderate/low/speculative), `justification`, `falsification_conditions`, and `key_assumptions`.

**StrategicHandoff** — The thinker's output: `intent`, `beliefs`, `plan_steps`, `decision_boundaries`, and `open_questions`.

**ExecutorFeedback** — The executor's report: `observations`, `escalation_type`, `challenged_beliefs` (with evidence), `new_evidence`, and `executor_recommendation`.

**ThinkerAmendment** — The thinker's revision: `amendment_type` (revise/clarify/delegate/abort), `updated_beliefs`, `revised_steps`, and `guidance`.