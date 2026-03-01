# Epistemic Agents

Structured epistemological communication between AI models. A multi-model panel debates and synthesizes positions, while a thinker-executor loop iteratively refines strategies through belief revision. Every claim carries explicit confidence levels, falsification conditions, assumptions, and evidence grounding — forcing models to be epistemically honest.

Features include belief dependency graphs with cycle detection, numeric confidence aggregation via log-odds averaging, adversarial attack graphs, prediction markets for provider trust, cross-session knowledge bases, and calibration games with known ground truth.

## Why

Existing multi-agent frameworks handle orchestration plumbing (routing, tool use, message passing) but treat the communication between models as an afterthought — usually raw text or loose JSON. Nobody formalizes the **epistemological layer**: what does one model *believe*, how confident is it, what would prove it wrong, and how should beliefs update when new evidence arrives?

This project explores that gap. The core idea: if you force a thinker to state falsification conditions upfront, and force an executor to check those conditions against ground truth, you get meaningfully deeper analysis than naive handoff. And if you then throw 7+ models into an adversarial debate, you surface blind spots and tensions that no single model catches alone.

## How It Works

The system has three analysis tiers, automatically routed by the `Orchestrator`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QUICK TIER                                  │
│  Task → Thinker (Opus) → Verdict                                   │
│  For simple questions. Single call, no loop.                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       STANDARD TIER                                 │
│                                                                     │
│  Task → RAG context injection                                       │
│       → Thinker (Opus) → StrategicHandoff                          │
│       → Executor (Sonnet) → ExecutorFeedback                       │
│       → [if escalation] Thinker revises → ThinkerAmendment         │
│       → [loop until convergence or max rounds]                     │
│       → Verdict                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         DEEP TIER                                   │
│                                                                     │
│  Phase 1: Panel Debate (3 rounds)                                   │
│    → All providers analyze independently (parallel)                 │
│    → 2 rounds of cross-model debate                                 │
│                                                                     │
│  Phase 2: Cross-Model Synthesis                                     │
│    → Claude Opus synthesizes agreements, tensions,                  │
│      blind spots, unique insights                                   │
│                                                                     │
│  Phase 3: Panel Refutation                                          │
│    → Each model challenges the synthesis adversarially              │
│                                                                     │
│  Phase 4: Final Re-Synthesis                                        │
│    → Synthesizer incorporates valid refutations,                    │
│      holds ground on weak ones                                      │
│                                                                     │
│  Phase 5: Verdict Generation                                        │
│    → Cheap model (Haiku) distills into Verdict                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Multi-Model Panel

The deep tier runs a **multi-model debate** across 7+ AI providers. Each model analyzes the task independently, then responds directly to other models' positions across multiple rounds.

### Supported Providers

| Provider | Model | API | Env Var |
|----------|-------|-----|---------|
| Claude (Anthropic) | Opus | CLI (Max plan) | Always available |
| Gemini (Google) | gemini-2.0-flash | REST | `GOOGLE_API_KEY` |
| Grok (xAI) | grok-3 | OpenAI-compatible | `XAI_API_KEY` |
| DeepSeek | deepseek-reasoner | OpenAI-compatible | `DEEPSEEK_API_KEY` |
| QwQ (Alibaba) | qwq-plus | DashScope | `DASHSCOPE_API_KEY` |
| GPT (OpenAI) | gpt-4o-mini | OpenAI API | `OPENAI_API_KEY` |
| Perplexity | sonar-pro | OpenAI-compatible | `PERPLEXITY_API_KEY` |

All external providers use `urllib.request` (stdlib only — zero new dependencies). A single `OpenAICompatProvider` class handles GPT, Grok, DeepSeek, QwQ, and Perplexity via different base URLs.

### Debate → Synthesis → Refutation Pipeline

1. **Independent analysis** — All models analyze the task in parallel, stating beliefs with confidence levels and assumptions.
2. **Cross-model debate** — Each model sees all other models' positions and responds directly. They challenge, build on, defend, and sharpen each other's arguments.
3. **Synthesis** — Claude Opus produces a structured `PanelSynthesis`: agreements, tensions, blind spots, unique insights, and a synthesized strategy.
4. **Refutation** — Every model gets to challenge the synthesis. They flag misrepresentation, false consensus, dismissed insights, and synthesizer bias.
5. **Re-synthesis** — The synthesizer incorporates valid refutations and holds ground on weak ones.

### Virtual Panelists

Four adversarial cognitive roles that wrap any real provider, adding structured epistemic pressure to the debate:

| Role | Function |
|------|----------|
| **Devil's Advocate** | Argues against emerging consensus, finds failure modes, challenges framing assumptions. "What if the opposite is true?" |
| **Steelman** | Constructs the strongest possible version of every argument. Finds hidden compatibility between opposed positions. |
| **Assumption Hunter** | Surfaces hidden assumptions, rates their fragility, traces cascade chains if they fail, suggests how to test them. |
| **Naive Outsider** | Asks the obvious questions experts skip. Challenges jargon, demands plain-language explanations. "Your confusion is a feature, not a bug." |

Virtual panelists are created via `create_virtual_panelists(provider)` — the panel treats them as regular providers, requiring zero changes to `ModelPanel`.

### Code Executor

The `CodeExecutorProvider` validates claims empirically by generating and running Python code:

1. Claude extracts testable factual claims from the analysis (math, logic, data relationships, algorithm properties).
2. For each claim, it generates a short Python test script.
3. Each test runs in a **sandboxed subprocess** with API keys stripped, `os.system` disabled, filesystem restricted to `/tmp`, and a 30-second timeout.
4. Results are reported as PASS/FAIL/ERROR with detail.

Claims that are opinions, predictions, or strategic recommendations are classified as untestable and listed separately. Enable with `include_code_executor=True` when configuring providers.

### RAG from Past Sessions

`build_rag_context()` injects relevant context from past sessions into new analyses using zero-dependency Jaccard similarity (no vector DB, no embeddings). Six context sections are included when data exists:

1. **Falsified beliefs** — Past beliefs proven wrong, so they aren't repeated.
2. **Revised beliefs** — Past beliefs that needed correction, to scrutinize similar claims.
3. **Calibration data** — Historical accuracy by confidence level from the `BeliefLedger`, with temporal decay weighting and per-domain filtering.
4. **Feedback patterns** — Which tiers/approaches worked based on user feedback.
5. **Provider track records** — Which providers contribute unique insights vs. echo consensus.
6. **Knowledge base** — Confirmed beliefs from the persistent `KnowledgeBase`, retrieved by similarity.

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

### Provider Configuration

Set API keys for the providers you want to include in the panel. Claude is always available via the CLI — no key needed.

```bash
export GOOGLE_API_KEY=...     # Gemini
export XAI_API_KEY=...        # Grok
export DEEPSEEK_API_KEY=...   # DeepSeek
export DASHSCOPE_API_KEY=...  # QwQ
export OPENAI_API_KEY=...     # GPT
export PERPLEXITY_API_KEY=... # Perplexity
```

### Running Examples

Run the full orchestrator demo (deep tier with all available models):
```bash
python examples/orchestrator_demo.py
```

Run the panel debate demo:
```bash
python examples/panel_demo.py
```

Run the epistemic feedback loop demo (standard tier):
```bash
python examples/research_task.py
```

Run the naive comparison (same task, plain text, no feedback loop):
```bash
python examples/naive_comparison.py
```

Run the virtual panelists demo (4 adversarial roles):
```bash
python examples/virtual_panel_demo.py
```

Run the code executor demo (empirical claim validation):
```bash
python examples/code_executor_demo.py
```

Run the tournament demo (compare analysis tiers):
```bash
python examples/tournament_demo.py
```

Run the knowledge base demo (persistent beliefs):
```bash
python examples/knowledge_base_demo.py
```

Run all demos:
```bash
python examples/run_all.py
```

### Programmatic Usage

```python
from epistemic_agents.config import get_available_providers
from epistemic_agents.panel import ModelPanel
from epistemic_agents.orchestrator import Orchestrator, Tier

# Discover providers from env vars (include_code_executor=True for empirical validation)
providers = get_available_providers(include_code_executor=True)
panel = ModelPanel(providers)
orchestrator = Orchestrator(panel=panel)

# Auto-route based on task complexity
result = orchestrator.auto_route("Should we migrate from REST to GraphQL?")

# Or specify a tier explicitly
result = orchestrator.run("Compare Redis vs Memcached for session storage", tier=Tier.DEEP)

print(result.verdict.recommendation)
```

### Running Tests

```bash
pytest tests/ -v
```

223 tests covering schema validation, BIS cycle detection, ledger calibration, confidence aggregation, belief extraction, agreement detection, position tracking, prediction markets, adversarial graphs, tournaments, knowledge base, and full integration tests for the loop, panel, orchestrator, and synthesizer.

## Project Structure

```
src/epistemic_agents/
├── schema.py              # Epistemic protocol — beliefs, handoffs, synthesis, verdict
├── thinker.py             # Thinker agent (Claude Opus) — deep analysis + revision
├── executor.py            # Executor agent (Claude Sonnet) — execution + challenge
├── loop.py                # Feedback loop controller — orchestration + convergence
├── client.py              # Claude CLI wrapper (uses your Max plan, no API key needed)
├── orchestrator.py        # Tiered routing — quick / standard / deep
├── panel.py               # Multi-model debate — belief extraction + targeted prompts
├── synthesizer.py         # Cross-model synthesis — programmatic detection + source tracing
├── config.py              # Provider discovery from environment variables
├── rag.py                 # RAG — retrieval-augmented context from past sessions
├── bis.py                 # Belief Importance Scoring + cycle detection + cascade falsification
├── ledger.py              # Cross-session belief tracking + calibration + temporal decay
├── tracker.py             # Cost + contribution tracking per provider
├── feedback.py            # Post-session user feedback collection
├── confidence.py          # Numeric confidence aggregation via log-odds averaging
├── belief_extractor.py    # Structured belief extraction from raw text (via Haiku)
├── agreement_detector.py  # Programmatic agreement/tension detection across providers
├── position_tracker.py    # Stance shift tracking across debate rounds
├── context_manager.py     # Token budgeting and round summarization
├── prediction_market.py   # Brier-scored provider trust weighting
├── adversarial_graph.py   # Attack priority graphs for belief testing
├── tournament.py          # Cross-tier comparison (quick vs standard vs deep)
├── knowledge_base.py      # Persistent confirmed beliefs with similarity search
├── calibration_games.py   # Synthetic tasks with known ground truth
└── providers/
    ├── base.py            # BaseProvider ABC
    ├── claude.py          # Claude via CLI
    ├── gemini.py          # Google Gemini via REST
    ├── openai_compat.py   # GPT, Grok, DeepSeek, QwQ, Perplexity
    ├── virtual.py         # Virtual panelists — 4 adversarial cognitive roles
    └── code_executor.py   # Empirical claim validation via Python execution
```

## The Protocol

The core contribution is the structured schema for model-to-model epistemic communication:

**Belief** — A claim with `confidence` (high/moderate/low/speculative), `confidence_score` (0.0–1.0 numeric), `grounding` (empirical/model_consensus/single_model/assumed), `justification`, `falsification_conditions`, `key_assumptions`, and `depends_on` (upstream belief IDs for dependency tracking). The `effective_score` property returns the numeric score if set, otherwise maps from the qualitative level.

**StrategicHandoff** — The thinker's output: `intent`, `beliefs`, `plan_steps`, `decision_boundaries`, `open_questions`, and `meta_reasoning`.

**ExecutorFeedback** — The executor's report: `observations`, `escalations` (with type, severity, detail), `challenged_beliefs` (with evidence), `new_evidence`, and `executor_recommendation`.

**ThinkerAmendment** — The thinker's revision: `amendment_type` (revise/clarify/delegate/abort), `updated_beliefs`, `revised_steps`, and `guidance`.

**ProviderPosition** — A single model's analysis: `provider_name`, `model_id`, `beliefs` (structured, auto-extracted via cheap model), and `raw_analysis`.

**PanelSynthesis** — The synthesizer's cross-model output: `provider_positions`, `agreements`, `tensions`, `blind_spots`, `unique_insights`, `synthesized_strategy`, and `meta_confidence`. Agreements and tensions include `source_refs` for traceability.

**Verdict** — Concise decision-oriented summary: `decision_point`, `recommendation`, `confidence`, `key_risk`, `dissent`, `tier_used`, and `cost_tokens`.

## Advanced Systems

### Belief Dependency Graph

Beliefs reference upstream beliefs via `depends_on`. The BIS module (`bis.py`) computes weighted importance scores and supports:

- **Cycle detection** — DFS coloring (WHITE/GRAY/BLACK) identifies circular dependencies without infinite recursion.
- **Weighted importance** — Each dependent's weight is its `effective_score`, so high-confidence dependents make their parent more important.
- **Testability boost** — Beliefs with more falsification conditions score higher (up to +50%).
- **Cascade falsification** — When a belief is falsified, all downstream beliefs are flagged.

### Numeric Confidence & Log-Odds Aggregation

Beliefs carry both qualitative levels (high/moderate/low/speculative) and optional numeric scores (0.0–1.0). The `confidence.py` module aggregates scores via log-odds averaging, which handles extreme values better than naive averaging.

### Prediction Market

The `PredictionMarket` tracks provider predictions and outcomes using Brier scores. Providers that make well-calibrated predictions earn higher trust weights (`max(0.1, 2.0 - 2.0 * brier_score)`), which can inform future synthesis weighting.

### Adversarial Attack Graph

`build_attack_graph()` prioritizes which beliefs to test first based on: importance (BIS score), vulnerability (`(1 - confidence) * assumption_count`), and testability. High-importance, low-confidence, testable beliefs are attacked first.

### Tournament System

`run_tournament()` runs the same task through all three analysis tiers (quick, standard, deep) and compares whether deeper analysis changes the verdict. The `TournamentLog` tracks whether depth adds value over time.

### Knowledge Base

The `KnowledgeBase` stores confirmed beliefs persistently (`.epistemic_knowledge.json`) with their verification method and task origin. Future sessions retrieve relevant confirmed beliefs via Jaccard similarity search, building institutional memory.

### Calibration Games

`calibration_games.py` provides 10 synthetic tasks with known ground truth (binary search complexity, TCP handshake, CAP theorem nuance, etc.) for testing model accuracy. Each provider's Brier score measures how well-calibrated its confidence levels are.

### Targeted Debate Prompting

In debate rounds 2+, each model receives a targeted context that includes its own previous position and specific counterarguments from other models, rather than the full identical transcript. This produces more focused, productive debate.

### Position Tracking

`track_positions()` detects stance shifts across debate rounds — when a provider strengthens, weakens, or reverses its position on a claim. Shift summaries are injected into the synthesis prompt.

### Context Management

`manage_context()` handles token budgeting for long debates: earlier rounds are summarized when the context would exceed the budget, while the latest round is always kept verbatim.
