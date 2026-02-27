# Epistemic Agents — Session Updates

## Multi-Model Panel Architecture

Transformed the epistemic agents system from a single-family Claude setup into a **6-model, multi-family debate engine** with full adversarial synthesis.

### New Provider Integrations

| Provider | Model | API | Status |
|----------|-------|-----|--------|
| Claude (Anthropic) | Opus | CLI (Max plan) | Always available |
| Gemini (Google) | gemini-2.0-flash | REST via `GOOGLE_API_KEY` | Free tier |
| Grok (xAI) | grok-3 | OpenAI-compatible via `XAI_API_KEY` | Free credits |
| DeepSeek | deepseek-reasoner | OpenAI-compatible via `DEEPSEEK_API_KEY` | ~$0.002/1K tokens |
| QwQ (Alibaba) | qwq-plus | DashScope via `DASHSCOPE_API_KEY` | Free (1M tokens) |
| GPT (OpenAI) | gpt-4o-mini | OpenAI API via `OPENAI_API_KEY` | Free tier |

All external providers use `urllib.request` (stdlib only, zero new dependencies). A single `OpenAICompatProvider` class handles GPT, Grok, DeepSeek, and QwQ via different base URLs.

### 4-Phase Debate Pipeline

```
Phase 1: Multi-Round Debate
  → All models analyze independently (parallel)
  → Models respond to each other's positions (2 debate rounds)

Phase 2: Cross-Model Synthesis
  → Claude Opus synthesizes agreements, tensions, blind spots, unique insights

Phase 3: Panel Refutation
  → Each model challenges the synthesis — adversarial accountability round

Phase 4: Final Re-Synthesis
  → Synthesizer incorporates valid refutations, holds ground on weak ones
```

### Thinker Improvements (from 6-model panel recommendation)

**Tier 1 — Bug Fixes & Core:**
- Fixed state-loss bug in `_apply_amendment()` — beliefs now merge by ID, not replaced wholesale
- Added `meta_reasoning` field — thinker self-reflects on its own flaws
- Added `depends_on` to beliefs — enables dependency graph tracking

**Tier 2 — Structured Escalation:**
- New `Escalation` model with `type`, `severity`, `detail`
- New escalation types: `context_shift`, `resource_opportunity`, `partial_success`, `convergence_failure`
- Supports multiple simultaneous escalations from executor

**Tier 3 — Panel-Informed Thinking:**
- Thinker can optionally consult the multi-model panel on uncertain beliefs
- Single-round panel query + synthesis → re-analysis with panel context

**Tier 4 — Belief Ledger:**
- Cross-session belief tracking (`BeliefLedger`)
- Records outcomes: confirmed, falsified, revised, untested
- Calibration injection — thinker sees its own historical accuracy by confidence level

### New Modules (Steps 0–6)

**Step 0: Post-Session Feedback** (`feedback.py`)
- Interactive terminal feedback after each session
- Persists to `.epistemic_feedback.json` for trend tracking

**Step 1: Verdict Schema** (`schema.py`)
- `Verdict` model — concise TL;DR output
- Fields: `decision_point`, `recommendation`, `confidence`, `key_risk`, `dissent`, `tier_used`, `cost_tokens`

**Step 2: BIS + Cascade Falsification** (`bis.py`)
- `importance_scores()` — ranks beliefs by transitive dependent count
- `cascade_falsify()` — when a belief falls, flags all downstream beliefs
- Integrated into thinker's `revise()` with CASCADE ALERT context

**Step 3: Tiered Orchestrator** (`orchestrator.py`)
- `Orchestrator` with `run(task, tier)` and `auto_route(task)`
- **Quick**: Single thinker call → verdict
- **Standard**: Thinker-executor loop → verdict
- **Deep**: Full 4-phase panel debate → thinker loop → verdict
- Heuristic task classification by keywords and length

**Step 4: Verdict Generation** (`orchestrator.py`)
- `generate_verdict()` distills full analysis into a `Verdict` via cheap model (Haiku)
- Automatically runs at the end of every tier

**Step 5: Executor → Panel Escalation** (`loop.py`)
- BLOCKING escalations auto-trigger targeted panel queries
- Panel perspectives get synthesized and injected into the thinker's context

**Step 6: Cost + Contribution Tracking** (`tracker.py`)
- `UsageTracker` — per-session token usage and cost estimation
- `extract_contributions()` — measures each provider's unique insights vs consensus echoing
- `cumulative_summary()` — passive meta-learning across sessions
- Pricing table for all 6 providers including free tiers

### File Structure

```
src/epistemic_agents/
├── __init__.py              # Updated exports
├── bis.py                   # NEW — Belief Importance Scoring
├── client.py                # Claude CLI wrapper
├── config.py                # NEW — Provider discovery from env vars
├── executor.py              # Updated escalation protocol
├── feedback.py              # NEW — Post-session feedback
├── ledger.py                # NEW — Cross-session belief tracking
├── loop.py                  # Updated — panel escalation triggers
├── orchestrator.py          # NEW — Tiered routing (quick/standard/deep)
├── panel.py                 # NEW — Multi-model debate + refutation
├── schema.py                # Updated — Verdict, Escalation, BIS types
├── synthesizer.py           # NEW — Cross-model synthesis
├── thinker.py               # Updated — panel-informed, BIS-aware
├── tracker.py               # NEW — Cost/contribution tracking
└── providers/
    ├── __init__.py
    ├── base.py              # NEW — BaseProvider ABC
    ├── claude.py            # NEW — Claude via CLI
    ├── gemini.py            # NEW — Google Gemini via REST
    └── openai_compat.py     # NEW — GPT, Grok, DeepSeek, QwQ

examples/
├── panel_demo.py            # NEW — Full 4-phase demo with all 6 models

tests/
├── test_bis.py              # NEW — 9 tests
├── test_ledger.py           # 7 tests
├── test_loop.py             # 7 tests
├── test_orchestrator.py     # NEW — 9 tests
├── test_providers.py        # 22 tests
├── test_schema.py           # Updated — 18 tests
└── test_tracker.py          # NEW — 8 tests
```

**Total: 81 tests, all passing.**

### Cost

Estimated cost per full 6-model panel run: ~$0.13 (mostly covered by free tiers). Gemini and QwQ are free. DeepSeek is fractions of a cent. Claude uses the Max plan (no per-token cost).
