# Epistemic Agents — Session Updates

## Epistemic Feature Expansion

Major feature expansion across 10 work packages, adding 10 new modules, 142 new tests (223 total), and 4 new examples. All changes are backward-compatible — existing code using default fields continues to work unchanged.

### WP1: Schema Foundation

- **BeliefGrounding enum** — `EMPIRICAL`, `MODEL_CONSENSUS`, `SINGLE_MODEL`, `ASSUMED`. Every belief now declares how it was established.
- **VerificationMethod enum** — `EXECUTOR_CHALLENGE`, `CODE_EXECUTION`, `MODEL_CONSENSUS`, `UNVERIFIED`. Tracks how beliefs are verified.
- **Numeric confidence** — `confidence_score: float | None` (0.0–1.0) alongside qualitative levels. `effective_score` property returns numeric if set, otherwise maps from qualitative via `CONFIDENCE_LEVEL_TO_SCORE`.
- **Source traceability** — `source_refs: list[str]` added to `AgreementPoint`, `TensionPoint`, `BlindSpot`, `UniqueInsight`. `combined_confidence_score` on `AgreementPoint`.

### WP2: BIS Fixes

- **Cycle detection** — `detect_cycles()` uses DFS WHITE/GRAY/BLACK coloring to identify circular belief dependencies.
- **Cycle-safe traversal** — `_count_transitive` rewritten as iterative DFS with `visited={bid}` to prevent infinite recursion.
- **Weighted importance** — Dependents weighted by `effective_score`, confidence multiplier is `effective_score + 0.5` (range 0.7–1.4), testability boost up to +50%.

### WP3: Calibration Fixes

- **Convergence bug fix** — Converged beliefs now marked `UNTESTED` (not `CONFIRMED`). Only explicit verification upgrades to `CONFIRMED`.
- **`mark_verified()`** — Explicit verification method for upgrading beliefs from `UNTESTED` to `CONFIRMED`/`FALSIFIED`.
- **Domain classification** — `classify_domain()` categorizes tasks into infrastructure, database, api, frontend, ml, architecture, security, general.
- **Temporal decay** — `calibration_report()` weights outcomes by `2^(-age_days / half_life)` (default 30 days). Per-domain filtering.

### WP4: Numeric Confidence (`confidence.py`)

- Log-odds averaging for confidence aggregation — handles extreme values better than naive averaging.
- `aggregate_confidence(scores)` and `aggregate_beliefs_confidence(beliefs)`.

### WP5: Belief Extraction (`belief_extractor.py`)

- `extract_beliefs()` uses a cheap model (Haiku) to extract 3–8 structured `Belief` objects from raw analysis text.
- Auto-sets `grounding=SINGLE_MODEL`, prefixes belief IDs with provider name.
- Integrated into `ModelPanel._query_provider()` — `ProviderPosition.beliefs` is now populated (not empty `[]`).

### WP6: Programmatic Detection (`agreement_detector.py`)

- `detect_agreements()` — Clusters beliefs by Jaccard similarity, creates `AgreementPoint` with `combined_confidence_score` (log-odds averaged) and `source_refs`.
- `detect_tensions()` — Finds similar claims with divergent confidence (gap > 0.3), creates `TensionPoint` with `source_refs`.
- Integrated into `Synthesizer._run_synthesis()` — programmatic analysis injected as context before LLM synthesis.

### WP7: Panel Improvements

- **Targeted debate prompts** — In rounds 2+, each model gets its own previous position + specific counterarguments from other models, rather than the full identical transcript.
- **Position tracking** (`position_tracker.py`) — `track_positions()` detects stance shifts (strengthened/weakened/reversed) across debate rounds. Summaries injected into synthesis prompt.
- **Context management** (`context_manager.py`) — `manage_context()` handles token budgeting for long debates. Earlier rounds summarized when context exceeds budget; latest round always verbatim.

### WP8: Advanced Systems

- **Prediction market** (`prediction_market.py`) — Brier-scored provider trust. `PredictionMarket.place_prediction()`, `resolve()`, `get_weight()`. Persists to `.epistemic_predictions.json`.
- **Adversarial graph** (`adversarial_graph.py`) — `build_attack_graph()` prioritizes belief testing by `importance * vulnerability * testability`.
- **Tournament** (`tournament.py`) — `run_tournament()` runs all 3 tiers on same task, compares verdicts. `TournamentLog` tracks value-of-depth over time.
- **Knowledge base** (`knowledge_base.py`) — Persistent confirmed beliefs with Jaccard similarity search. `build_context()` for RAG injection.
- **RAG integration** — `build_rag_context()` now accepts `knowledge_base` param (section 6).

### WP9: Integration Tests

- `tests/conftest.py` — `FakeProvider`, `make_belief`, `make_handoff`, `mock_structured_request` fixtures.
- `test_loop_integration.py` — Convergence, max-rounds, abort, panel escalation paths.
- `test_panel_integration.py` — Multi-round debate, refutation, belief extraction, targeted prompting.
- `test_orchestrator_integration.py` — Quick/standard/deep tier end-to-end, auto_route dispatching.
- `test_synthesizer.py` — Single-round, multi-round, re-synthesis, programmatic detection injection.

### WP10: Examples & Calibration Games

- `examples/virtual_panel_demo.py` — Virtual panelists wrapping a provider with all 4 adversarial roles.
- `examples/code_executor_demo.py` — CodeExecutorProvider validating claims empirically.
- `examples/tournament_demo.py` — Cross-tier comparison on the same task.
- `examples/knowledge_base_demo.py` — Storing and retrieving confirmed beliefs across sessions.
- `calibration_games.py` — 10 synthetic tasks with known ground truth, Brier score evaluation.

### Test Summary

**223 tests, all passing.** (was 81)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_schema.py | 29 | Schema fields, grounding, effective_score, source_refs |
| test_bis.py | 16 | Cycle detection, weighted scoring, testability boost |
| test_ledger.py | 17 | UNTESTED fix, mark_verified, domain, temporal decay |
| test_confidence.py | 11 | Log-odds aggregation, extremes, empty input |
| test_belief_extractor.py | 5 | Extraction, grounding assignment, ID prefixing |
| test_agreement_detector.py | 8 | Clustering, tension detection, edge cases |
| test_position_tracker.py | 7 | Shift detection, formatting, thresholds |
| test_prediction_market.py | 7 | Brier scoring, resolution, persistence |
| test_adversarial_graph.py | 5 | Priority ranking, vulnerability, testability |
| test_tournament.py | 4 | Result construction, logging, value-of-depth |
| test_knowledge_base.py | 7 | Store, search, persistence, build_context |
| test_loop_integration.py | 4 | Convergence, max-rounds, abort, panel escalation |
| test_panel_integration.py | 5 | Debate flow, refutation, belief extraction |
| test_orchestrator_integration.py | 6 | All tiers end-to-end, auto_route |
| test_synthesizer.py | 5 | Synthesis, re-synthesis, programmatic detection |
| test_providers.py | 22 | Provider initialization, analyze, availability |
| test_tracker.py | 8 | Usage recording, persistence, contributions |
| test_loop.py | 7 | Loop mechanics, convergence, escalation |
| test_orchestrator.py | 9 | Tier routing, classification, verdict |
| test_virtual.py | 9 | Virtual panelists, role prompts, panel integration |
| test_feedback.py | 7 | Feedback collection, persistence, trend tracking |

---

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
