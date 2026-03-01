"""RAG — retrieval-augmented context from past sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from epistemic_agents.feedback import FeedbackLog, SessionFeedback
    from epistemic_agents.knowledge_base import KnowledgeBase
    from epistemic_agents.ledger import BeliefLedger, BeliefRecord
    from epistemic_agents.tracker import UsageTracker

# Simple stopwords for tokenisation — no external deps
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could must need dare "
    "i me my we our you your he him his she her it its they them their "
    "this that these those what which who whom how when where why "
    "and or but not nor so yet for if then than to of in on at by with "
    "from as into about between through during before after above below "
    "up down out off over under again further once here there all each "
    "every both few more most other some such no any".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on whitespace/punctuation, remove stopwords."""
    words: set[str] = set()
    for raw in text.lower().split():
        cleaned = "".join(c for c in raw if c.isalnum())
        if cleaned and cleaned not in _STOPWORDS and len(cleaned) > 1:
            words.add(cleaned)
    return words


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Word-overlap similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


def _find_relevant_records(
    task_tokens: set[str],
    records: list[BeliefRecord],
    top_k: int = 5,
) -> list[BeliefRecord]:
    """Return the most task-relevant belief records."""
    scored = []
    for r in records:
        tokens = _tokenize(r.claim + " " + r.task_summary)
        sim = _jaccard_similarity(task_tokens, tokens)
        if sim > 0:
            scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


def _find_relevant_feedback(
    task_tokens: set[str],
    entries: list[SessionFeedback],
    top_k: int = 5,
) -> list[SessionFeedback]:
    """Return the most task-relevant feedback entries."""
    scored = []
    for e in entries:
        tokens = _tokenize(e.task_summary + " " + e.comment)
        sim = _jaccard_similarity(task_tokens, tokens)
        if sim > 0:
            scored.append((sim, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def build_rag_context(
    task: str,
    ledger: BeliefLedger | None = None,
    feedback_log: FeedbackLog | None = None,
    tracker: UsageTracker | None = None,
    knowledge_base: KnowledgeBase | None = None,
    max_length: int = 3000,
) -> str:
    """Build retrieval-augmented context from past sessions.

    Sections (included only when data exists):
    1. Falsified beliefs to avoid
    2. Revised beliefs needing scrutiny
    3. Calibration data (from ledger)
    4. Feedback patterns — which tiers/approaches worked
    5. Provider track records — who contributes unique insights
    6. Knowledge base — confirmed beliefs from previous sessions

    Args:
        task: The current task being analysed.
        ledger: Belief outcome ledger.
        feedback_log: User feedback log.
        tracker: Usage and contribution tracker.
        knowledge_base: Persistent knowledge base of confirmed beliefs.
        max_length: Approximate max character length for the context block.

    Returns:
        A formatted context string, or "" if no relevant data.
    """
    task_tokens = _tokenize(task)
    sections: list[str] = []

    # --- 1. Falsified beliefs to avoid ---
    if ledger and ledger.records:
        from epistemic_agents.ledger import BeliefOutcome

        falsified = [r for r in ledger.records if r.outcome == BeliefOutcome.FALSIFIED]
        relevant_falsified = _find_relevant_records(task_tokens, falsified)
        if relevant_falsified:
            lines = ["FALSIFIED BELIEFS (avoid repeating these mistakes):"]
            for r in relevant_falsified:
                lines.append(
                    f"  - [{r.confidence.value}] {r.claim}"
                    f" | Reason: {r.failure_reason or 'unknown'}"
                )
            sections.append("\n".join(lines))

    # --- 2. Revised beliefs needing scrutiny ---
    if ledger and ledger.records:
        from epistemic_agents.ledger import BeliefOutcome

        revised = [r for r in ledger.records if r.outcome == BeliefOutcome.REVISED]
        relevant_revised = _find_relevant_records(task_tokens, revised)
        if relevant_revised:
            lines = ["REVISED BELIEFS (these needed correction — scrutinise similar claims):"]
            for r in relevant_revised:
                lines.append(
                    f"  - [{r.confidence.value}] {r.claim}"
                    f" | {r.failure_reason or 'revised during loop'}"
                )
            sections.append("\n".join(lines))

    # --- 3. Calibration data ---
    if ledger:
        cal = ledger.calibration_context()
        if cal:
            sections.append(cal)

    # --- 4. Feedback patterns ---
    if feedback_log and feedback_log.entries:
        relevant_fb = _find_relevant_feedback(task_tokens, feedback_log.entries)

        # Aggregate tier effectiveness
        tier_stats: dict[str, dict[str, int]] = {}
        for e in feedback_log.entries:
            tier_stats.setdefault(e.tier_used, {"yes": 0, "partially": 0, "no": 0})
            if e.useful in tier_stats[e.tier_used]:
                tier_stats[e.tier_used][e.useful] += 1

        lines = ["FEEDBACK PATTERNS:"]
        for tier, counts in tier_stats.items():
            total = sum(counts.values())
            if total > 0:
                lines.append(
                    f"  {tier}: {counts['yes']}/{total} useful, "
                    f"{counts['partially']}/{total} partial, "
                    f"{counts['no']}/{total} not useful"
                )

        # Include relevant user comments
        comments = [e.comment for e in relevant_fb if e.comment]
        if comments:
            lines.append("  Relevant user suggestions:")
            for c in comments[:3]:
                lines.append(f"    - {c}")

        if len(lines) > 1:  # more than just header
            sections.append("\n".join(lines))

    # --- 5. Provider track records ---
    if tracker and tracker.sessions:
        provider_insights: dict[str, int] = {}
        provider_tensions: dict[str, int] = {}
        for s in tracker.sessions:
            for c in s.provider_contributions:
                provider_insights[c.provider_name] = (
                    provider_insights.get(c.provider_name, 0) + c.unique_insights
                )
                provider_tensions[c.provider_name] = (
                    provider_tensions.get(c.provider_name, 0) + c.tensions_involved
                )

        if provider_insights:
            lines = ["PROVIDER TRACK RECORDS:"]
            for name in sorted(
                provider_insights, key=lambda n: provider_insights[n], reverse=True
            ):
                insights = provider_insights[name]
                tensions = provider_tensions.get(name, 0)
                lines.append(
                    f"  {name}: {insights} unique insights, "
                    f"{tensions} tensions contributed"
                )
            sections.append("\n".join(lines))

    # --- 6. Knowledge base ---
    if knowledge_base:
        kb_context = knowledge_base.build_context(task)
        if kb_context:
            sections.append(kb_context)

    if not sections:
        return ""

    # Truncate to max_length
    result = "\n\n".join(sections)
    if len(result) > max_length:
        result = result[:max_length].rsplit("\n", 1)[0] + "\n  [truncated]"

    return result
