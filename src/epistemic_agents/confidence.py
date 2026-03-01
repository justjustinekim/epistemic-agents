"""Numeric confidence aggregation via log-odds averaging."""

from __future__ import annotations

import math

from epistemic_agents.schema import Belief


def _to_log_odds(p: float) -> float:
    """Convert probability to log-odds, clamping to [0.01, 0.99]."""
    p = max(0.01, min(0.99, p))
    return math.log(p / (1 - p))


def _from_log_odds(lo: float) -> float:
    """Convert log-odds back to probability."""
    return 1.0 / (1.0 + math.exp(-lo))


def aggregate_confidence(scores: list[float]) -> float:
    """Aggregate multiple confidence scores via log-odds averaging.

    Log-odds averaging is better than naive averaging because it correctly
    handles extreme values — averaging 0.95 and 0.95 in log-odds gives 0.95,
    not 0.95 (same as naive), but averaging 0.95 and 0.05 gives 0.50, not 0.50
    (also same). The difference shows with asymmetric distributions: log-odds
    respects the information content of extreme confidence.

    Returns 0.5 if the input list is empty.
    """
    if not scores:
        return 0.5
    total = sum(_to_log_odds(s) for s in scores)
    avg_lo = total / len(scores)
    return _from_log_odds(avg_lo)


def aggregate_beliefs_confidence(beliefs: list[Belief]) -> float:
    """Aggregate confidence from a list of Belief objects using effective_score."""
    if not beliefs:
        return 0.5
    scores = [b.effective_score for b in beliefs]
    return aggregate_confidence(scores)
