"""Prediction market — track provider accuracy and compute Brier-weighted trust."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from epistemic_agents.rag import _tokenize, _jaccard_similarity
from epistemic_agents.schema import ProviderPosition


class Prediction(BaseModel):
    """A provider's prediction on a claim."""

    provider_name: str
    claim: str
    confidence_score: float = Field(description="0.0-1.0 confidence in claim being true")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Resolution(BaseModel):
    """How a claim was resolved."""

    claim: str
    outcome: bool = Field(description="True if the claim turned out to be correct")
    method: str = Field(default="manual", description="How resolution was determined")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProviderTrackRecord(BaseModel):
    """Aggregated prediction accuracy for a provider."""

    provider_name: str
    total_predictions: int = 0
    brier_score_sum: float = 0.0
    domain_scores: dict[str, DomainRecord] = Field(default_factory=dict)

    @property
    def brier_score(self) -> float:
        """Average Brier score (lower = better calibrated). Range [0, 1]."""
        if self.total_predictions == 0:
            return 0.5  # Uninformative prior
        return self.brier_score_sum / self.total_predictions

    @property
    def weight(self) -> float:
        """Trust weight derived from Brier score. Range [0.1, 2.0].

        Perfect calibration (brier=0) → weight=2.0
        Random guessing (brier=0.25) → weight=1.5
        Always wrong (brier=1.0) → weight=0.1 (floor)
        """
        return max(0.1, 2.0 - 2.0 * self.brier_score)


class DomainRecord(BaseModel):
    """Per-domain calibration record for a provider."""
    domain: str
    brier_score_sum: float = 0.5  # Bayesian prior: 2 predictions at 0.25 Brier
    count: int = 2  # Prior count

    @property
    def brier_score(self) -> float:
        if self.count == 0:
            return 0.25
        return self.brier_score_sum / self.count

    @property
    def weight(self) -> float:
        return max(0.1, 2.0 - 2.0 * self.brier_score)


class PairwiseTracker(BaseModel):
    """Track pairwise agreement rates between providers across debate rounds."""

    # Maps "provA|provB" (sorted) -> list of agreement booleans per round
    _pair_history: dict[str, list[bool]] = {}

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        self._pair_history = data.get("_pair_history", {})

    def record_round(self, positions: list[ProviderPosition]) -> None:
        """Compare beliefs across providers and record pairwise agreement."""
        providers = [(p.provider_name, p.beliefs) for p in positions if p.beliefs]
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                name_a, beliefs_a = providers[i]
                name_b, beliefs_b = providers[j]
                key = "|".join(sorted([name_a, name_b]))
                agreed = self._check_agreement(beliefs_a, beliefs_b)
                self._pair_history.setdefault(key, []).append(agreed)

    @staticmethod
    def _check_agreement(beliefs_a, beliefs_b) -> bool:
        """Two providers agree if any belief pair has claim similarity > 0.4 AND confidence gap < 0.2."""
        for ba in beliefs_a:
            tokens_a = _tokenize(ba.claim)
            for bb in beliefs_b:
                tokens_b = _tokenize(bb.claim)
                if _jaccard_similarity(tokens_a, tokens_b) > 0.4:
                    gap = abs(ba.effective_score - bb.effective_score)
                    if gap < 0.2:
                        return True
        return False

    def pairwise_rates(self) -> dict[tuple[str, str], float]:
        """Rolling agreement rate per provider pair."""
        rates: dict[tuple[str, str], float] = {}
        for key, history in self._pair_history.items():
            if not history:
                continue
            parts = key.split("|")
            rate = sum(history) / len(history)
            rates[(parts[0], parts[1])] = rate
        return rates

    def avg_correlation(self) -> float:
        """Mean of all pairwise agreement rates."""
        rates = self.pairwise_rates()
        if not rates:
            return 0.0
        return sum(rates.values()) / len(rates)

    def n_eff(self, n_providers: int) -> float:
        """Effective independent panel size: N / (1 + (N-1) * avg_corr)."""
        if n_providers <= 1:
            return float(n_providers)
        corr = max(0.0, self.avg_correlation())
        return n_providers / (1.0 + (n_providers - 1) * corr)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {"pair_history": self._pair_history}

    @classmethod
    def from_dict(cls, data: dict) -> "PairwiseTracker":
        """Deserialize from JSON storage."""
        tracker = cls()
        tracker._pair_history = data.get("pair_history", {})
        return tracker


class PredictionMarket:
    """Track and score provider predictions for calibration-weighted trust."""

    def __init__(self, path: str | Path = ".epistemic_predictions.json"):
        self._path = Path(path)
        self._predictions: list[Prediction] = []
        self._resolutions: list[Resolution] = []
        self._records: dict[str, ProviderTrackRecord] = {}
        self.pairwise_tracker: PairwiseTracker = PairwiseTracker()
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._predictions = [Prediction.model_validate(p) for p in data.get("predictions", [])]
        self._resolutions = [Resolution.model_validate(r) for r in data.get("resolutions", [])]
        self._records = {
            name: ProviderTrackRecord.model_validate(rec)
            for name, rec in data.get("records", {}).items()
        }
        pw_data = data.get("pairwise_tracker")
        if pw_data:
            self.pairwise_tracker = PairwiseTracker.from_dict(pw_data)

    def _save(self) -> None:
        data = {
            "predictions": [p.model_dump(mode="json") for p in self._predictions],
            "resolutions": [r.model_dump(mode="json") for r in self._resolutions],
            "records": {
                name: rec.model_dump(mode="json")
                for name, rec in self._records.items()
            },
            "pairwise_tracker": self.pairwise_tracker.to_dict(),
        }
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def place_prediction(self, prediction: Prediction) -> None:
        """Record a provider's prediction."""
        self._predictions.append(prediction)
        self._save()

    def resolve(self, resolution: Resolution) -> list[str]:
        """Resolve a claim and update all matching providers' Brier scores.

        Returns list of provider names whose scores were updated.
        """
        self._resolutions.append(resolution)
        updated: list[str] = []

        for pred in self._predictions:
            if pred.claim == resolution.claim:
                # Brier score: (forecast - outcome)^2
                outcome_val = 1.0 if resolution.outcome else 0.0
                brier = (pred.confidence_score - outcome_val) ** 2

                if pred.provider_name not in self._records:
                    self._records[pred.provider_name] = ProviderTrackRecord(
                        provider_name=pred.provider_name,
                    )
                record = self._records[pred.provider_name]
                record.total_predictions += 1
                record.brier_score_sum += brier
                updated.append(pred.provider_name)

        self._save()
        return updated

    def get_weight(self, provider_name: str, domain: str | None = None) -> float:
        """Get a provider's trust weight based on prediction track record."""
        record = self._records.get(provider_name)
        if record is None:
            return 1.0  # No track record → neutral weight
        if domain and domain in record.domain_scores:
            return record.domain_scores[domain].weight
        return record.weight

    @property
    def records(self) -> dict[str, ProviderTrackRecord]:
        return dict(self._records)
