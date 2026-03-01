"""Prediction market — track provider accuracy and compute Brier-weighted trust."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


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


class PredictionMarket:
    """Track and score provider predictions for calibration-weighted trust."""

    def __init__(self, path: str | Path = ".epistemic_predictions.json"):
        self._path = Path(path)
        self._predictions: list[Prediction] = []
        self._resolutions: list[Resolution] = []
        self._records: dict[str, ProviderTrackRecord] = {}
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

    def _save(self) -> None:
        data = {
            "predictions": [p.model_dump(mode="json") for p in self._predictions],
            "resolutions": [r.model_dump(mode="json") for r in self._resolutions],
            "records": {
                name: rec.model_dump(mode="json")
                for name, rec in self._records.items()
            },
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

    def get_weight(self, provider_name: str) -> float:
        """Get a provider's trust weight based on prediction track record."""
        record = self._records.get(provider_name)
        if record is None:
            return 1.0  # No track record → neutral weight
        return record.weight

    @property
    def records(self) -> dict[str, ProviderTrackRecord]:
        return dict(self._records)
