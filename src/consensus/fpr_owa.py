from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.consensus.fpr import dominance_scores, fuzzy_preference_relation, reliability_scores
from src.consensus.owa import reliability_ordered_owa_weights


@dataclass
class ConsensusResult:
    aggregated_value: float
    weights: np.ndarray
    reliability_scores: np.ndarray
    fpr_matrix: np.ndarray
    ordered_sensor_indices: np.ndarray
    anomaly_flags: np.ndarray
    timestamp: str
    diagnostic_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return float(np.mean(self.reliability_scores))


def fpr_owa_consensus(
    values: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.8,
    kappa: float = 10.0,
    timestamp: str | None = None,
) -> ConsensusResult:
    clean = np.asarray(values, dtype=float)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    reliability = reliability_scores(clean)
    matrix = fuzzy_preference_relation(reliability, kappa)
    dominance = dominance_scores(matrix)
    ordered_indices = np.argsort(-dominance)
    ordered_values = clean[ordered_indices]
    weights = reliability_ordered_owa_weights(clean.size, alpha, beta)
    aggregated = float(np.dot(weights, ordered_values))
    anomaly_threshold = max(0.15, float(np.median(reliability) - 2.0 * np.std(reliability)))
    flags = reliability < anomaly_threshold
    return ConsensusResult(
        aggregated_value=aggregated,
        weights=weights,
        reliability_scores=reliability,
        fpr_matrix=matrix,
        ordered_sensor_indices=ordered_indices,
        anomaly_flags=flags,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        diagnostic_metadata={
            "alpha": alpha,
            "beta": beta,
            "kappa": kappa,
            "dominance_scores": dominance.tolist(),
            "anomaly_threshold": anomaly_threshold,
            "confidence": float(np.mean(reliability)),
        },
    )
