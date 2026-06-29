from __future__ import annotations

import numpy as np


def weak_majority_quantifier(r: float, alpha: float = 0.3, beta: float = 0.8) -> float:
    if not 0 <= alpha < beta <= 1:
        raise ValueError("Expected 0 <= alpha < beta <= 1")
    if r < alpha:
        return 0.0
    if r > beta:
        return 1.0
    return float((r - alpha) / (beta - alpha))


def owa_weights(n: int, alpha: float = 0.3, beta: float = 0.8) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    weights = np.array(
        [
            weak_majority_quantifier(i / n, alpha, beta)
            - weak_majority_quantifier((i - 1) / n, alpha, beta)
            for i in range(1, n + 1)
        ],
        dtype=float,
    )
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return weights / total


def reliability_ordered_owa_weights(n: int, alpha: float = 0.3, beta: float = 0.8) -> np.ndarray:
    """OWA weights for values already sorted from most to least reliable."""
    return owa_weights(n, alpha, beta)[::-1]


def direct_owa(values: np.ndarray, alpha: float = 0.3, beta: float = 0.8) -> float:
    clean = np.asarray(values, dtype=float)
    weights = owa_weights(clean.size, alpha, beta)
    ordered = np.sort(clean)
    return float(np.dot(weights, ordered))
