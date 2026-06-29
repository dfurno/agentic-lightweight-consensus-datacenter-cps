from __future__ import annotations

import numpy as np


def robust_center_scale(values: np.ndarray, epsilon: float = 1e-9) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    return center, 1.4826 * mad + epsilon


def reliability_scores(values: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    center, scale = robust_center_scale(clean, epsilon)
    return np.exp(-np.abs(clean - center) / scale)


def fuzzy_preference_relation(reliability: np.ndarray, kappa: float = 10.0) -> np.ndarray:
    scores = np.asarray(reliability, dtype=float)
    delta = scores[:, None] - scores[None, :]
    matrix = 1.0 / (1.0 + np.exp(-kappa * delta))
    np.fill_diagonal(matrix, 0.5)
    return matrix


def dominance_scores(fpr_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(fpr_matrix, dtype=float)
    n = matrix.shape[0]
    if n == 1:
        return np.array([1.0])
    return (matrix.sum(axis=1) - np.diag(matrix)) / (n - 1)
