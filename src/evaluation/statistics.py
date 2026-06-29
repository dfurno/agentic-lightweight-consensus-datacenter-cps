from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, samples: int = 1000, seed: int = 42) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    if clean.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(clean, size=clean.size, replace=True)) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))
