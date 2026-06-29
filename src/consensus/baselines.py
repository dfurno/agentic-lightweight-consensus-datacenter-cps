from __future__ import annotations

import numpy as np

from src.consensus.owa import direct_owa


def mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def median(values: np.ndarray) -> float:
    return float(np.median(values))


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    clean = np.sort(np.asarray(values, dtype=float))
    trim = int(len(clean) * proportion)
    if trim == 0 or 2 * trim >= len(clean):
        return float(np.mean(clean))
    return float(np.mean(clean[trim:-trim]))


def hampel_filter_then_mean(values: np.ndarray, threshold: float = 3.0) -> float:
    clean = np.asarray(values, dtype=float)
    center = np.median(clean)
    mad = np.median(np.abs(clean - center))
    scale = 1.4826 * mad + 1e-9
    kept = clean[np.abs(clean - center) <= threshold * scale]
    if kept.size == 0:
        return float(center)
    return float(np.mean(kept))


def huber_location(values: np.ndarray, c: float = 1.345, max_iter: int = 25, tol: float = 1e-6) -> float:
    clean = np.asarray(values, dtype=float)
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    scale = 1.4826 * mad + 1e-9
    for _ in range(max_iter):
        residual = (clean - center) / scale
        weights = np.minimum(1.0, c / (np.abs(residual) + 1e-12))
        updated = float(np.sum(weights * clean) / (np.sum(weights) + 1e-12))
        if abs(updated - center) <= tol:
            return updated
        center = updated
    return center


def tukey_biweight_location(values: np.ndarray, c: float = 4.685, max_iter: int = 25, tol: float = 1e-6) -> float:
    clean = np.asarray(values, dtype=float)
    center = float(np.median(clean))
    mad = float(np.median(np.abs(clean - center)))
    scale = 1.4826 * mad + 1e-9
    for _ in range(max_iter):
        u = (clean - center) / (c * scale)
        weights = np.square(1.0 - np.square(u))
        weights[np.abs(u) >= 1.0] = 0.0
        if float(np.sum(weights)) <= 1e-12:
            return center
        updated = float(np.sum(weights * clean) / np.sum(weights))
        if abs(updated - center) <= tol:
            return updated
        center = updated
    return center


def run_baseline(name: str, values: np.ndarray, alpha: float = 0.3, beta: float = 0.8) -> float:
    if name == "mean":
        return mean(values)
    if name == "median":
        return median(values)
    if name == "trimmed_mean":
        return trimmed_mean(values)
    if name == "hampel_filter_then_mean":
        return hampel_filter_then_mean(values)
    if name == "direct_owa":
        return direct_owa(values, alpha, beta)
    if name == "huber_location":
        return huber_location(values)
    if name == "tukey_biweight":
        return tukey_biweight_location(values)
    raise ValueError(f"Unknown baseline {name!r}")
