from __future__ import annotations

import numpy as np


def mae(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(truth))))


def rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    diff = np.asarray(predicted) - np.asarray(truth)
    return float(np.sqrt(np.mean(diff * diff)))


def error_summary(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    errors = np.abs(np.asarray(predicted) - np.asarray(truth))
    return {
        "mae": float(np.mean(errors)),
        "rmse": rmse(predicted, truth),
        "median_absolute_error": float(np.median(errors)),
        "p95_absolute_error": float(np.percentile(errors, 95)),
        "max_absolute_error": float(np.max(errors)),
    }


def precision_recall_f1(predicted_flags: np.ndarray, true_flags: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted_flags, dtype=bool)
    truth = np.asarray(true_flags, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
