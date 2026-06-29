import numpy as np

from src.evaluation.metrics import error_summary, mae, precision_recall_f1, rmse


def test_metric_functions_on_toy_examples():
    truth = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 3.0, 5.0])
    assert mae(pred, truth) == 1.0
    assert np.isclose(rmse(pred, truth), np.sqrt(5 / 3))
    summary = error_summary(pred, truth)
    assert summary["max_absolute_error"] == 2.0


def test_precision_recall_f1():
    values = precision_recall_f1(np.array([True, False, True]), np.array([True, True, False]))
    assert values["precision"] == 0.5
    assert values["recall"] == 0.5
    assert values["f1"] == 0.5
