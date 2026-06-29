import numpy as np

from src.consensus.fpr import fuzzy_preference_relation, reliability_scores


def test_fpr_values_are_probabilities_and_diagonal_is_half():
    reliability = np.array([0.1, 0.5, 0.9])
    matrix = fuzzy_preference_relation(reliability)
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
    assert np.allclose(np.diag(matrix), 0.5)


def test_anomalous_sensor_gets_lower_reliability():
    scores = reliability_scores(np.array([25.0, 25.1, 24.9, 40.0]))
    assert scores[-1] < scores[:3].min()
