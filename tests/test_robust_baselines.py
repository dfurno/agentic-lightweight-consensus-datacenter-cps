import numpy as np

from src.consensus.baselines import huber_location, run_baseline, tukey_biweight_location


def test_m_estimators_reduce_single_outlier_influence():
    values = np.array([25.0, 25.1, 24.9, 25.2, 40.0])

    assert huber_location(values) < np.mean(values)
    assert tukey_biweight_location(values) < np.mean(values)
    assert abs(run_baseline("huber_location", values) - huber_location(values)) < 1e-9
    assert abs(run_baseline("tukey_biweight", values) - tukey_biweight_location(values)) < 1e-9
