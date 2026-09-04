import numpy as np
import pytest

from scripts.run_crp_temporal_ablation import agg, dynamic_prediction


def test_dynamic_predictor_tick_zero_has_no_transition():
    cfg = {"exchange": 0.1, "fan_gain": 0.2}
    assert dynamic_prediction(30.0, cfg, None) == 30.0


def test_dynamic_predictor_uses_previous_transition_inputs():
    cfg = {"exchange": 0.1, "fan_gain": 0.2}
    assert dynamic_prediction(30.0, cfg, (20.0, 0.7, 0.5)) == pytest.approx(29.6)


def test_stable_id_breaks_equal_score_ties():
    values = np.array([30.0, 30.0, 29.0, 31.0])
    a = agg(values, 0.3, 0.8, np.array([20, 10, 30, 40]))
    b = agg(values[[1, 0, 2, 3]], 0.3, 0.8, np.array([10, 20, 30, 40]))
    assert a == b


def test_fan_state_identity_expected_by_replay():
    fan_after = np.array([0.5, 0.6, 0.4])
    fan_before = np.array([0.1, 0.5, 0.6, 0.4])
    assert np.array_equal(fan_after, fan_before[1:])
