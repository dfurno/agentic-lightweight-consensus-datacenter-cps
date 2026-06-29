import numpy as np

from src.simulation.attacks import apply_attack


def test_bias_attack_changes_window_and_labels():
    signal = np.ones(10)
    attacked, labels = apply_attack(signal, "bias", np.random.default_rng(1), start_fraction=0.2, end_fraction=0.5)
    assert labels.sum() == 3
    assert np.all(attacked[labels] > signal[labels])


def test_freeze_attack_is_constant_in_window():
    signal = np.arange(10, dtype=float)
    attacked, labels = apply_attack(signal, "freeze", np.random.default_rng(1), start_fraction=0.2, end_fraction=0.6)
    assert len(set(attacked[labels])) == 1
