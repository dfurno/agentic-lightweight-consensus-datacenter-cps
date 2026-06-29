import numpy as np

from src.consensus.owa import owa_weights, reliability_ordered_owa_weights, weak_majority_quantifier


def test_owa_weights_sum_to_one_for_multiple_n():
    for n in [1, 2, 3, 5, 9, 20]:
        assert np.isclose(owa_weights(n).sum(), 1.0)


def test_owa_weights_are_non_negative():
    for n in [1, 4, 11]:
        assert np.all(owa_weights(n) >= 0.0)


def test_weak_majority_quantifier():
    assert weak_majority_quantifier(0.1) == 0.0
    assert weak_majority_quantifier(0.9) == 1.0
    assert np.isclose(weak_majority_quantifier(0.55), 0.5)


def test_reliability_ordered_weights_reward_reliable_prefix():
    weights = reliability_ordered_owa_weights(9)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] >= weights[-1]
    assert weights[:5].sum() > weights[-4:].sum()
