import numpy as np

from src.consensus.fpr_owa import ConsensusResult, fpr_owa_consensus


def test_fpr_owa_returns_diagnostics():
    result = fpr_owa_consensus(np.array([25.0, 25.1, 24.9, 40.0]))
    assert isinstance(result, ConsensusResult)
    assert result.weights.shape == (4,)
    assert result.fpr_matrix.shape == (4, 4)
    assert result.ordered_sensor_indices.shape == (4,)
    assert result.anomaly_flags.shape == (4,)
    assert result.aggregated_value < 40.0


def test_strict_dominance_order_is_preserved():
    result = fpr_owa_consensus(np.array([0.0, 1.0, 10.0]))
    assert result.ordered_sensor_indices.tolist() == [1, 0, 2]


def test_equal_reliability_equal_values_use_ascending_stable_id():
    result = fpr_owa_consensus(np.array([2.0, 2.0]), sensor_ids=np.array([9, 3]))
    assert result.ordered_sensor_indices.tolist() == [1, 0]


def test_equal_reliability_distinct_values_use_ascending_stable_id():
    # The two outer values are equally distant from the median and therefore
    # have equal reliability/dominance despite carrying distinct values.
    result = fpr_owa_consensus(
        np.array([0.0, 1.0, 2.0]), sensor_ids=np.array([30, 20, 10])
    )
    assert result.ordered_sensor_indices.tolist() == [1, 2, 0]


def test_tie_order_is_repeatable_and_default_ids_match_input_positions():
    values = np.array([0.0, 1.0, 2.0])
    orders = [fpr_owa_consensus(values).ordered_sensor_indices.tolist() for _ in range(5)]
    assert orders == [[1, 0, 2]] * 5


def test_sensor_ids_must_map_one_to_one_to_input_order():
    with np.testing.assert_raises(ValueError):
        fpr_owa_consensus(np.array([1.0, 2.0]), sensor_ids=np.array([4, 4]))
