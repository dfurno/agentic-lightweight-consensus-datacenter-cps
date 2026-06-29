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
