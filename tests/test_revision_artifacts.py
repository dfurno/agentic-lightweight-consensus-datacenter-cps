import pytest

from src.revision.artifacts import scenario_parts


def test_scenario_parts_parses_bias_beta_without_matching_attack_name():
    parts = scenario_parts("full_seed11_bias_0p2_noise0p25_s9_a0p2_b0p7")

    assert parts["attack_type"] == "bias"
    assert parts["alpha"] == pytest.approx(0.2)
    assert parts["beta"] == pytest.approx(0.7)


def test_scenario_parts_parses_non_bias_control():
    parts = scenario_parts("full_seed37_drift_0p4_noise0p5_s13_a0p3_b0p8")

    assert parts["attack_type"] == "drift"
    assert parts["beta"] == pytest.approx(0.8)


def test_scenario_parts_rejects_missing_numeric_beta():
    with pytest.raises(ValueError, match="Malformed scenario name"):
        scenario_parts("full_seed11_bias_0p2_noise0p25_s9_a0p2_bias")
