from datetime import datetime, timezone

import pytest
import yaml

from src.agents.schemas import ConsensusSnapshot
from src.runtime.control import ControlRuntime
from src.simulation.actuators import ActuatorState
from src.simulation.thermal_pilot import (ControllerState, ThermalExperimentVerifier,
                                          exogenous_arrays, supervisory_action)


@pytest.fixture
def cfg():
    return yaml.safe_load(open("configs/thermal_pilot.yaml"))


def snap(temp=30.0, confidence=1.0):
    return ConsensusSnapshot(temperature=temp, confidence=confidence, anomaly_count=0,
                             timestamp=datetime(2026, 9, 3, tzinfo=timezone.utc).isoformat(), zone="zone-0")


def test_discrete_plant_is_stable_and_target_reachable(cfg):
    p, c = cfg["plant"], cfg["control"]
    assert abs(1 - p["thermal_exchange_per_step"]) < 1
    fan = (p["thermal_exchange_per_step"] * (p["ambient_base_c"] - c["target_temperature_c"])
           + p["load_base_c_per_step"]) / p["fan_gain_c_per_step"]
    assert c["fan_min"] <= fan <= c["fan_max"]


def test_controller_increases_and_relaxes_after_dwell(cfg):
    state = ActuatorState(fan_speed=0.5, setpoint_c=24); control = ControllerState()
    assert supervisory_action(snap(31), state, control, cfg).action_type == "increase_fan_speed"
    actions = [supervisory_action(snap(29), state, control, cfg) for _ in range(cfg["control"]["safe_dwell_ticks"])]
    assert [a.action_type for a in actions[:-1]] == ["maintain"] * (len(actions) - 1)
    assert actions[-1].action_type == "decrease_fan_speed"


def test_runtime_enforces_bidirectional_fan_limits(cfg):
    verifier = ThermalExperimentVerifier(cfg); rt = ControlRuntime(verifier)
    rt.state.fan_speed = 0.05
    action = supervisory_action(snap(29), rt.state, ControllerState(below_band_streak=2), cfg)
    token = rt.authorize(action, snap(29)); assert rt.apply_authorized(action, snap(29), token)
    assert rt.state.fan_speed == pytest.approx(0.0)
    assert not rt.apply_authorized(action, snap(29), token)


def test_experimental_verifier_progress_positive_and_negative(cfg):
    verifier = ThermalExperimentVerifier(cfg); verifier.set_exogenous(26, 0.7)
    state = ActuatorState(fan_speed=0.5, setpoint_c=24)
    cooling = supervisory_action(snap(33), state, ControllerState(), cfg)
    assert verifier.verify(cooling, snap(33), state).accepted
    maintain = cooling.model_copy(update={"action_type": "maintain", "magnitude": 0.0})
    decision = verifier.verify(maintain, snap(33), state)
    assert not decision.accepted and "above_sla_requires_cooling_progress" in decision.rejection_reasons


def test_below_sla_rejects_relaxation_that_crosses_sla(cfg):
    verifier = ThermalExperimentVerifier(cfg); verifier.set_exogenous(35, 1.0)
    state = ActuatorState(fan_speed=0.1, setpoint_c=24)
    action = supervisory_action(snap(29), state, ControllerState(below_band_streak=2), cfg)
    assert not verifier.verify(action, snap(31.9), state).accepted


def test_pairing_is_deterministic(cfg):
    a, b = exogenous_arrays(11, 4, cfg), exogenous_arrays(11, 4, cfg)
    assert (a["process_noise"] == b["process_noise"]).all()
    assert (a["sensor_noise"] == b["sensor_noise"]).all()
    assert (a["attacked"] == b["attacked"]).all()
