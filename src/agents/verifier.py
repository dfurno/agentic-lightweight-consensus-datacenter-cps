from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.schemas import ConsensusSnapshot, ControlAction, VerifierDecision
from src.simulation.actuators import ActuatorState
from src.utils.io import read_yaml


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class VerifierAgent:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or read_yaml("configs/sla.yaml")

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/sla.yaml") -> "VerifierAgent":
        return cls(read_yaml(path))

    def verify(
        self,
        action: ControlAction,
        consensus: ConsensusSnapshot,
        actuator_state: ActuatorState | None = None,
        now: datetime | None = None,
    ) -> VerifierDecision:
        actuator_state = actuator_state or ActuatorState()
        now = now or datetime.now(timezone.utc)
        reasons: list[str] = []
        process_checks = 0
        process_passed = 0
        if action.used_consensus:
            process_passed += 1
        else:
            reasons.append("Planner did not mark consensus as used.")
        process_checks += 1
        try:
            age = (now - parse_timestamp(consensus.timestamp)).total_seconds()
        except ValueError:
            age = float("inf")
        if age <= float(self.config["max_consensus_age_seconds"]):
            process_passed += 1
        else:
            reasons.append("Consensus is stale.")
        process_checks += 1
        if consensus.confidence >= float(self.config["min_confidence"]):
            process_passed += 1
        else:
            reasons.append("Consensus confidence is below threshold.")
        process_checks += 1
        if action.consensus_timestamp == consensus.timestamp:
            process_passed += 1
        else:
            reasons.append("Action does not reference latest consensus timestamp.")
        process_checks += 1
        safety_checks = 0
        safety_passed = 0
        emergency = float(self.config["emergency_temperature_c"])
        if consensus.temperature >= emergency and action.action_type != "raise_alarm":
            reasons.append("Emergency threshold requires raise_alarm.")
        else:
            safety_passed += 1
        safety_checks += 1
        predicted = self._predict_temperature(consensus.temperature, action)
        if predicted <= float(self.config["temperature_sla_c"]) or action.action_type == "raise_alarm":
            safety_passed += 1
        else:
            reasons.append("Predicted temperature remains above SLA threshold.")
        safety_checks += 1
        if self._within_bounds(action, actuator_state):
            safety_passed += 1
        else:
            reasons.append("Action violates actuator bounds or slew-rate constraints.")
        safety_checks += 1
        accepted = not reasons
        return VerifierDecision(
            accepted=accepted,
            process_reward=process_passed / process_checks,
            outcome_reward=safety_passed / safety_checks,
            rejection_reasons=reasons,
            required_refinement=None if accepted else "Revise action to satisfy freshness, confidence, and deterministic SLA constraints.",
        )

    def _predict_temperature(self, temperature: float, action: ControlAction) -> float:
        prediction = self.config.get("prediction", {})
        if action.action_type == "increase_fan_speed":
            return temperature - action.magnitude * float(prediction.get("cooling_gain_per_fan_unit", 3.0))
        if action.action_type == "decrease_setpoint":
            return temperature - action.magnitude * float(prediction.get("setpoint_gain", 0.5))
        return temperature

    def _within_bounds(self, action: ControlAction, state: ActuatorState) -> bool:
        actuators = self.config["actuators"]
        if action.action_type == "increase_fan_speed":
            new_value = state.fan_speed + action.magnitude
            cfg = actuators["fan_speed"]
            return cfg["min"] <= new_value <= cfg["max"] and action.magnitude <= cfg["max_slew_rate"]
        if action.action_type == "decrease_setpoint":
            new_value = state.setpoint_c - action.magnitude
            cfg = actuators["setpoint_c"]
            return cfg["min"] <= new_value <= cfg["max"] and action.magnitude <= cfg["max_slew_rate"]
        return True
