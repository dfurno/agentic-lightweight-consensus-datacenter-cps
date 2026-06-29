from __future__ import annotations

from src.agents.schemas import ConsensusSnapshot, ControlAction, VerifierDecision


def refine_action(action: ControlAction, consensus: ConsensusSnapshot, decision: VerifierDecision) -> ControlAction:
    if consensus.temperature >= 38.0:
        return action.model_copy(update={"action_type": "raise_alarm", "magnitude": 0.0, "used_consensus": True})
    if "Action violates actuator bounds or slew-rate constraints." in decision.rejection_reasons:
        return action.model_copy(update={"magnitude": min(action.magnitude, 0.1)})
    if "Consensus confidence is below threshold." in decision.rejection_reasons:
        return action.model_copy(update={"action_type": "raise_alarm", "magnitude": 0.0, "used_consensus": True})
    if "Predicted temperature remains above SLA threshold." in decision.rejection_reasons:
        return action.model_copy(update={"action_type": "increase_fan_speed", "magnitude": 0.1, "used_consensus": True})
    return action.model_copy(update={"consensus_timestamp": consensus.timestamp, "used_consensus": True})
