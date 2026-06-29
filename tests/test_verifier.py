from datetime import datetime, timedelta, timezone

from src.agents.schemas import ConsensusSnapshot, ControlAction
from src.agents.verifier import VerifierAgent
from src.simulation.actuators import ActuatorState


def action(timestamp: str, magnitude: float = 0.1) -> ControlAction:
    return ControlAction(
        action_type="increase_fan_speed",
        target_zone="zone-0",
        magnitude=magnitude,
        duration=60,
        reasoning_summary="test",
        consensus_timestamp=timestamp,
        used_consensus=True,
    )


def test_verifier_rejects_stale_consensus():
    verifier = VerifierAgent.from_yaml()
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    decision = verifier.verify(action(old), ConsensusSnapshot(temperature=31.0, timestamp=old, confidence=0.9))
    assert not decision.accepted
    assert "Consensus is stale." in decision.rejection_reasons


def test_verifier_rejects_out_of_bounds_actuation():
    verifier = VerifierAgent.from_yaml()
    now = datetime.now(timezone.utc).isoformat()
    decision = verifier.verify(
        action(now, magnitude=0.4),
        ConsensusSnapshot(temperature=33.0, timestamp=now, confidence=0.9),
        ActuatorState(fan_speed=0.9),
    )
    assert not decision.accepted


def test_verifier_accepts_safe_action():
    verifier = VerifierAgent.from_yaml()
    now = datetime.now(timezone.utc).isoformat()
    decision = verifier.verify(action(now), ConsensusSnapshot(temperature=31.0, timestamp=now, confidence=0.9))
    assert decision.accepted
