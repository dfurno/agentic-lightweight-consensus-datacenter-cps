from __future__ import annotations

from datetime import datetime, timezone

from src.agents.schemas import ControlAction


def deterministic_policy(temperature: float, threshold: float, target_zone: str = "zone-0") -> ControlAction:
    if temperature >= threshold:
        return ControlAction(
            action_type="increase_fan_speed",
            target_zone=target_zone,
            magnitude=0.1,
            duration=60,
            reasoning_summary="Temperature is above SLA threshold.",
            consensus_timestamp=datetime.now(timezone.utc).isoformat(),
            used_consensus=True,
        )
    return ControlAction(
        action_type="maintain",
        target_zone=target_zone,
        magnitude=0.0,
        duration=60,
        reasoning_summary="Temperature is within SLA threshold.",
        consensus_timestamp=datetime.now(timezone.utc).isoformat(),
        used_consensus=True,
    )
