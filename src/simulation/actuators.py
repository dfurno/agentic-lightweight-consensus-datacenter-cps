from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActuatorState:
    fan_speed: float = 0.5
    setpoint_c: float = 24.0
