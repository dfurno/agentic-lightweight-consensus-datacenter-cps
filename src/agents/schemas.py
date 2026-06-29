from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ActionType = Literal["increase_fan_speed", "decrease_setpoint", "maintain", "raise_alarm"]


class ControlAction(BaseModel):
    action_type: ActionType
    target_zone: str
    magnitude: float = Field(ge=0.0)
    duration: int = Field(gt=0)
    reasoning_summary: str
    consensus_timestamp: str
    used_consensus: bool


class ConsensusSnapshot(BaseModel):
    temperature: float
    timestamp: str
    confidence: float
    anomaly_count: int = 0


class VerifierDecision(BaseModel):
    accepted: bool
    process_reward: float
    outcome_reward: float
    rejection_reasons: list[str]
    required_refinement: str | None = None
