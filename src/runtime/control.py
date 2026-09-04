from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

from src.agents.schemas import ConsensusSnapshot, ControlAction, VerifierDecision
from src.agents.verifier import VerifierAgent, parse_timestamp
from src.simulation.actuators import ActuatorState
from src.simulation.policies import deterministic_policy


@dataclass(frozen=True)
class PlannedAction:
    action: ControlAction
    delay_seconds: float = 0.0


@dataclass(frozen=True)
class VerifierResponse:
    decision: VerifierDecision
    delay_seconds: float = 0.0


@dataclass
class RuntimeEvent:
    tick: int
    request_id: str | None
    snapshot_timestamp: str
    snapshot_id: str
    snapshot_payload: dict[str, object]
    triggers: list[str]
    planner_status: str
    decision: str
    proposal_payload: dict[str, object] | None
    proposal_id: str | None
    verifier_accepted: bool | None
    verifier_rejection_reasons: list[str]
    reasons: list[str]
    fallback: str | None
    action_source: str
    executed_action: str
    state_before: dict[str, float]
    state_after: dict[str, float]
    planner_latency_seconds: float | None
    verifier_latency_seconds: float | None
    step_latency_seconds: float


@dataclass(frozen=True)
class _Pending:
    request_id: str
    planned: PlannedAction
    source_snapshot: ConsensusSnapshot
    source_snapshot_id: str
    ready_at: float
    deadline_at: float
    planner_latency_seconds: float


class ControlRuntime:
    """Shared simulated control path with an internal, single-use gate."""

    def __init__(self, verifier: VerifierAgent, *, zone: str = "zone-0", policy: dict | None = None) -> None:
        self.verifier = verifier
        self.zone = zone
        self.policy = policy or {}
        self.state = ActuatorState()
        self.tick = 0
        self.last_trigger_tick = -10**9
        self.pending: _Pending | None = None
        self._tokens: dict[str, str] = {}
        self._token_sequence = 0
        self._request_sequence = 0
        self.events: list[RuntimeEvent] = []

    @staticmethod
    def _canonical(model: ConsensusSnapshot | ControlAction) -> str:
        return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=True)

    def _snapshot_key(self, snapshot: ConsensusSnapshot) -> str:
        return hashlib.sha256(self._canonical(snapshot).encode()).hexdigest()

    def _payload_key(self, action: ControlAction, snapshot: ConsensusSnapshot) -> str:
        return hashlib.sha256((self._canonical(action) + self._snapshot_key(snapshot)).encode()).hexdigest()

    def authorize(self, action: ControlAction, snapshot: ConsensusSnapshot) -> str:
        """Issue an internal one-shot capability; exposed only for gate tests."""
        self._token_sequence += 1
        payload_key = self._payload_key(action, snapshot)
        token = hashlib.sha256(f"{payload_key}:{self._token_sequence}".encode()).hexdigest()
        self._tokens[token] = payload_key
        return token

    def apply_authorized(self, action: ControlAction, snapshot: ConsensusSnapshot, token: str) -> bool:
        expected = self._tokens.pop(token, None)
        if expected is None or expected != self._payload_key(action, snapshot):
            return False
        return self._apply_contract(action)

    def _validate_snapshot(self, snapshot: ConsensusSnapshot, now: datetime) -> list[str]:
        reasons: list[str] = []
        if snapshot.zone != self.zone:
            reasons.append("snapshot_zone_mismatch")
        if not math.isfinite(snapshot.temperature) or not math.isfinite(snapshot.confidence) or snapshot.anomaly_count < 0:
            reasons.append("invalid_snapshot_schema")
        try:
            age = (now - parse_timestamp(snapshot.timestamp)).total_seconds()
            if age < 0:
                reasons.append("future_snapshot")
            elif age > float(self.verifier.config["max_consensus_age_seconds"]):
                reasons.append("stale_snapshot")
        except (TypeError, ValueError):
            reasons.append("invalid_snapshot_timestamp")
        return reasons

    def trigger_reasons(self, snapshot: ConsensusSnapshot) -> list[str]:
        reasons: list[str] = []
        if bool(self.policy.get("always_supervise", False)):
            reasons.append("scheduled_control")
        if snapshot.confidence < float(self.policy.get("confidence_threshold", 0.35)):
            reasons.append("low_confidence")
        if snapshot.anomaly_count >= int(self.policy.get("anomaly_threshold", 3)):
            reasons.append("anomaly_count")
        margin = float(self.policy.get("temperature_margin", 0.02))
        if snapshot.temperature >= float(self.verifier.config["temperature_sla_c"]) - margin:
            reasons.append("thermal_margin")
        return reasons

    def _apply_contract(self, action: ControlAction) -> bool:
        if action.target_zone != self.zone or not math.isfinite(action.magnitude):
            return False
        cfg = self.verifier.config["actuators"]
        if action.action_type == "increase_fan_speed":
            item = cfg["fan_speed"]
            value = self.state.fan_speed + action.magnitude
            if action.magnitude > item["max_slew_rate"] or not item["min"] <= value <= item["max"]:
                return False
            self.state.fan_speed = value
        elif action.action_type == "decrease_fan_speed":
            item = cfg["fan_speed"]
            value = self.state.fan_speed - action.magnitude
            if action.magnitude > item["max_slew_rate"] or not item["min"] <= value <= item["max"]:
                return False
            self.state.fan_speed = value
        elif action.action_type == "decrease_setpoint":
            item = cfg["setpoint_c"]
            value = self.state.setpoint_c - action.magnitude
            if action.magnitude > item["max_slew_rate"] or not item["min"] <= value <= item["max"]:
                return False
            self.state.setpoint_c = value
        elif action.action_type not in {"maintain", "raise_alarm"}:
            return False
        return True

    def _local_alarm(self, timestamp: str) -> ControlAction:
        return ControlAction(action_type="raise_alarm", target_zone=self.zone, magnitude=0, duration=1,
                             reasoning_summary="Local hold/alarm fallback", consensus_timestamp=timestamp,
                             used_consensus=False)

    def step(self, snapshot: ConsensusSnapshot, *, monotonic_now: float, wall_now: datetime,
             planner: Callable[[ConsensusSnapshot], PlannedAction] | None = None,
             verifier_call: Callable[[ControlAction, ConsensusSnapshot, ActuatorState, datetime], VerifierDecision | VerifierResponse] | None = None) -> RuntimeEvent:
        step_start = time.perf_counter()
        authoritative = snapshot.model_copy(deep=True)
        snapshot_id = self._snapshot_key(authoritative)
        before = asdict(self.state)
        invalid = self._validate_snapshot(authoritative, wall_now)
        triggers = self.trigger_reasons(authoritative) if not invalid else []
        reasons = list(invalid)
        planner_status = "not_called"
        fallback: str | None = None
        decision = "fast_path"
        selected: ControlAction | None = None
        request_id: str | None = None
        planner_latency: float | None = None

        if invalid:
            if self.pending is not None:
                reasons.append("pending_cancelled_invalid_snapshot")
                self.pending = None
            selected = self._local_alarm(authoritative.timestamp)
            decision = "local_fallback"
            fallback = "local_hold_alarm"
        else:
            if self.pending is not None:
                request_id = self.pending.request_id
                planner_latency = self.pending.planner_latency_seconds
                if monotonic_now > self.pending.deadline_at:
                    reasons.append("planner_timeout")
                    planner_status = "timeout"
                    self.pending = None
                elif self.pending.source_snapshot_id != snapshot_id:
                    reasons.append("superseded_snapshot")
                    planner_status = "superseded"
                    self.pending = None
                elif monotonic_now >= self.pending.ready_at:
                    selected = self.pending.planned.action.model_copy(deep=True)
                    planner_status = "ready"
                    self.pending = None
                else:
                    planner_status = "pending"

            cooldown = int(self.policy.get("min_llm_interval_ticks", 0))
            may_launch = self.pending is None and selected is None and planner_status not in {"timeout", "superseded"}
            if may_launch and triggers and self.tick - self.last_trigger_tick >= cooldown and planner is not None:
                self.last_trigger_tick = self.tick
                self._request_sequence += 1
                request_id = f"request-{self._request_sequence:06d}"
                planner_start = time.perf_counter()
                try:
                    raw = planner(authoritative.model_copy(deep=True))
                    planner_latency = time.perf_counter() - planner_start
                    planned = PlannedAction(raw.action.model_copy(deep=True), raw.delay_seconds)
                    deadline = float(self.policy.get("planner_deadline_seconds", 0.05))
                    self.pending = _Pending(request_id, planned, authoritative.model_copy(deep=True), snapshot_id,
                                            monotonic_now + planned.delay_seconds, monotonic_now + deadline,
                                            planner_latency)
                    planner_status = "pending" if planned.delay_seconds > 0 else "ready"
                    if planned.delay_seconds <= 0:
                        selected = planned.action.model_copy(deep=True)
                        self.pending = None
                except Exception as exc:
                    planner_latency = time.perf_counter() - planner_start
                    reasons.append(f"planner_exception:{type(exc).__name__}")
                    planner_status = "exception"

            if selected is None:
                selected = deterministic_policy(authoritative.temperature, self.verifier.config["temperature_sla_c"],
                                                self.zone, authoritative.timestamp)
                if planner_status in {"pending", "timeout", "exception", "superseded"}:
                    fallback = "local_fast_path"
            else:
                decision = "supervisor"

        proposal_payload = selected.model_dump(mode="json")
        proposal_id = self._payload_key(selected, authoritative)
        verifier_accepted: bool | None = None
        verifier_rejections: list[str] = []
        verifier_latency: float | None = None
        executed = "none"
        action_source = "none"

        if invalid:
            if self._apply_contract(selected):
                executed = selected.action_type
                action_source = "local_hold_alarm"
        else:
            call = verifier_call or (lambda a, s, state, now: self.verifier.verify(a, s, state, now))
            verifier_start = time.perf_counter()
            try:
                response = call(selected.model_copy(deep=True), authoritative.model_copy(deep=True),
                                ActuatorState(**asdict(self.state)), wall_now)
                verifier_latency = time.perf_counter() - verifier_start
                if isinstance(response, VerifierResponse):
                    if response.delay_seconds > float(self.policy.get("verifier_deadline_seconds", 0.02)):
                        reasons.append("verifier_timeout")
                        verifier_accepted = False
                    else:
                        verifier_accepted = response.decision.accepted
                        verifier_rejections = list(response.decision.rejection_reasons)
                else:
                    verifier_accepted = response.accepted
                    verifier_rejections = list(response.rejection_reasons)
                reasons.extend(verifier_rejections)
            except Exception as exc:
                verifier_latency = time.perf_counter() - verifier_start
                verifier_accepted = False
                reasons.append(f"verifier_exception:{type(exc).__name__}")

            if verifier_accepted:
                token = self.authorize(selected, authoritative)
                if self.apply_authorized(selected, authoritative, token):
                    executed = selected.action_type
                    action_source = "supervisor" if decision == "supervisor" else "fast_path"
                else:
                    reasons.append("actuation_contract_rejected")
                    verifier_accepted = False
            if not verifier_accepted:
                fallback = "local_hold_alarm"
                local = self._local_alarm(authoritative.timestamp)
                if self._apply_contract(local):
                    executed = local.action_type
                    action_source = "local_hold_alarm"

        event = RuntimeEvent(
            self.tick, request_id, authoritative.timestamp, snapshot_id,
            authoritative.model_dump(mode="json"), triggers, planner_status, decision,
            proposal_payload, proposal_id, verifier_accepted, verifier_rejections, reasons,
            fallback, action_source, executed, before, asdict(self.state), planner_latency,
            verifier_latency, time.perf_counter() - step_start,
        )
        self.events.append(event)
        self.tick += 1
        return event
