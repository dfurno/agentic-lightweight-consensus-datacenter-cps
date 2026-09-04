from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.agents.schemas import ConsensusSnapshot, ControlAction, VerifierDecision
from src.agents.verifier import VerifierAgent
from src.runtime.control import ControlRuntime, PlannedAction, VerifierResponse


NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def snapshot(*, temperature=25.0, confidence=0.8, anomalies=0, seconds=0, zone="zone-0"):
    return ConsensusSnapshot(temperature=temperature, confidence=confidence, anomaly_count=anomalies, timestamp=(NOW + timedelta(seconds=seconds)).isoformat(), zone=zone)


def action(s, *, kind="maintain", magnitude=0.0, zone="zone-0", used=True):
    return ControlAction(action_type=kind, target_zone=zone, magnitude=magnitude, duration=1, reasoning_summary="controlled test", consensus_timestamp=s.timestamp, used_consensus=used)


def runtime(**policy):
    base = {"confidence_threshold": 0.35, "anomaly_threshold": 3, "temperature_margin": 0.02, "planner_deadline_seconds": 0.05, "verifier_deadline_seconds": 0.02}
    base.update(policy)
    return ControlRuntime(VerifierAgent.from_yaml(), policy=base)


def test_nominal_fast_path_and_trigger_boundaries():
    rt = runtime()
    event = rt.step(snapshot(), monotonic_now=0, wall_now=NOW)
    assert event.executed_action == "maintain" and event.triggers == []
    assert rt.trigger_reasons(snapshot(confidence=0.35, anomalies=2, temperature=31.979)) == []
    assert rt.trigger_reasons(snapshot(confidence=0.349, anomalies=3, temperature=31.98)) == ["low_confidence", "anomaly_count", "thermal_margin"]


@pytest.mark.parametrize("kwargs,expected", [
    ({"confidence": 0.34}, ["low_confidence"]),
    ({"anomalies": 3}, ["anomaly_count"]),
    ({"temperature": 31.98}, ["thermal_margin"]),
])
def test_isolated_triggers(kwargs, expected):
    assert runtime().trigger_reasons(snapshot(**kwargs)) == expected


def test_planner_pending_fast_path_then_late_snapshot_rejected():
    rt = runtime()
    first = snapshot(anomalies=3)
    e0 = rt.step(first, monotonic_now=0, wall_now=NOW, planner=lambda s: PlannedAction(action(s), 0.01))
    e1 = rt.step(snapshot(anomalies=3, seconds=1), monotonic_now=0.02, wall_now=NOW + timedelta(seconds=1))
    assert e0.planner_status == "pending" and e0.executed_action == "maintain"
    assert e1.planner_status == "superseded" and "superseded_snapshot" in e1.reasons


def test_planner_timeout_and_exception_continue_fast_path():
    rt = runtime()
    e0 = rt.step(snapshot(anomalies=3), monotonic_now=0, wall_now=NOW, planner=lambda s: PlannedAction(action(s), 1.0))
    e1 = rt.step(snapshot(anomalies=3, seconds=1), monotonic_now=0.1, wall_now=NOW + timedelta(seconds=1))
    assert e0.executed_action == "maintain" and e1.planner_status == "timeout"
    rt2 = runtime()
    e2 = rt2.step(snapshot(anomalies=3), monotonic_now=0, wall_now=NOW, planner=lambda s: (_ for _ in ()).throw(RuntimeError("injected")))
    assert e2.planner_status == "exception" and e2.executed_action == "maintain" and e2.fallback == "local_fast_path"


def test_verifier_timeout_and_exception_do_not_change_state():
    accepted = VerifierDecision(accepted=True, process_reward=1, outcome_reward=1, rejection_reasons=[])
    for call, reason in [
        (lambda *args: VerifierResponse(accepted, 1.0), "verifier_timeout"),
        (lambda *args: (_ for _ in ()).throw(RuntimeError("injected")), "verifier_exception:RuntimeError"),
    ]:
        rt = runtime(); before = rt.state.fan_speed
        event = rt.step(snapshot(temperature=32.1), monotonic_now=0, wall_now=NOW, verifier_call=call)
        assert reason in event.reasons and rt.state.fan_speed == before


@pytest.mark.parametrize("bad", [
    snapshot(seconds=-181), snapshot(seconds=1), snapshot(zone="zone-x"), snapshot(temperature=float("nan")),
])
def test_invalid_snapshot_never_changes_actuator(bad):
    rt = runtime(); before = rt.state.fan_speed
    event = rt.step(bad, monotonic_now=0, wall_now=NOW)
    assert event.reasons and rt.state.fan_speed == before


def test_malformed_json_and_wrong_action_zone_or_limit_are_rejected():
    with pytest.raises(ValidationError):
        ConsensusSnapshot.model_validate_json("{bad json")
    for proposed in [lambda s: action(s, kind="increase_fan_speed", magnitude=0.3), lambda s: action(s, zone="zone-x")]:
        rt = runtime(); s = snapshot(anomalies=3)
        event = rt.step(s, monotonic_now=0, wall_now=NOW, planner=lambda x: PlannedAction(proposed(x)))
        assert event.executed_action == "raise_alarm" and rt.state.fan_speed == 0.5


def test_authorization_is_single_use_and_payload_bound():
    rt = runtime(); s = snapshot(); original = action(s, kind="increase_fan_speed", magnitude=0.1)
    token = rt.authorize(original, s)
    mutated = original.model_copy(update={"magnitude": 0.2})
    assert not rt.apply_authorized(mutated, s, token)
    assert not rt.apply_authorized(original, s, token)
    assert not rt.apply_authorized(original, s, "missing")


def test_persistent_state_accumulates_only_to_limit():
    rt = runtime()
    for i in range(4):
        s = snapshot(anomalies=3, seconds=i)
        rt.step(s, monotonic_now=i, wall_now=NOW + timedelta(seconds=i), planner=lambda x: PlannedAction(action(x, kind="increase_fan_speed", magnitude=0.2)))
    assert rt.state.fan_speed == pytest.approx(0.9)
    s = snapshot(anomalies=3, seconds=5)
    event = rt.step(s, monotonic_now=5, wall_now=NOW + timedelta(seconds=5), planner=lambda x: PlannedAction(action(x, kind="increase_fan_speed", magnitude=0.2)))
    assert rt.state.fan_speed == pytest.approx(0.9) and event.executed_action == "raise_alarm"


def test_cooldown_records_trigger_but_does_not_call_planner():
    rt = runtime(min_llm_interval_ticks=2); calls = []
    planner = lambda s: (calls.append(s.timestamp) or PlannedAction(action(s)))
    for i in range(2):
        rt.step(snapshot(anomalies=3, seconds=i), monotonic_now=i, wall_now=NOW + timedelta(seconds=i), planner=planner)
    assert len(calls) == 1 and rt.events[1].triggers == ["anomaly_count"]


@pytest.mark.parametrize("bad", [
    snapshot(temperature=32.1, seconds=1),
    snapshot(temperature=32.1, zone="zone-x"),
    snapshot(temperature=32.1, confidence=float("inf")),
])
def test_invalid_actuating_snapshot_is_held(bad):
    rt = runtime()
    event = rt.step(bad, monotonic_now=0, wall_now=NOW)
    assert event.state_after == event.state_before
    assert event.action_source == "local_hold_alarm"
    assert event.executed_action == "raise_alarm"
    assert event.verifier_accepted is None


def test_pending_request_retains_original_deadline_and_completes():
    rt = runtime(); s = snapshot(anomalies=3); calls = []
    planner = lambda current: (calls.append(current.timestamp) or PlannedAction(
        action(current, kind="increase_fan_speed", magnitude=0.1), 0.04))
    for i in range(5):
        rt.step(s, monotonic_now=i * 0.01, wall_now=NOW, planner=planner)
    assert len(calls) == 1
    assert rt.events[-1].planner_status == "ready"
    assert rt.state.fan_speed == pytest.approx(0.6)
    assert len({event.request_id for event in rt.events if event.request_id}) == 1


def test_reissued_authorization_is_unique_and_old_token_stays_consumed():
    rt = runtime(); s = snapshot(); proposed = action(s, kind="increase_fan_speed", magnitude=0.1)
    old = rt.authorize(proposed, s)
    assert rt.apply_authorized(proposed, s, old)
    new = rt.authorize(proposed, s)
    assert old != new
    assert not rt.apply_authorized(proposed, s, old)
    assert rt.apply_authorized(proposed, s, new)


def test_callbacks_receive_copies_and_proposal_remains_bound():
    rt = runtime(); s = snapshot(temperature=50, anomalies=3)
    def mutating_planner(received):
        received.temperature = 25
        return PlannedAction(action(received, kind="increase_fan_speed", magnitude=0.1))
    event = rt.step(s, monotonic_now=0, wall_now=NOW, planner=mutating_planner)
    assert s.temperature == 50
    assert event.snapshot_payload["temperature"] == 50
    assert event.executed_action == "raise_alarm"
    assert event.verifier_accepted is False
