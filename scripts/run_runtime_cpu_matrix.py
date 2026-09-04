#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.schemas import ConsensusSnapshot, ControlAction, VerifierDecision
from src.agents.verifier import VerifierAgent
from src.runtime.control import ControlRuntime, PlannedAction, VerifierResponse


NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def snap(**updates):
    data = {"temperature": 25.0, "confidence": 0.8, "anomaly_count": 0, "timestamp": NOW.isoformat(), "zone": "zone-0"}
    data.update(updates); return ConsensusSnapshot(**data)


def act(snapshot, **updates):
    data = {"action_type": "maintain", "target_zone": "zone-0", "magnitude": 0.0, "duration": 1, "reasoning_summary": "controlled CPU matrix", "consensus_timestamp": snapshot.timestamp, "used_consensus": True}
    data.update(updates); return ControlAction(**data)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--matrix", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--commit", required=True); args = parser.parse_args()
    actual_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if args.commit != actual_commit: raise SystemExit(f"Commit mismatch: requested {args.commit}, checked out {actual_commit}")
    if args.output.exists(): raise SystemExit("Output destination already exists")
    args.output.mkdir(parents=True)
    matrix = json.loads(args.matrix.read_text()); verifier = VerifierAgent.from_yaml()
    rows = []; logs = []
    accepted = VerifierDecision(accepted=True, process_reward=1, outcome_reward=1, rejection_reasons=[])

    for spec in matrix["cases"]:
        kind = spec["kind"]; rt = ControlRuntime(verifier, policy={"confidence_threshold": 0.35, "anomaly_threshold": 3, "temperature_margin": 0.02, "min_llm_interval_ticks": 2 if kind == "cooldown" else 0, "planner_deadline_seconds": matrix["planner_deadline_seconds"], "verifier_deadline_seconds": matrix["verifier_deadline_seconds"]})
        events = []; reason = None; planner_calls = 0
        def planner(s):
            nonlocal planner_calls; planner_calls += 1
            if kind == "planner_exception": raise RuntimeError("injected planner failure")
            if kind == "planner_malformed":
                ConsensusSnapshot.model_validate_json("{malformed")
            if kind == "planner_mutation":
                s.temperature = 25.0
            delay = 1.0 if kind == "planner_timeout" else (0.04 if kind == "pending_success" else (0.01 if kind == "late_superseded" else 0.0))
            updates = {}
            if kind in {"valid_action", "persistent_limit"}: updates = {"action_type": "increase_fan_speed", "magnitude": 0.2}
            elif kind == "pending_success": updates = {"action_type": "increase_fan_speed", "magnitude": 0.1}
            elif kind == "planner_mutation": updates = {"action_type": "increase_fan_speed", "magnitude": 0.1}
            elif kind == "action_limit": updates = {"action_type": "increase_fan_speed", "magnitude": 0.3}
            elif kind == "action_zone": updates = {"target_zone": "zone-x"}
            elif kind == "timestamp_mismatch": updates = {"action_type": "increase_fan_speed", "magnitude": 0.1, "consensus_timestamp": "2000-01-01T00:00:00+00:00"}
            elif kind == "no_consensus": updates = {"used_consensus": False}
            return PlannedAction(act(s, **updates), delay)
        verifier_call = None
        if kind == "verifier_exception": verifier_call = lambda *x: (_ for _ in ()).throw(RuntimeError("injected verifier failure"))
        if kind == "verifier_timeout": verifier_call = lambda *x: VerifierResponse(accepted, 1.0)
        try:
            if kind == "auth_reissue":
                s = snap(); original = act(s, action_type="increase_fan_speed", magnitude=0.1)
                old = rt.authorize(original, s); assert rt.apply_authorized(original, s, old)
                new = rt.authorize(original, s)
                if not rt.apply_authorized(original, s, old): reason = "consumed_token_rejected"
                assert rt.apply_authorized(original, s, new)
            elif kind in {"mutated_auth", "missing_auth"}:
                s = snap(); original = act(s, action_type="increase_fan_speed", magnitude=0.1)
                token = rt.authorize(original, s) if kind == "mutated_auth" else "missing"
                candidate = original.model_copy(update={"magnitude": 0.2}) if kind == "mutated_auth" else original
                if not rt.apply_authorized(candidate, s, token): reason = "authorization_rejected"
            else:
                s = snap()
                if kind == "trigger_confidence": s = snap(confidence=0.34)
                elif kind == "trigger_anomaly": s = snap(anomaly_count=3)
                elif kind == "trigger_thermal": s = snap(temperature=31.98)
                elif kind == "trigger_combined": s = snap(confidence=0.34, anomaly_count=3, temperature=31.98)
                elif kind == "threshold_equalities": s = snap(confidence=0.35, anomaly_count=3, temperature=31.98)
                elif kind == "stale_actuating": s = snap(temperature=32.1, timestamp=(NOW - timedelta(seconds=181)).isoformat())
                elif kind == "future_actuating": s = snap(temperature=32.1, timestamp=(NOW + timedelta(seconds=1)).isoformat())
                elif kind == "zone_actuating": s = snap(temperature=32.1, zone="zone-x")
                elif kind == "nonfinite_confidence": s = snap(temperature=32.1, confidence=float("inf"))
                elif kind == "planner_mutation": s = snap(temperature=50.0, anomaly_count=3)
                trigger_kind = kind.startswith("trigger") or kind in {"threshold_equalities", "cooldown", "valid_action", "action_zone", "action_limit", "timestamp_mismatch", "planner_malformed", "planner_mutation", "planner_exception", "planner_timeout", "pending_success", "late_superseded", "persistent_limit"}
                if trigger_kind and not rt.trigger_reasons(s): s = s.model_copy(update={"anomaly_count": 3})
                count = 5 if kind in {"persistent_limit", "pending_success"} else (2 if kind in {"cooldown", "planner_timeout", "late_superseded"} else 1)
                for i in range(count):
                    keep_snapshot = kind == "pending_success"
                    current = s if keep_snapshot else (s.model_copy(update={"timestamp": (NOW + timedelta(seconds=i)).isoformat()}) if i else s)
                    mono_step = 0.01 if kind == "pending_success" else (0.1 if kind == "planner_timeout" else 0.02)
                    event = rt.step(current, monotonic_now=i * mono_step, wall_now=NOW if keep_snapshot else NOW + timedelta(seconds=i), planner=planner if trigger_kind else None, verifier_call=verifier_call)
                    events.append(event)
        except ValidationError:
            reason = "schema_parse_rejected"

        for event in events: logs.append({"case_id": spec["id"], **asdict(event)})
        reasons = [r for e in events for r in e.reasons] + ([reason] if reason else [])
        state_before = events[0].state_before if events else {"fan_speed": 0.5, "setpoint_c": 24.0}; state_after = asdict(rt.state)
        within_bounds = 0 <= rt.state.fan_speed <= 1 and 18 <= rt.state.setpoint_c <= 28
        observed_change = state_before != state_after
        passed = within_bounds
        if "expect_reason" in spec: passed &= spec["expect_reason"] in reasons
        if "expect_reason_prefix" in spec: passed &= any(r.startswith(spec["expect_reason_prefix"]) for r in reasons)
        if "expect_trigger" in spec: passed &= spec["expect_trigger"] in events[0].triggers
        if "expect_trigger_count" in spec: passed &= len(events[0].triggers) == spec["expect_trigger_count"]
        if "expect_change" in spec: passed &= observed_change == spec["expect_change"]
        if "expect_planner_calls" in spec: passed &= planner_calls == spec["expect_planner_calls"]
        if "expect_final_fan" in spec: passed &= abs(rt.state.fan_speed - spec["expect_final_fan"]) < 1e-12
        if "expect_action" in spec: passed &= bool(events) and events[-1].executed_action == spec["expect_action"]
        if "expect_source" in spec: passed &= bool(events) and events[-1].action_source == spec["expect_source"]
        # Independent per-event actuator-contract oracle: bounds and slew.
        for event in events:
            fan_delta = event.state_after["fan_speed"] - event.state_before["fan_speed"]
            setpoint_delta = event.state_after["setpoint_c"] - event.state_before["setpoint_c"]
            passed &= 0 <= event.state_after["fan_speed"] <= 1 and 18 <= event.state_after["setpoint_c"] <= 28
            passed &= fan_delta <= verifier.config["actuators"]["fan_speed"]["max_slew_rate"] + 1e-12
            passed &= -setpoint_delta <= verifier.config["actuators"]["setpoint_c"]["max_slew_rate"] + 1e-12
        rows.append({"case_id": spec["id"], "passed": bool(passed), "event_count": len(events), "planner_calls": planner_calls, "reasons": ";".join(reasons), "observed_state_change": observed_change, "final_fan_speed": rt.state.fan_speed, "final_setpoint_c": rt.state.setpoint_c, "oracle_within_bounds": within_bounds})

    pd.DataFrame(rows).to_csv(args.output / "outcomes.csv", index=False)
    with (args.output / "events.jsonl").open("w") as handle:
        for item in logs: handle.write(json.dumps(item, allow_nan=True) + "\n")
    (args.output / "matrix.json").write_text(args.matrix.read_text())
    provenance = {"run_id": matrix["run_id"], "source_code_commit": actual_commit, "commit_verified_against_git": True, "python": platform.python_version(), "interpreter": sys.executable, "matrix_sha256": sha(args.matrix), "sla_config_sha256": sha(Path("configs/sla.yaml")), "runner_sha256": sha(Path(__file__)), "case_count": len(rows), "event_count": len(logs), "passed": sum(r["passed"] for r in rows), "failed": sum(not r["passed"] for r in rows), "llm_calls": 0, "thermal_pilot_runs": 0}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if provenance["failed"]: raise SystemExit(f"Matrix failures: {provenance['failed']}")


if __name__ == "__main__": main()
