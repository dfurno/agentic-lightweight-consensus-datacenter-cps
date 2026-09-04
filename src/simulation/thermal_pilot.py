from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from src.agents.schemas import ConsensusSnapshot, ControlAction, VerifierDecision
from src.consensus.crp import CRPResult, CRPState, crp_consensus
from src.runtime.control import ControlRuntime, PlannedAction
from src.simulation.actuators import ActuatorState


@dataclass
class ControllerState:
    below_band_streak: int = 0


class ThermalExperimentVerifier:
    """Pilot-only counterfactual progress verifier; historical mode is unchanged."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        control = cfg["control"]
        self.config = {
            "temperature_sla_c": control["sla_temperature_c"],
            "emergency_temperature_c": control["emergency_temperature_c"],
            "min_confidence": cfg["diagnostics"]["confidence_threshold"],
            "max_consensus_age_seconds": cfg["time"]["dt_seconds"] * 2,
            "actuators": {
                "fan_speed": {"min": control["fan_min"], "max": control["fan_max"],
                              "max_slew_rate": control["fan_max_slew_per_step"]},
                "setpoint_c": {"min": cfg["plant"]["fixed_setpoint_c"], "max": cfg["plant"]["fixed_setpoint_c"],
                               "max_slew_rate": 0.0},
            },
        }
        self.ambient_c = float(cfg["plant"]["ambient_base_c"])
        self.load_c_per_step = float(cfg["plant"]["load_base_c_per_step"])
        self.last_prediction: dict[str, float] = {}

    def set_exogenous(self, ambient_c: float, load_c_per_step: float) -> None:
        self.ambient_c = ambient_c
        self.load_c_per_step = load_c_per_step

    def _next(self, temperature: float, fan: float) -> float:
        pred = self.cfg["predictor"]
        return (temperature + pred["thermal_exchange_per_step"] * (self.ambient_c - temperature)
                + self.load_c_per_step - pred["fan_gain_c_per_step"] * fan)

    def verify(self, action: ControlAction, consensus: ConsensusSnapshot,
               actuator_state: ActuatorState | None = None, now: datetime | None = None) -> VerifierDecision:
        state = actuator_state or ActuatorState()
        reasons: list[str] = []
        if not action.used_consensus:
            reasons.append("consensus_not_used")
        if action.consensus_timestamp != consensus.timestamp:
            reasons.append("snapshot_binding_mismatch")
        if action.target_zone != consensus.zone:
            reasons.append("zone_mismatch")
        if consensus.confidence < self.config["min_confidence"]:
            reasons.append("low_confidence")
        candidate_fan = state.fan_speed
        if action.action_type == "increase_fan_speed":
            candidate_fan += action.magnitude
        elif action.action_type == "decrease_fan_speed":
            candidate_fan -= action.magnitude
        elif action.action_type not in {"maintain", "raise_alarm"}:
            reasons.append("unsupported_thermal_action")
        bounds = self.config["actuators"]["fan_speed"]
        if not bounds["min"] <= candidate_fan <= bounds["max"] or action.magnitude > bounds["max_slew_rate"]:
            reasons.append("actuator_contract")
        emergency = consensus.temperature >= self.config["emergency_temperature_c"]
        if emergency and action.action_type != "raise_alarm":
            reasons.append("emergency_requires_alarm")
        if not emergency and action.action_type == "raise_alarm":
            reasons.append("spurious_alarm")
        predicted_candidate = self._next(consensus.temperature, candidate_fan)
        predicted_maintain = self._next(consensus.temperature, state.fan_speed)
        self.last_prediction = {"candidate_next_c": predicted_candidate, "maintain_next_c": predicted_maintain}
        if not emergency:
            sla = self.config["temperature_sla_c"]
            tolerance = float(self.cfg["predictor"]["improvement_tolerance_c"])
            if consensus.temperature <= sla:
                if predicted_candidate > sla:
                    reasons.append("predicted_next_above_sla")
            elif action.action_type == "increase_fan_speed":
                if predicted_candidate > predicted_maintain - tolerance:
                    reasons.append("insufficient_predicted_improvement")
            else:
                reasons.append("above_sla_requires_cooling_progress")
        return VerifierDecision(accepted=not reasons, process_reward=float(not reasons),
                                outcome_reward=float(not reasons), rejection_reasons=reasons,
                                required_refinement=None if not reasons else "Use a bounded action satisfying the pilot rule.")


def supervisory_action(snapshot: ConsensusSnapshot, state: ActuatorState,
                       controller: ControllerState, cfg: dict) -> ControlAction:
    ctl = cfg["control"]
    target, band, step = ctl["target_temperature_c"], ctl["hysteresis_c"], ctl["fan_step"]
    kind, magnitude = "maintain", 0.0
    if snapshot.temperature >= ctl["emergency_temperature_c"]:
        kind = "raise_alarm"
        controller.below_band_streak = 0
    elif snapshot.temperature > target + band and state.fan_speed < ctl["fan_max"] - 1e-12:
        kind, magnitude = "increase_fan_speed", min(step, ctl["fan_max"] - state.fan_speed)
        controller.below_band_streak = 0
    elif snapshot.temperature < target - band:
        controller.below_band_streak += 1
        if controller.below_band_streak >= ctl["safe_dwell_ticks"] and state.fan_speed > ctl["fan_min"] + 1e-12:
            kind, magnitude = "decrease_fan_speed", min(step, state.fan_speed - ctl["fan_min"])
            controller.below_band_streak = 0
    else:
        controller.below_band_streak = 0
    return ControlAction(action_type=kind, target_zone="zone-0", magnitude=float(magnitude), duration=1,
                         reasoning_summary="Frozen bidirectional thermal pilot controller",
                         consensus_timestamp=snapshot.timestamp, used_consensus=True)


def exogenous_arrays(seed: int, scenario_index: int, cfg: dict) -> dict[str, np.ndarray]:
    n, sensors = cfg["time"]["horizon_ticks"], cfg["sensors"]["count"]
    rng = np.random.default_rng(seed * 1000 + scenario_index)
    process = rng.normal(0, cfg["plant"]["process_noise_std_c"], n)
    sensor = rng.normal(0, cfg["sensors"]["noise_std_c"], (n, sensors))
    attacked = np.sort(rng.choice(sensors, max(1, round(sensors * cfg["sensors"]["attack_fraction"])), replace=False))
    return {"process_noise": process, "sensor_noise": sensor, "attacked": attacked}


def _attack(values: np.ndarray, history: list[np.ndarray], kind: str, intensity: float,
            tick: int, cfg: dict, attacked: np.ndarray) -> np.ndarray:
    out = values.copy(); start = cfg["attacks"]["start_tick"]; end = cfg["attacks"]["end_tick_exclusive"]
    if kind == "nominal" or not start <= tick < end:
        return out
    elapsed = tick - start
    if kind == "bias": out[attacked] += intensity
    elif kind == "drift": out[attacked] += intensity * (elapsed + 1)
    elif kind == "replay": out[attacked] = history[max(0, tick - int(intensity))][attacked]
    elif kind == "freeze":
        if elapsed < int(intensity): out[attacked] = history[start - 1][attacked]
    elif kind == "spike": out[attacked] += intensity * (1 if elapsed % 2 == 0 else -1)
    elif kind == "scaling": out[attacked] *= intensity
    elif kind == "stuck": out[attacked] = intensity
    else: raise ValueError(f"Unknown attack {kind}")
    return out


def _baseline_diagnostics(values: np.ndarray, cfg: dict) -> tuple[float, int]:
    flags = np.abs(values - np.median(values)) > cfg["diagnostics"]["baseline_deviation_threshold_c"]
    return float(1.0 - np.mean(flags)), int(flags.sum())


def run_condition(seed: int, scenario_index: int, attack: str, intensity_label: str | None,
                  condition: str, cfg: dict, arrays: dict[str, np.ndarray], *,
                  initial_temperature_c: float | None = None,
                  common_bias_c: float | None = None,
                  common_bias_end_tick: int = 0) -> tuple[list[dict], dict]:
    intensity = 0.0 if attack == "nominal" else float(cfg["attacks"]["definitions"][attack][intensity_label])
    verifier = ThermalExperimentVerifier(cfg)
    runtime = ControlRuntime(verifier, policy={"always_supervise": True, "confidence_threshold": -1.0,
                                               "anomaly_threshold": 10**9, "temperature_margin": -100.0})
    runtime.state.fan_speed = cfg["control"]["fan_initial"]
    runtime.state.setpoint_c = cfg["plant"]["fixed_setpoint_c"]
    controller = ControllerState(); crp_state = CRPState(lam=cfg["crp"]["ewma_lambda"])
    temperature = float(cfg["plant"]["initial_temperature_c"] if initial_temperature_c is None else initial_temperature_c); raw_history: list[np.ndarray] = []
    rows: list[dict] = []; crp_nonconverged = 0; crp_rounds = 0
    start_tick, end_tick = cfg["attacks"]["start_tick"], cfg["attacks"]["end_tick_exclusive"]
    for tick in range(cfg["time"]["horizon_ticks"]):
        transition = cfg["plant"]["transition_start_tick"] <= tick < cfg["plant"]["transition_end_tick"]
        ambient = cfg["plant"]["ambient_base_c"] + (cfg["plant"]["ambient_transition_c"] if transition else 0.0)
        load = cfg["plant"]["load_base_c_per_step"] + (cfg["plant"]["load_transition_c_per_step"] if transition else 0.0)
        clean = temperature + arrays["sensor_noise"][tick]
        raw_history.append(clean.copy())
        observed = _attack(clean, raw_history, attack, intensity, tick, cfg, arrays["attacked"])
        effective_mask = np.zeros(cfg["sensors"]["count"], dtype=bool)
        if attack != "nominal" and start_tick <= tick < end_tick:
            effective_mask[arrays["attacked"]] = not (attack == "freeze" and tick - start_tick >= int(intensity))
        if common_bias_c is not None and tick < common_bias_end_tick:
            observed = observed + common_bias_c
            effective_mask[:] = True
        crp_latency = None; crp: CRPResult | None = None
        if condition == "fpr_crp":
            crp_start = time.perf_counter()
            crp = crp_consensus(observed, crp_state, alpha=cfg["crp"]["alpha"], beta=cfg["crp"]["beta"],
                                alpha_crp=cfg["crp"]["alpha_crp"], m_z=cfg["crp"]["m_z_c"],
                                tau_c=cfg["crp"]["tau_c"], tau_r=cfg["crp"]["tau_r"],
                                persistence_l=cfg["crp"]["persistence_l"], max_rounds=cfg["crp"]["max_rounds"],
                                min_sensors=cfg["crp"]["min_sensors"])
            crp_latency = time.perf_counter() - crp_start
            estimate, confidence = crp.aggregated_value, crp.confidence
            anomaly_count = int(crp.persistent_flags.sum())
            crp_nonconverged += int(not crp.converged); crp_rounds += crp.rounds
        else:
            confidence, anomaly_count = _baseline_diagnostics(observed, cfg)
            estimate = temperature if condition == "oracle" else float(np.mean(observed))
        timestamp = (datetime(2026, 9, 3, tzinfo=timezone.utc) + timedelta(seconds=tick * cfg["time"]["dt_seconds"])).isoformat()
        snap = ConsensusSnapshot(temperature=estimate, confidence=confidence, anomaly_count=anomaly_count,
                                 timestamp=timestamp, zone="zone-0")
        verifier.set_exogenous(ambient, load)
        event = runtime.step(snap, monotonic_now=float(tick),
                             wall_now=datetime.fromisoformat(timestamp),
                             planner=lambda s: PlannedAction(supervisory_action(s, runtime.state, controller, cfg)))
        fan = runtime.state.fan_speed
        next_temperature = (temperature + cfg["plant"]["thermal_exchange_per_step"] * (ambient - temperature)
                            + load - cfg["plant"]["fan_gain_c_per_step"] * fan + arrays["process_noise"][tick])
        row = {"seed": seed, "scenario_index": scenario_index, "attack": attack,
               "intensity": intensity_label or "none", "condition": condition, "tick": tick,
               "true_temperature_c": temperature, "estimate_c": estimate, "estimator_error_c": estimate - temperature,
               "ambient_c": ambient, "load_c_per_step": load, "fan_before": event.state_before["fan_speed"],
               "fan_after": fan, "executed_action": event.executed_action, "action_source": event.action_source,
               "verifier_accepted": event.verifier_accepted, "rejection_reasons": ";".join(event.verifier_rejection_reasons),
               "fallback": event.fallback or "", "anomaly_count": anomaly_count, "confidence": confidence,
               "predicted_candidate_next_c": verifier.last_prediction.get("candidate_next_c"),
               "predicted_maintain_next_c": verifier.last_prediction.get("maintain_next_c"),
               "next_true_temperature_c": next_temperature, "crp_rounds": crp.rounds if crp else 0,
               "crp_converged": crp.converged if crp else None, "crp_excluded": int(crp.excluded.sum()) if crp else 0,
               "crp_latency_seconds": crp_latency, "runtime_step_latency_seconds": event.step_latency_seconds,
               "attack_active": bool(effective_mask.any()),
               "clean_sensor_values": json.dumps(clean.tolist()),
               "observed_sensor_values": json.dumps(observed.tolist()),
               "effective_attack_mask": json.dumps(effective_mask.tolist()),
               "crp_persistent_flags": json.dumps(crp.persistent_flags.tolist()) if crp else "[]",
               "crp_excluded_flags": json.dumps(crp.excluded.tolist()) if crp else "[]",
               "runtime_snapshot_id": event.snapshot_id, "runtime_proposal_id": event.proposal_id,
               "runtime_snapshot_payload": json.dumps(event.snapshot_payload, sort_keys=True),
               "runtime_proposal_payload": json.dumps(event.proposal_payload, sort_keys=True)}
        rows.append(row); temperature = next_temperature
        if not math.isfinite(temperature): raise RuntimeError("nonfinite_plant_state")
    ctl = cfg["control"]; dt_min = cfg["time"]["dt_seconds"] / 60.0
    temps = np.array([r["true_temperature_c"] for r in rows]); fans = np.array([r["fan_after"] for r in rows])
    excess = np.maximum(temps - ctl["sla_temperature_c"], 0)
    post = np.abs(temps[end_tick:] - ctl["target_temperature_c"]) <= ctl["recovery_band_c"]
    sustain = ctl["recovery_sustain_ticks"]; recovery = None
    for i in range(max(0, len(post) - sustain + 1)):
        if post[i:i+sustain].all(): recovery = i * dt_min; break
    summary = {"seed": seed, "scenario_index": scenario_index, "attack": attack,
               "intensity": intensity_label or "none", "condition": condition, "ticks": len(rows),
               "iae_degC_min": float(np.abs(temps - ctl["target_temperature_c"]).sum() * dt_min),
               "duration_above_sla_min": float((excess > 0).sum() * dt_min),
               "peak_excess_degC": float(excess.max()), "integrated_excess_degC_min": float(excess.sum() * dt_min),
               "recovered": recovery is not None, "recovery_time_min": recovery,
               "fan_effort_normalized_min": float(fans.sum() * dt_min),
               "fan_total_variation": float(np.abs(np.diff(np.r_[cfg["control"]["fan_initial"], fans])).sum()),
               "estimator_mae_c": float(np.mean(np.abs([r["estimator_error_c"] for r in rows]))),
               "verifier_accepts": sum(r["verifier_accepted"] is True for r in rows),
               "verifier_rejects": sum(r["verifier_accepted"] is False for r in rows),
               "fallback_ticks": sum(bool(r["fallback"]) for r in rows),
               "crp_nonconverged_ticks": crp_nonconverged, "crp_total_rounds": crp_rounds,
               "saturation_ticks": int(((fans <= ctl["fan_min"] + 1e-12) | (fans >= ctl["fan_max"] - 1e-12)).sum())}
    return rows, summary
