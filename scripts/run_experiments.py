#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.data.real_trace import RealDatasetUnavailable
from src.agents.react_planner import ReactPlanner
from src.agents.schemas import ConsensusSnapshot
from src.agents.tools import consensus_to_snapshot
from src.agents.verifier import VerifierAgent
from src.consensus.baselines import run_baseline
from src.consensus.fpr_owa import fpr_owa_consensus
from src.evaluation.metrics import error_summary
from src.evaluation.plots import save_basic_figures
from src.simulation.digital_twin import build_digital_twin_from_available_data
from src.simulation.policies import deterministic_policy
from src.runtime.control import ControlRuntime, PlannedAction
from src.utils.io import ensure_dir, read_yaml, write_yaml


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _checkpoint_paths(results: Path, scenario: str) -> dict[str, Path]:
    base = results / "checkpoints"
    return {
        "metrics": base / "metrics" / f"{scenario}.csv",
        "safety": base / "safety" / f"{scenario}.csv",
        "latency": base / "latency" / f"{scenario}.csv",
        "done": base / "done" / f"{scenario}.done",
    }


def _scenario_checkpoint_complete(results: Path, scenario: str) -> bool:
    paths = _checkpoint_paths(results, scenario)
    return all(paths[key].exists() and paths[key].stat().st_size > 0 for key in ("metrics", "safety", "latency", "done"))


def _read_checkpoint_dir(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True)


def _aggregate_checkpoints(results: Path, config: dict) -> None:
    checkpoint_root = results / "checkpoints"
    metrics = _read_checkpoint_dir(checkpoint_root / "metrics")
    safety = _read_checkpoint_dir(checkpoint_root / "safety")
    latency = _read_checkpoint_dir(checkpoint_root / "latency")
    metrics.to_csv(results / "metrics.csv", index=False)
    if not metrics.empty:
        metrics.groupby(["attack_type", "attack_ratio", "method"], as_index=False).mean(numeric_only=True).to_csv(
            results / "metrics_by_scenario.csv", index=False
        )
        save_basic_figures(metrics)
    else:
        pd.DataFrame(columns=["attack_type", "attack_ratio", "method"]).to_csv(results / "metrics_by_scenario.csv", index=False)
    safety.to_csv(results / "safety_metrics.csv", index=False)
    latency.to_csv(results / "latency_metrics.csv", index=False)
    token_cols = ["scenario", "agent_variant", "llm_calls", "verifier_calls"]
    if all(col in safety.columns for col in token_cols):
        safety[token_cols].to_csv(results / "token_usage.csv", index=False)
    else:
        pd.DataFrame(columns=token_cols).to_csv(results / "token_usage.csv", index=False)


def _sensor_subset(sensor_cols: list[str], count: int) -> list[str]:
    if count >= len(sensor_cols):
        return sensor_cols
    return sensor_cols[:count]


def _add_measurement_noise(frame: pd.DataFrame, sensor_cols: list[str], noise_std: float, seed: int) -> pd.DataFrame:
    if noise_std <= 0:
        return frame
    noisy = frame.copy()
    rng = np.random.default_rng(seed)
    for col in sensor_cols:
        noisy[col] = noisy[col].astype(float) + rng.normal(0.0, noise_std, len(noisy))
    return noisy


def _scenario_suffix(seed: int, attack_type: str, attack_ratio: float, noise: float, sensors: int, alpha: float, beta: float) -> str:
    return f"seed{seed}_{attack_type}_{attack_ratio}_noise{noise}_s{sensors}_a{alpha}_b{beta}".replace(".", "p")


def _scenario_grid(suite: dict, sim_base: dict) -> Iterable[tuple[int, float, str, float, int, float, float]]:
    for seed in suite["seeds"]:
        for attack_ratio in suite["attack_ratios"]:
            for attack_type in suite["attack_types"]:
                for noise in suite.get("sensor_noise_levels", [sim_base.get("sensor_noise_std", 0.0)]):
                    for sensor_count in suite.get("sensors_per_group", [sim_base.get("num_sensors", 9)]):
                        for params in suite.get("owa_parameters", [{"alpha": 0.3, "beta": 0.8}]):
                            yield (
                                int(seed),
                                float(attack_ratio),
                                str(attack_type),
                                float(noise),
                                int(sensor_count),
                                float(params["alpha"]),
                                float(params["beta"]),
                            )


def evaluate_agent_variant(
    variant: str,
    consensus: ConsensusSnapshot,
    planner: ReactPlanner,
    verifier: VerifierAgent,
    max_refinement_cycles: int,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    proposed = 0
    blocked = 0
    executed = 0
    accepted = 0
    alarms = 0
    refinement_cycles = 0
    verifier_calls = 0
    if variant == "deterministic_policy_only":
        action = deterministic_policy(consensus.temperature, verifier.config["temperature_sla_c"])
        proposed = 1
        executed = 1
        accepted = 1
        alarms = int(action.action_type == "raise_alarm")
    elif variant == "react_without_verifier":
        action = planner.propose(consensus, variant=variant)
        proposed = 1
        executed = 1
        alarms = int(action.action_type == "raise_alarm")
        # Unsafe execution is defined deterministically even when verifier is bypassed.
        shadow = verifier.verify(action, consensus)
        verifier_calls = 1
        executed = 1
        accepted = int(shadow.accepted)
        blocked = 0
    else:
        action = planner.propose(consensus, variant=variant)
        proposed = 1
        decision = verifier.verify(action, consensus)
        verifier_calls = 1
        while not decision.accepted and variant == "react_with_verifier_and_self_refinement" and refinement_cycles < max_refinement_cycles:
            blocked += 1
            refinement_cycles += 1
            action = planner.propose(
                consensus,
                variant=variant,
                refinement_cycle=refinement_cycles,
                verifier_feedback=decision.rejection_reasons,
            )
            decision = verifier.verify(action, consensus)
            verifier_calls += 1
            proposed += 1
        if decision.accepted:
            accepted = 1
            executed = 1
            alarms = int(action.action_type == "raise_alarm")
        else:
            blocked += 1
    elapsed = time.perf_counter() - start
    unsafe_executed = int(variant == "react_without_verifier" and accepted == 0 and executed == 1)
    return {
        "agent_variant": variant,
        "unsafe_actions_proposed": int(proposed - accepted),
        "unsafe_actions_blocked_by_verifier": int(blocked),
        "unsafe_actions_executed": unsafe_executed,
        "safe_actions_accepted": int(accepted),
        "emergency_alarms_raised": int(alarms),
        "average_refinement_cycles": float(refinement_cycles),
        "planner_verifier_latency_seconds": elapsed,
        "llm_calls": int(proposed if variant != "deterministic_policy_only" else 0),
        "verifier_calls": int(verifier_calls),
    }


def _event_triggered_llm_required(consensus: ConsensusSnapshot, suite: dict, verifier: VerifierAgent) -> bool:
    policy = suite.get("llm_policy", {})
    confidence_threshold = float(policy.get("confidence_threshold", 0.55))
    anomaly_threshold = int(policy.get("anomaly_threshold", 1))
    temperature_margin = float(policy.get("temperature_margin", 0.10))
    temperature_sla = float(verifier.config["temperature_sla_c"])
    return (
        consensus.confidence < confidence_threshold
        or consensus.anomaly_count >= anomaly_threshold
        or consensus.temperature >= temperature_sla - temperature_margin
    )


def run_realtime_suite(data_mode: str, resume: bool = False) -> None:
    results = ensure_dir("results")
    ensure_dir(results / "traces")
    config = read_yaml("configs/experiments.yaml")
    sim_base = read_yaml("configs/simulation.yaml")
    suite = config["realtime"]
    verifier = VerifierAgent.from_yaml()
    planner = ReactPlanner.from_yaml()
    realtime_rows = []
    realtime_safety_rows = []
    grid = list(_scenario_grid(suite, sim_base))
    for index, (seed, attack_ratio, attack_type, noise, sensor_count, alpha, beta) in enumerate(grid, start=1):
        scenario = f"realtime_{_scenario_suffix(seed, attack_type, attack_ratio, noise, sensor_count, alpha, beta)}"
        trace_path = results / "traces" / f"{scenario}.csv"
        metadata_path = results / "traces" / f"{scenario}.metadata.json"
        if resume and trace_path.exists() and trace_path.stat().st_size > 0:
            print(f"[realtime resume] reusing trace {index}/{len(grid)}: {scenario}", flush=True)
            frame = pd.read_csv(trace_path)
        else:
            print(f"[realtime] building scenario {index}/{len(grid)}: {scenario}", flush=True)
            sim_config = dict(sim_base)
            sim_config["seed"] = seed
            sim_config["num_sensors"] = sensor_count
            sim_config["sensor_noise_std"] = noise
            sim_config["attacks"] = dict(sim_base["attacks"])
            sim_config["attacks"]["attack_ratio"] = attack_ratio
            sim_config["attacks"]["attack_type"] = attack_type
            try:
                trace = build_digital_twin_from_available_data(sim_config, data_mode=data_mode)
            except RealDatasetUnavailable as exc:
                raise SystemExit(f"Real dataset mode requested but unavailable: {exc}") from exc
            sensor_cols = _sensor_subset([col for col in trace.frame.columns if col.startswith("sensor_")], sensor_count)
            frame = _add_measurement_noise(trace.frame, sensor_cols, noise, seed)
            frame.to_csv(trace_path, index=False)
            pd.Series(trace.metadata).to_json(metadata_path, indent=2)

        sensor_cols = _sensor_subset([col for col in frame.columns if col.startswith("sensor_")], sensor_count)
        loop_latencies = []
        llm_latencies = []
        llm_calls = 0
        llm_failures = 0
        verifier_calls = 0
        unsafe_proposed = 0
        unsafe_blocked = 0
        unsafe_executed = 0
        safe_accepted = 0
        deterministic_safe = 0
        sla_violations = 0
        predictions = []
        truth = frame["temperature_ground_truth"].to_numpy()
        cooldown_ticks = int(suite.get("llm_policy", {}).get("min_llm_interval_ticks", 0))
        last_llm_tick = -cooldown_ticks - 1
        runtime = ControlRuntime(verifier, policy=suite.get("llm_policy", {}))
        for tick_index, (_, row) in enumerate(frame.iterrows()):
            values = row[sensor_cols].to_numpy(dtype=float)
            loop_start = time.perf_counter()
            consensus_timestamp = pd.Timestamp.now("UTC").isoformat()
            consensus_result = fpr_owa_consensus(values, alpha=alpha, beta=beta, timestamp=consensus_timestamp)
            consensus = consensus_to_snapshot(consensus_result)
            event = runtime.step(
                consensus,
                monotonic_now=loop_start,
                wall_now=datetime.now(timezone.utc),
                planner=lambda snap: PlannedAction(planner.propose(snap, variant="event_triggered_supervisor")),
            )
            verifier_calls += int(event.verifier_accepted is not None)
            deterministic_safe += int(event.action_source == "fast_path" and event.verifier_accepted is True)
            if event.planner_status in {"ready", "exception"}:
                llm_calls += 1
                last_llm_tick = tick_index
                llm_latencies.append(event.planner_latency_seconds or 0.0)
                llm_failures += int(event.planner_status == "exception")
            safe_accepted += int(event.decision == "supervisor" and event.verifier_accepted is True and event.action_source == "supervisor")
            unsafe_proposed += int(event.decision == "supervisor" and event.verifier_accepted is False)
            unsafe_blocked += int(event.decision == "supervisor" and event.verifier_accepted is False)
            loop_latencies.append(time.perf_counter() - loop_start)
            predictions.append(consensus_result.aggregated_value)
            sla_violations += int(consensus.temperature > float(verifier.config["temperature_sla_c"]))
        loop_lat = np.asarray(loop_latencies)
        llm_lat = np.asarray(llm_latencies) if llm_latencies else np.asarray([0.0])
        summary = error_summary(np.asarray(predictions), truth)
        base_row = {
            "scenario": scenario,
            "seed": seed,
            "attack_type": attack_type,
            "attack_ratio": attack_ratio,
            "sensor_noise_level": noise,
            "sensors_per_group": sensor_count,
            "alpha": alpha,
            "beta": beta,
            "ticks": int(len(frame)),
            "llm_calls": int(llm_calls),
            "llm_failures": int(llm_failures),
            "llm_invocation_rate": float(llm_calls / max(len(frame), 1)),
            "llm_failure_rate": float(llm_failures / max(llm_calls, 1)),
            "verifier_calls": int(verifier_calls),
            "mean_loop_latency_ms": float(loop_lat.mean() * 1000.0),
            "p95_loop_latency_ms": float(np.quantile(loop_lat, 0.95) * 1000.0),
            "p99_loop_latency_ms": float(np.quantile(loop_lat, 0.99) * 1000.0),
            "mean_llm_latency_seconds": float(llm_lat.mean()),
            "p95_llm_latency_seconds": float(np.quantile(llm_lat, 0.95)),
            "unsafe_actions_proposed": int(unsafe_proposed),
            "unsafe_actions_blocked_by_verifier": int(unsafe_blocked),
            "unsafe_actions_executed": float("nan"),
            "unsafe_actions_executed_measurement": "not_measured_thermal_safety",
            "safe_actions_accepted": int(safe_accepted),
            "deterministic_safe_actions": int(deterministic_safe),
            "thermal_sla_violation_rate": float(sla_violations / max(len(frame), 1)),
            **summary,
        }
        for deadline_ms in suite.get("deadlines_ms", [100, 250, 500, 1000, 5000]):
            row_out = dict(base_row)
            row_out["deadline_ms"] = int(deadline_ms)
            row_out["deadline_miss_rate"] = float(np.mean(loop_lat * 1000.0 > int(deadline_ms)))
            realtime_rows.append(row_out)
        realtime_safety_rows.append(base_row)
        _atomic_write_csv(pd.DataFrame(realtime_rows), results / "realtime_metrics.csv")
        _atomic_write_csv(pd.DataFrame(realtime_safety_rows), results / "realtime_safety_metrics.csv")
    write_yaml(results / "realtime_config_snapshot.yaml", {"simulation": sim_base, "experiments": suite, "data_mode": data_mode, "resume": resume})


def run_suite(size: str, data_mode: str, resume: bool = False) -> None:
    results = ensure_dir("results")
    ensure_dir(results / "traces")
    config = read_yaml("configs/experiments.yaml")
    sim_base = read_yaml("configs/simulation.yaml")
    suite = config[size]
    verifier = VerifierAgent.from_yaml()
    planner = ReactPlanner.from_yaml()
    grid = list(_scenario_grid(suite, sim_base))
    for index, (seed, attack_ratio, attack_type, noise, sensor_count, alpha, beta) in enumerate(grid, start=1):
        scenario = f"{size}_{_scenario_suffix(seed, attack_type, attack_ratio, noise, sensor_count, alpha, beta)}"
        if resume and _scenario_checkpoint_complete(results, scenario):
            print(f"[resume] skipping completed scenario {index}/{len(grid)}: {scenario}", flush=True)
            continue
        trace_path = results / "traces" / f"{scenario}.csv"
        metadata_path = results / "traces" / f"{scenario}.metadata.json"
        if resume and trace_path.exists() and trace_path.stat().st_size > 0:
            print(f"[resume] reusing existing trace {index}/{len(grid)}: {scenario}", flush=True)
            frame = pd.read_csv(trace_path)
        else:
            print(f"[run] building scenario {index}/{len(grid)}: {scenario}", flush=True)
            sim_config = dict(sim_base)
            sim_config["seed"] = seed
            sim_config["num_sensors"] = sensor_count
            sim_config["sensor_noise_std"] = noise
            sim_config["attacks"] = dict(sim_base["attacks"])
            sim_config["attacks"]["attack_ratio"] = attack_ratio
            sim_config["attacks"]["attack_type"] = attack_type
            try:
                trace = build_digital_twin_from_available_data(sim_config, data_mode=data_mode)
            except RealDatasetUnavailable as exc:
                raise SystemExit(
                    "Real dataset mode requested, but required materialized dataset files are unavailable.\n"
                    f"Reason: {exc}\n"
                    "Run: ALLOW_DATASET_DOWNLOADS=true make real-paper-pipeline\n"
                    "If BDG2 was cloned but files are Git LFS pointers, run git lfs pull in "
                    "data/raw/kaggle_hot_corridor/building_data_genome_2."
                ) from exc
            all_sensor_cols = [col for col in trace.frame.columns if col.startswith("sensor_")]
            sensor_cols = _sensor_subset(all_sensor_cols, sensor_count)
            frame = _add_measurement_noise(trace.frame, sensor_cols, noise, seed)
            frame.to_csv(trace_path, index=False)
            pd.Series(trace.metadata).to_json(metadata_path, indent=2)
        all_sensor_cols = [col for col in frame.columns if col.startswith("sensor_")]
        sensor_cols = _sensor_subset(all_sensor_cols, sensor_count)
        truth = frame["temperature_ground_truth"].to_numpy()
        rows = []
        safety_rows = []
        latency_rows = []
        for method in config["methods"]:
            predictions = []
            anomaly_flags = []
            confidences = []
            start = time.perf_counter()
            for _, row in frame.iterrows():
                values = row[sensor_cols].to_numpy(dtype=float)
                if method == "fpr_owa":
                    result = fpr_owa_consensus(values, alpha=alpha, beta=beta, timestamp=str(row["timestamp"]))
                    predictions.append(result.aggregated_value)
                    anomaly_flags.append(int(result.anomaly_flags.sum()))
                    confidences.append(result.confidence)
                else:
                    predictions.append(run_baseline(method, values, alpha=alpha, beta=beta))
                    anomaly_flags.append(0)
                    confidences.append(1.0)
            elapsed = time.perf_counter() - start
            summary = error_summary(np.asarray(predictions), truth)
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "attack_type": attack_type,
                    "attack_ratio": attack_ratio,
                    "sensor_noise_level": noise,
                    "sensors_per_group": sensor_count,
                    "alpha": alpha,
                    "beta": beta,
                    "method": method,
                    **summary,
                    "mean_anomaly_flags": float(np.mean(anomaly_flags)),
                    "mean_confidence": float(np.mean(confidences)),
                }
            )
            latency_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "fuzzy_consensus_latency_seconds": elapsed,
                    "deterministic_simulation_latency_seconds": 0.0,
                }
            )
        last_values = frame.iloc[-1][sensor_cols].to_numpy(dtype=float)
        consensus = consensus_to_snapshot(
            fpr_owa_consensus(last_values, alpha=alpha, beta=beta, timestamp=pd.Timestamp.now("UTC").isoformat())
        )
        for max_cycles in suite.get("max_refinement_cycles", [0]):
            for variant in config.get("agent_variants", []):
                row = evaluate_agent_variant(variant, consensus, planner, verifier, int(max_cycles))
                safety_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "attack_type": attack_type,
                        "attack_ratio": attack_ratio,
                        "sensor_noise_level": noise,
                        "sensors_per_group": sensor_count,
                        "alpha": alpha,
                        "beta": beta,
                        "max_refinement_cycles": int(max_cycles),
                        **row,
                    }
                )
                latency_rows.append(
                    {
                        "scenario": scenario,
                        "method": f"agent_{variant}",
                        "fuzzy_consensus_latency_seconds": 0.0,
                        "deterministic_simulation_latency_seconds": row["planner_verifier_latency_seconds"],
                    }
                )
        paths = _checkpoint_paths(results, scenario)
        _atomic_write_csv(pd.DataFrame(rows), paths["metrics"])
        _atomic_write_csv(pd.DataFrame(safety_rows), paths["safety"])
        _atomic_write_csv(pd.DataFrame(latency_rows), paths["latency"])
        _atomic_write_text("complete\n", paths["done"])
        print(f"[checkpoint] completed scenario {index}/{len(grid)}: {scenario}", flush=True)
        _aggregate_checkpoints(results, config)
    _aggregate_checkpoints(results, config)
    write_yaml(results / "config_snapshot.yaml", {"simulation": sim_base, "experiments": config, "data_mode": data_mode, "resume": resume})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "full", "realtime"], default="small")
    parser.add_argument(
        "--data-mode",
        choices=["auto", "real", "synthetic"],
        default="auto",
        help="auto uses real materialized datasets when available, real fails if unavailable, synthetic always uses fallback.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing traces and skip scenarios with complete checkpoints.")
    args = parser.parse_args()
    if args.size == "realtime":
        run_realtime_suite(args.data_mode, resume=args.resume)
    else:
        run_suite(args.size, args.data_mode, resume=args.resume)


if __name__ == "__main__":
    main()
