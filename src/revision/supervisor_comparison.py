from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.schemas import ConsensusSnapshot, ControlAction
from src.agents.tools import consensus_to_snapshot
from src.agents.verifier import VerifierAgent
from src.consensus.fpr_owa import fpr_owa_consensus
from src.evaluation.metrics import error_summary
from src.revision.artifacts import scenario_parts, trace_files, write_text
from src.simulation.policies import deterministic_policy
from src.utils.io import ensure_dir, read_yaml


def _sensor_subset(sensor_cols: list[str], count: int) -> list[str]:
    if count >= len(sensor_cols):
        return sensor_cols
    return sensor_cols[:count]


def _event_triggered_supervisor_required(consensus: ConsensusSnapshot, policy: dict, verifier: VerifierAgent) -> bool:
    confidence_threshold = float(policy.get("confidence_threshold", 0.35))
    anomaly_threshold = int(policy.get("anomaly_threshold", 3))
    temperature_margin = float(policy.get("temperature_margin", 0.02))
    temperature_sla = float(verifier.config["temperature_sla_c"])
    return (
        consensus.confidence < confidence_threshold
        or consensus.anomaly_count >= anomaly_threshold
        or consensus.temperature >= temperature_sla - temperature_margin
    )


def rule_based_supervisory_action(consensus: ConsensusSnapshot, verifier: VerifierAgent, target_zone: str = "zone-0") -> ControlAction:
    """Deterministic supervisor using the same information exposed to the LLM path.

    The policy is intentionally simple and bounded. It does not bypass the verifier;
    every action is still accepted or rejected by the same VerifierAgent.
    """
    temperature_sla = float(verifier.config["temperature_sla_c"])
    emergency = float(verifier.config["emergency_temperature_c"])
    fan_cfg = verifier.config["actuators"]["fan_speed"]
    setpoint_cfg = verifier.config["actuators"]["setpoint_c"]
    fan_slew = float(fan_cfg["max_slew_rate"])
    setpoint_slew = float(setpoint_cfg["max_slew_rate"])
    cooling_gain = float(verifier.config.get("prediction", {}).get("cooling_gain_per_fan_unit", 3.0))
    setpoint_gain = float(verifier.config.get("prediction", {}).get("setpoint_gain", 0.5))

    if consensus.temperature >= emergency:
        return ControlAction(
            action_type="raise_alarm",
            target_zone=target_zone,
            magnitude=0.0,
            duration=60,
            reasoning_summary="Rule supervisor: emergency temperature requires alarm.",
            consensus_timestamp=consensus.timestamp,
            used_consensus=True,
        )

    thermal_excess = max(0.0, consensus.temperature - temperature_sla)
    if thermal_excess > 0.0:
        required_fan = min(fan_slew, max(0.05, thermal_excess / max(cooling_gain, 1e-9)))
        return ControlAction(
            action_type="increase_fan_speed",
            target_zone=target_zone,
            magnitude=float(required_fan),
            duration=60,
            reasoning_summary="Rule supervisor: bounded fan increase for thermal SLA margin.",
            consensus_timestamp=consensus.timestamp,
            used_consensus=True,
        )

    # For anomaly/low-confidence triggers below SLA, use a conservative low-magnitude
    # cooling response only when anomaly evidence is strong; otherwise maintain.
    if consensus.anomaly_count >= 3:
        return ControlAction(
            action_type="increase_fan_speed",
            target_zone=target_zone,
            magnitude=float(min(fan_slew, 0.05)),
            duration=60,
            reasoning_summary="Rule supervisor: conservative cooling under anomaly burst.",
            consensus_timestamp=consensus.timestamp,
            used_consensus=True,
        )

    if consensus.temperature >= temperature_sla - 0.02:
        required_setpoint = min(setpoint_slew, max(0.1, (temperature_sla - consensus.temperature + 0.02) / max(setpoint_gain, 1e-9)))
        return ControlAction(
            action_type="decrease_setpoint",
            target_zone=target_zone,
            magnitude=float(required_setpoint),
            duration=60,
            reasoning_summary="Rule supervisor: small setpoint decrease near SLA margin.",
            consensus_timestamp=consensus.timestamp,
            used_consensus=True,
        )

    return ControlAction(
        action_type="maintain",
        target_zone=target_zone,
        magnitude=0.0,
        duration=60,
        reasoning_summary="Rule supervisor: no supervisory actuation required.",
        consensus_timestamp=consensus.timestamp,
        used_consensus=True,
    )


def run_rule_supervisor(
    results_dir: Path = Path("results"),
    outputs_dir: Path = Path("outputs"),
    max_traces: int | None = None,
) -> pd.DataFrame:
    outputs = ensure_dir(outputs_dir)
    config = read_yaml("configs/experiments.yaml")
    suite = config["realtime"]
    policy = suite.get("llm_policy", {})
    deadlines = suite.get("deadlines_ms", [100, 250, 500, 1000, 5000])
    verifier = VerifierAgent.from_yaml()
    files = [path for path in trace_files(results_dir) if path.name.startswith("realtime_")]
    if max_traces is not None:
        files = files[:max_traces]
    rows: list[dict[str, object]] = []
    safety_rows: list[dict[str, object]] = []
    for trace_path in files:
        scenario = trace_path.stem
        parts = scenario_parts(scenario)
        frame = pd.read_csv(trace_path)
        sensor_count = int(parts.get("sensors_per_group", len([col for col in frame.columns if col.startswith("sensor_")])))
        alpha = float(parts.get("alpha", 0.3))
        beta = float(parts.get("beta", 0.8))
        sensor_cols = _sensor_subset([col for col in frame.columns if col.startswith("sensor_")], sensor_count)
        truth = frame["temperature_ground_truth"].to_numpy(dtype=float)
        cooldown_ticks = int(policy.get("min_llm_interval_ticks", 0))
        last_supervisor_tick = -cooldown_ticks - 1
        loop_latencies: list[float] = []
        supervisor_latencies: list[float] = []
        supervisor_calls = 0
        verifier_calls = 0
        unsafe_proposed = 0
        unsafe_blocked = 0
        unsafe_executed = 0
        safe_accepted = 0
        deterministic_safe = 0
        fallback_count = 0
        thermal_sla_violations = 0
        predictions: list[float] = []
        accepted_action_types: list[str] = []
        blocked_action_types: list[str] = []
        for tick_index, (_, row) in enumerate(frame.iterrows()):
            values = row[sensor_cols].to_numpy(dtype=float)
            loop_start = time.perf_counter()
            consensus_timestamp = pd.Timestamp.now("UTC").isoformat()
            result = fpr_owa_consensus(values, alpha=alpha, beta=beta, timestamp=consensus_timestamp)
            consensus = consensus_to_snapshot(result)
            deterministic_action = deterministic_policy(consensus.temperature, verifier.config["temperature_sla_c"])
            deterministic_action.consensus_timestamp = consensus.timestamp
            deterministic_decision = verifier.verify(deterministic_action, consensus, now=datetime.now(timezone.utc))
            verifier_calls += 1
            deterministic_safe += int(deterministic_decision.accepted)
            eligible = tick_index - last_supervisor_tick >= cooldown_ticks
            if eligible and _event_triggered_supervisor_required(consensus, policy, verifier):
                supervisor_start = time.perf_counter()
                supervisor_calls += 1
                last_supervisor_tick = tick_index
                action = rule_based_supervisory_action(consensus, verifier)
                supervisor_latencies.append(time.perf_counter() - supervisor_start)
                decision = verifier.verify(action, consensus, now=datetime.now(timezone.utc))
                verifier_calls += 1
                if decision.accepted:
                    safe_accepted += 1
                    accepted_action_types.append(action.action_type)
                else:
                    unsafe_proposed += 1
                    unsafe_blocked += 1
                    fallback_count += 1
                    blocked_action_types.append(action.action_type)
            loop_latencies.append(time.perf_counter() - loop_start)
            predictions.append(result.aggregated_value)
            thermal_sla_violations += int(consensus.temperature > float(verifier.config["temperature_sla_c"]))
        loop_lat = np.asarray(loop_latencies, dtype=float)
        sup_lat = np.asarray(supervisor_latencies, dtype=float) if supervisor_latencies else np.asarray([0.0])
        summary = error_summary(np.asarray(predictions), truth)
        base = {
            "scenario": scenario,
            "supervisor": "deterministic_rule_based",
            "seed": parts.get("seed"),
            "attack_type": parts.get("attack_type"),
            "attack_ratio": parts.get("attack_ratio"),
            "sensor_noise_level": parts.get("sensor_noise_level"),
            "sensors_per_group": sensor_count,
            "alpha": alpha,
            "beta": beta,
            "ticks": int(len(frame)),
            "supervisor_calls": int(supervisor_calls),
            "supervisor_invocation_rate": float(supervisor_calls / max(len(frame), 1)),
            "supervisor_failures": 0,
            "supervisor_failure_rate": 0.0,
            "verifier_calls": int(verifier_calls),
            "mean_loop_latency_ms": float(loop_lat.mean() * 1000.0),
            "p95_loop_latency_ms": float(np.quantile(loop_lat, 0.95) * 1000.0),
            "p99_loop_latency_ms": float(np.quantile(loop_lat, 0.99) * 1000.0),
            "mean_supervisor_latency_seconds": float(sup_lat.mean()),
            "p95_supervisor_latency_seconds": float(np.quantile(sup_lat, 0.95)),
            "unsafe_actions_proposed": int(unsafe_proposed),
            "unsafe_actions_blocked_by_verifier": int(unsafe_blocked),
            "unsafe_actions_executed": int(unsafe_executed),
            "safe_actions_accepted": int(safe_accepted),
            "deterministic_safe_actions": int(deterministic_safe),
            "fallback_count": int(fallback_count),
            "thermal_sla_violation_rate": float(thermal_sla_violations / max(len(frame), 1)),
            "accepted_action_types": jsonish_counts(accepted_action_types),
            "blocked_action_types": jsonish_counts(blocked_action_types),
            **summary,
        }
        for deadline in deadlines:
            row_out = dict(base)
            row_out["deadline_ms"] = int(deadline)
            row_out["deadline_miss_rate"] = float(np.mean(loop_lat * 1000.0 > int(deadline)))
            rows.append(row_out)
        safety_rows.append(base)
    rule_metrics = pd.DataFrame(rows)
    rule_safety = pd.DataFrame(safety_rows)
    rule_metrics.to_csv(outputs / "rule_supervisor_realtime_metrics.csv", index=False)
    rule_safety.to_csv(outputs / "rule_supervisor_safety_metrics.csv", index=False)
    comparison = compare_with_llm(rule_metrics, results_dir, outputs)
    write_findings(comparison, outputs)
    plot_supervisor_comparison(comparison, outputs / "fig_supervisor_comparison.pdf")
    return comparison


def jsonish_counts(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def compare_with_llm(rule_metrics: pd.DataFrame, results_dir: Path, outputs: Path) -> pd.DataFrame:
    rows = []
    if not rule_metrics.empty:
        rows.extend(_summary_rows(rule_metrics, "deterministic_rule_based", "supervisor"))
    llm_path = results_dir / "realtime_metrics.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        if not llm.empty:
            normalized = llm.copy()
            normalized["supervisor_calls"] = normalized.get("llm_calls", 0)
            normalized["supervisor_invocation_rate"] = normalized.get("llm_invocation_rate", 0.0)
            normalized["supervisor_failures"] = normalized.get("llm_failures", 0)
            normalized["supervisor_failure_rate"] = normalized.get("llm_failure_rate", 0.0)
            normalized["mean_supervisor_latency_seconds"] = normalized.get("mean_llm_latency_seconds", 0.0)
            normalized["p95_supervisor_latency_seconds"] = normalized.get("p95_llm_latency_seconds", 0.0)
            normalized["fallback_count"] = normalized.get("unsafe_actions_blocked_by_verifier", 0)
            rows.extend(_summary_rows(normalized, "langgraph_gemma_event_triggered", "llm"))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(outputs / "supervisor_comparison.csv", index=False)
    return comparison


def _summary_rows(frame: pd.DataFrame, label: str, source: str) -> list[dict[str, object]]:
    dedup = frame.drop_duplicates("scenario") if "scenario" in frame else frame
    deadline_frame = frame
    rows = []
    metrics = [
        "supervisor_invocation_rate",
        "supervisor_failure_rate",
        "mean_loop_latency_ms",
        "p95_loop_latency_ms",
        "p99_loop_latency_ms",
        "mean_supervisor_latency_seconds",
        "p95_supervisor_latency_seconds",
        "deadline_miss_rate",
        "thermal_sla_violation_rate",
        "mae",
        "rmse",
        "safe_actions_accepted",
        "unsafe_actions_proposed",
        "unsafe_actions_blocked_by_verifier",
        "unsafe_actions_executed",
        "fallback_count",
    ]
    for metric in metrics:
        src = deadline_frame if metric == "deadline_miss_rate" else dedup
        if metric in src:
            rows.append(
                {
                    "supervisor": label,
                    "source": source,
                    "metric": metric,
                    "mean": float(src[metric].mean(numeric_only=True)),
                    "median": float(src[metric].median(numeric_only=True)),
                    "p95": float(np.percentile(src[metric].to_numpy(dtype=float), 95)),
                    "n": int(len(src)),
                }
            )
    return rows


def write_findings(comparison: pd.DataFrame, outputs: Path) -> None:
    if comparison.empty:
        write_text(outputs / "llm_value_findings.md", "# LLM value findings\n\nNo comparison data were available.\n")
        return

    def get(supervisor: str, metric: str) -> float:
        row = comparison[(comparison["supervisor"] == supervisor) & (comparison["metric"] == metric)]
        return float(row["mean"].iloc[0]) if not row.empty else math.nan

    rule_label = "deterministic_rule_based"
    llm_label = "langgraph_gemma_event_triggered"
    lines = [
        "# LLM value findings",
        "",
        "This comparison uses the same realtime traces, event-trigger conditions, and non-bypassable verifier. "
        "The deterministic supervisor does not call an LLM; LangGraph/Gemma values are read from `results/realtime_metrics.csv` when available.",
        "",
        "## Summary metrics",
        f"- Rule-based invocation rate: {get(rule_label, 'supervisor_invocation_rate'):.6g}.",
        f"- LangGraph/Gemma invocation rate: {get(llm_label, 'supervisor_invocation_rate'):.6g}.",
        f"- Rule-based mean loop latency (ms): {get(rule_label, 'mean_loop_latency_ms'):.6g}.",
        f"- LangGraph/Gemma mean loop latency (ms): {get(llm_label, 'mean_loop_latency_ms'):.6g}.",
        f"- Rule-based unsafe executed: {get(rule_label, 'unsafe_actions_executed'):.6g}.",
        f"- LangGraph/Gemma unsafe executed: {get(llm_label, 'unsafe_actions_executed'):.6g}.",
        "",
        "## Interpretation guidance",
        "If the rule-based supervisor matches the LLM on safety/SLA outcomes with substantially lower latency, the paper should qualify the LLM-value claim: "
        "the LLM is valuable as an auditable semantic supervisor, but not necessarily required for low-level realtime safety. "
        "If LangGraph/Gemma improves accepted safe actions, fallback reduction, or recovery metrics, those differences should be reported as measured value-add.",
    ]
    write_text(outputs / "llm_value_findings.md", "\n".join(lines) + "\n")


def plot_supervisor_comparison(comparison: pd.DataFrame, path: Path) -> None:
    if comparison.empty:
        return
    selected = comparison[comparison["metric"].isin(["mean_loop_latency_ms", "deadline_miss_rate", "safe_actions_accepted", "unsafe_actions_executed"])]
    if selected.empty:
        return
    pivot = selected.pivot(index="metric", columns="supervisor", values="mean")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean value")
    ax.set_title("Rule-based supervisor vs. LangGraph/Gemma")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
