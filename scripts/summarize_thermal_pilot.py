#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["iae_degC_min", "duration_above_sla_min", "peak_excess_degC",
           "integrated_excess_degC_min", "fan_effort_normalized_min",
           "fan_total_variation", "estimator_mae_c", "verifier_rejects",
           "fallback_ticks", "crp_nonconverged_ticks"]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit("Output destination already exists")
    args.output.mkdir(parents=True)
    summary = pd.read_csv(args.run / "summary.csv")
    tick_files = sorted((args.run / "checkpoints").glob("*.ticks.csv"))
    ticks = pd.concat([pd.read_csv(p) for p in tick_files], ignore_index=True)
    condition = summary.groupby("condition")[METRICS].agg(["mean", "median", "max"])
    condition.columns = [f"{a}_{b}" for a, b in condition.columns]
    condition.reset_index().to_csv(args.output / "condition_summary.csv", index=False)

    oracle = summary[summary.condition == "oracle"].set_index(["seed", "scenario_index"])
    paired = []
    for _, row in summary[summary.condition != "oracle"].iterrows():
        ref = oracle.loc[(row.seed, row.scenario_index)]
        item = {"seed": row.seed, "scenario_index": row.scenario_index, "attack": row.attack,
                "intensity": row.intensity, "condition": row.condition}
        for metric in METRICS[:7]: item[f"delta_{metric}_vs_oracle"] = row[metric] - ref[metric]
        paired.append(item)
    pd.DataFrame(paired).to_csv(args.output / "paired_differences.csv", index=False)

    diag = []
    for keys, frame in ticks.groupby(["seed", "scenario_index", "attack", "intensity", "condition"]):
        seed, index, attack, intensity, system = keys
        active = frame[frame.attack_active]
        detections = active.index[active.anomaly_count > 0].tolist()
        attack_start = int(active.tick.min()) if not active.empty else None
        first_tick = int(frame.loc[detections[0], "tick"]) if detections else None
        diag.append({"seed": seed, "scenario_index": index, "attack": attack, "intensity": intensity,
                     "condition": system, "detected_during_attack": bool(detections),
                     "detection_delay_ticks": None if first_tick is None else first_tick - attack_start,
                     "anomaly_ticks": int((frame.anomaly_count > 0).sum()),
                     "false_alarm_ticks_nominal": int((frame.anomaly_count > 0).sum()) if attack == "nominal" else 0,
                     "rejected_ticks": int((frame.verifier_accepted == False).sum()),
                     "crp_nonconverged_ticks": int((frame.crp_converged == False).sum()) if system == "fpr_crp" else 0,
                     "mean_crp_rounds": float(frame.crp_rounds.mean()) if system == "fpr_crp" else 0.0,
                     "mean_crp_latency_seconds": float(frame.crp_latency_seconds.mean()) if system == "fpr_crp" else None})
    pd.DataFrame(diag).to_csv(args.output / "diagnostics_summary.csv", index=False)

    deltas = ticks.fan_after - ticks.fan_before
    finite = np.isfinite(ticks[["true_temperature_c", "next_true_temperature_c", "estimate_c", "fan_after"]]).all().all()
    hashes_paired = summary.groupby(["seed", "scenario_index"]).paired_input_sha256.nunique()
    audit = {"condition_runs": int(len(summary)), "tick_logs": len(tick_files), "ticks": int(len(ticks)),
             "runtime_failures": int(135 - len(summary)), "finite_numeric_state": bool(finite),
             "paired_input_hashes_match": bool((hashes_paired == 1).all()),
             "fan_min_observed": float(ticks.fan_after.min()), "fan_max_observed": float(ticks.fan_after.max()),
             "max_absolute_fan_slew": float(deltas.abs().max()),
             "actuator_contract_violations": int(((ticks.fan_after < 0) | (ticks.fan_after > 1) | (deltas.abs() > 0.2 + 1e-12)).sum()),
             "nonrecovered_runs": int((~summary.recovered).sum()),
             "runs_with_sla_excess": int((summary.duration_above_sla_min > 0).sum()),
             "crp_nonconverged_ticks": int(summary.crp_nonconverged_ticks.sum()),
             "crp_tick_denominator": int((ticks.condition == "fpr_crp").sum())}
    (args.output / "validity_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__": main()
