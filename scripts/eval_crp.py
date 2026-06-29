#!/usr/bin/env python3
"""Replay-evaluate the CRP operator vs single-round FPR-OWA on recorded traces.

Usage:
    python3 scripts/eval_crp.py <results_dir> <outputs_dir> [alpha_crp] [M_z] [tag]

Outputs (in <outputs_dir>, suffixed with <tag> when given):
    crp_estimation_per_scenario.csv
    crp_anomaly_detection_metrics.csv
    crp_convergence.csv
Plus a console summary (global + per-attack MAE; mean AUROC over attacked ratios; convergence).

The recomputed fpr_owa global MAE is the sanity gate: it must match the manuscript (~1.6171).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus.crp import CRPState, crp_consensus
from src.consensus.fpr_owa import fpr_owa_consensus
from src.evaluation.metrics import error_summary
from src.revision.artifacts import (
    auc_from_curve, average_precision, binary_metrics, detection_delay,
    label_columns, roc_pr_points, scenario_from_trace, scenario_parts,
    sensor_columns, trace_files,
)

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs")
ALPHA_CRP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
M_Z = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
TAG = ("_" + sys.argv[5]) if len(sys.argv) > 5 else ""
OUT.mkdir(parents=True, exist_ok=True)

est_rows: list[dict] = []
conv_rows: list[dict] = []
acc: dict[tuple, dict[str, list]] = {}
delay_acc: dict[tuple, list[float]] = {}

files = [path for path in trace_files(RESULTS) if path.name.startswith("full_")]
print(f"traces: {len(files)}  alpha_crp={ALPHA_CRP} M_z={M_Z} tag='{TAG}'", flush=True)
if not files:
    sys.exit(f"ERROR: no full-suite trace CSVs under {RESULTS/'traces'} — run on the host that holds the per-tick full_*.csv traces.")

for fi, trace_path in enumerate(files):
    frame = pd.read_csv(trace_path)
    parts = scenario_parts(scenario_from_trace(trace_path))
    scols = sensor_columns(frame); lcols = label_columns(frame)
    if not scols or not lcols or "temperature_ground_truth" not in frame:
        continue
    n = min(len(scols), len(lcols)); scols = scols[:n]; lcols = lcols[:n]
    a = float(parts.get("alpha", 0.3)); b = float(parts.get("beta", 0.8))
    attack = parts.get("attack_type"); ratio = float(parts.get("attack_ratio", 0.0))
    S = frame[scols].to_numpy(dtype=float)
    L = frame[lcols].to_numpy(dtype=bool)
    truth = frame["temperature_ground_truth"].to_numpy(dtype=float)
    T = S.shape[0]

    state = CRPState(lam=0.3)
    est_fpr = np.empty(T); est_crp = np.empty(T)
    sc_fpr = np.empty((T, n)); fl_fpr = np.zeros((T, n), dtype=bool)
    sc_crp = np.empty((T, n)); fl_crp = np.zeros((T, n), dtype=bool)
    rounds = np.empty(T); conv = np.empty(T); cz = np.empty(T); excl = np.empty(T)
    for t in range(T):
        vals = S[t]
        r_fpr = fpr_owa_consensus(vals, alpha=a, beta=b)
        est_fpr[t] = r_fpr.aggregated_value
        sc_fpr[t] = 1.0 - r_fpr.reliability_scores
        fl_fpr[t] = r_fpr.anomaly_flags.astype(bool)
        r_crp = crp_consensus(vals, state, alpha=a, beta=b, alpha_crp=ALPHA_CRP, m_z=M_Z)
        est_crp[t] = r_crp.aggregated_value
        sc_crp[t] = 1.0 - r_crp.combined_reliability
        fl_crp[t] = r_crp.anomaly_flags
        rounds[t] = r_crp.rounds; conv[t] = float(r_crp.converged)
        cz[t] = r_crp.consensus_degree; excl[t] = float(r_crp.excluded.sum())

    for method, est in (("fpr_owa", est_fpr), ("crp", est_crp)):
        est_rows.append({**parts, "method": method, **error_summary(est, truth)})
    conv_rows.append({**parts, "mean_rounds": float(rounds.mean()), "converged_rate": float(conv.mean()),
                      "mean_consensus_degree": float(cz.mean()), "mean_excluded_per_tick": float(excl.mean())})
    for method, sc, fl in (("fpr_owa", sc_fpr, fl_fpr), ("crp", sc_crp, fl_crp)):
        key = (method, attack, ratio)
        d = acc.setdefault(key, {"y": [], "s": [], "f": []})
        d["y"].append(L.ravel()); d["s"].append(sc.ravel()); d["f"].append(fl.ravel())
        dk = delay_acc.setdefault(key, [])
        for j in range(n):
            dk.append(detection_delay(L[:, j], fl[:, j]))
    if (fi + 1) % 300 == 0:
        print(f"  processed {fi+1}/{len(files)}", flush=True)

est_df = pd.DataFrame(est_rows)
est_df.to_csv(OUT / f"crp_estimation_per_scenario{TAG}.csv", index=False)
glob = est_df.groupby("method")[["mae", "rmse", "median_absolute_error", "p95_absolute_error", "max_absolute_error"]].mean().round(5)
per_attack = est_df.groupby(["attack_type", "method"])["mae"].mean().round(5).unstack()
print("\n=== GLOBAL estimation (recomputed; fpr_owa MAE must ~match 1.6171) ===\n", glob.to_string())
print("\n=== per-attack MAE ===\n", per_attack.to_string())

rows = []
for (method, attack, ratio), d in acc.items():
    y = np.concatenate(d["y"]); s = np.concatenate(d["s"]); f = np.concatenate(d["f"])
    pts = roc_pr_points(y, s, method, str(attack), float(ratio))
    dl = np.array(delay_acc[(method, attack, ratio)], dtype=float)
    rows.append({"method": method, "attack_type": attack, "attack_ratio": ratio,
                 "auroc": auc_from_curve(pts["fpr"].to_numpy(), pts["tpr"].to_numpy()),
                 "auprc": average_precision(pts), **binary_metrics(y, f),
                 "mean_detection_delay_ticks": float(np.nanmean(dl)) if np.isfinite(dl).any() else np.nan})
ad = pd.DataFrame(rows).sort_values(["method", "attack_type", "attack_ratio"])
ad.to_csv(OUT / f"crp_anomaly_detection_metrics{TAG}.csv", index=False)
comp = ad[ad["attack_ratio"] > 0].groupby(["attack_type", "method"])[["auroc", "auprc", "f1", "false_positive_rate"]].mean().round(4)
print("\n=== anomaly detection mean over ratio>0 (auroc/auprc/f1/fpr) ===\n", comp.to_string())

conv_df = pd.DataFrame(conv_rows)
conv_df.to_csv(OUT / f"crp_convergence{TAG}.csv", index=False)
print("\n=== CRP convergence (overall) ===")
print(conv_df[["mean_rounds", "converged_rate", "mean_consensus_degree", "mean_excluded_per_tick"]].mean().round(4).to_string())
print("\n=== CRP converged_rate by attack ===\n", conv_df.groupby("attack_type")["converged_rate"].mean().round(4).to_string())
print("DONE", flush=True)
