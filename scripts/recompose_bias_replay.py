#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.artifacts import scenario_parts
from src.revision.replay_manifest import sha256_file


CONFIGS = [(a, m) for a in (0.3, 0.5, 0.7) for m in (2.0, 4.0)]
ERROR_COLUMNS = ["mae", "rmse", "median_absolute_error", "p95_absolute_error", "max_absolute_error"]
CONVERGENCE_COLUMNS = ["mean_rounds", "converged_rate", "mean_consensus_degree", "mean_excluded_per_tick"]


def affected(scenario: str) -> bool:
    p = scenario_parts(scenario)
    return p["seed"] in {11, 42, 101} and p["attack_type"] == "bias" and p["beta"] == 0.7


def corrected_beta(scenario: str) -> float:
    return float(scenario_parts(scenario)["beta"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--destination", default="recomposed")
    args = parser.parse_args()
    destination = args.run_dir / args.destination
    destination.mkdir(parents=True, exist_ok=False)
    metrics = pd.read_csv(args.original / "results" / "metrics.csv")
    gate = metrics[metrics["method"] == "fpr_owa"].set_index("scenario")
    summaries = []
    invariants = []

    for alpha_crp, m_z in CONFIGS:
        config_id = f"acrp{alpha_crp}_mz{m_z}"
        replay_dir = args.run_dir / "outputs" / config_id
        replay_est = pd.read_csv(replay_dir / f"crp_estimation_per_scenario_{config_id}.csv")
        replay_conv = pd.read_csv(replay_dir / f"crp_convergence_{config_id}.csv")
        old_est_path = args.original / "outputs" / f"crp_estimation_per_scenario_{config_id}.csv"
        old_conv_path = args.original / "outputs" / f"crp_convergence_{config_id}.csv"
        old_est = pd.read_csv(old_est_path)
        old_conv = pd.read_csv(old_conv_path)
        if len(old_est) != 3780 or old_est[["scenario", "method"]].duplicated().any() or len(old_conv) != 1890 or old_conv["scenario"].duplicated().any():
            raise SystemExit(f"Historical cardinality/key failure for {config_id}")

        replay_fpr = replay_est[replay_est["method"] == "fpr_owa"].set_index("scenario")
        common = replay_fpr.index.intersection(gate.index)
        gate_diff = float(np.max(np.abs(replay_fpr.loc[common, ERROR_COLUMNS].to_numpy() - gate.loc[common, ERROR_COLUMNS].to_numpy())))
        if len(common) != 141 or gate_diff > 1e-12:
            raise SystemExit(f"FPR gate failed for {config_id}: n={len(common)}, max_diff={gate_diff}")

        old_idx = old_est.set_index(["scenario", "method"])
        replay_idx = replay_est.set_index(["scenario", "method"])
        replace_keys = [key for key in replay_idx.index if affected(key[0])]
        if len(replace_keys) != 270 or not set(replace_keys).issubset(old_idx.index):
            raise SystemExit(f"Replacement key failure for {config_id}")
        before = old_est.copy()
        old_idx.loc[replace_keys, ERROR_COLUMNS] = replay_idx.loc[replace_keys, ERROR_COLUMNS]
        recomposed = old_idx.reset_index()[old_est.columns]
        recomposed["beta"] = recomposed["scenario"].map(corrected_beta)
        if len(recomposed) != 3780 or recomposed[["scenario", "method"]].duplicated().any():
            raise SystemExit(f"Recomposed estimation cardinality failure for {config_id}")

        old_conv_idx = old_conv.set_index("scenario")
        replay_conv_idx = replay_conv.set_index("scenario")
        conv_common = replay_conv_idx.index.intersection(old_conv_idx.index)
        conv_diff = float(np.max(np.abs(replay_conv_idx.loc[conv_common, CONVERGENCE_COLUMNS].to_numpy() - old_conv_idx.loc[conv_common, CONVERGENCE_COLUMNS].to_numpy())))
        if len(conv_common) != 141 or conv_diff > 1e-12:
            raise SystemExit(f"Convergence invariance failed for {config_id}: n={len(conv_common)}, max_diff={conv_diff}")
        corrected_conv = old_conv.copy()
        corrected_conv["beta"] = corrected_conv["scenario"].map(corrected_beta)

        est_out = destination / f"crp_estimation_per_scenario_{config_id}.csv"
        conv_out = destination / f"crp_convergence_{config_id}.csv"
        recomposed.to_csv(est_out, index=False)
        corrected_conv.to_csv(conv_out, index=False)
        for method in ("fpr_owa", "crp"):
            old_rows = before.loc[before.method == method, ERROR_COLUMNS]
            new_rows = recomposed.loc[recomposed.method == method, ERROR_COLUMNS]
            for metric in ERROR_COLUMNS:
                old_mean = old_rows[metric].mean()
                new_mean = new_rows[metric].mean()
                summaries.append({"configuration_id": config_id, "method": method, "metric": metric, "historical_global_value": old_mean, "corrected_global_value": new_mean, "delta": new_mean - old_mean, "replaced_rows": 135})
        invariants.append({"configuration_id": config_id, "fpr_gate_scenarios": len(common), "fpr_gate_max_abs_diff": gate_diff, "convergence_scenarios": len(conv_common), "convergence_max_abs_diff": conv_diff, "estimation_rows": len(recomposed), "convergence_rows": len(corrected_conv)})

    pd.DataFrame(summaries).to_csv(destination / "global_metrics_before_after.csv", index=False)
    pd.DataFrame(invariants).to_csv(destination / "validation_invariants.csv", index=False)
    shutil.copy2(destination / "crp_estimation_per_scenario_acrp0.5_mz4.0.csv", destination / "crp_estimation_per_scenario.csv")
    shutil.copy2(destination / "crp_convergence_acrp0.5_mz4.0.csv", destination / "crp_convergence.csv")
    outputs = sorted(destination.glob("*.csv"))
    (destination / "output_hashes.json").write_text(json.dumps({p.name: sha256_file(p) for p in outputs}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
