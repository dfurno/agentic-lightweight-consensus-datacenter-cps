#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus.fpr_owa import fpr_owa_consensus
from src.consensus.owa import direct_owa
from src.evaluation.metrics import error_summary
from src.revision.artifacts import calibration, scenario_parts
from src.revision.replay_manifest import sha256_file


def affected(scenario: str) -> bool:
    p = scenario_parts(scenario)
    return p["seed"] in {11, 42, 101} and p["attack_type"] == "bias" and p["beta"] == 0.7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    items = [item for item in manifest["traces"] if item["affected"]]
    if len(items) != 135:
        raise SystemExit("Expected 135 affected traces")

    ablation_path = args.original / "outputs" / "fpr_ablation.csv"
    diagnostic_path = args.original / "outputs" / "diagnostic_sensor_tick_records.csv.gz"
    historical_hashes = {Path(item["path"]): item["sha256"] for item in manifest["historical_inputs"]}
    extra_inputs = {str(ablation_path): sha256_file(ablation_path), str(diagnostic_path): sha256_file(diagnostic_path)}
    tick_errors: dict[tuple[str, int], float] = {}
    replacement_rows = []
    for item in items:
        path = Path(item["path"])
        frame = pd.read_csv(path)
        p = scenario_parts(path.stem)
        sensors = [c for c in frame if c.startswith("sensor_")]
        truth = frame["temperature_ground_truth"].to_numpy(float)
        values = frame[sensors].to_numpy(float)
        preds = {
            "owa_only": np.array([direct_owa(row, p["alpha"], p["beta"]) for row in values]),
            "full_fpr_owa": np.array([fpr_owa_consensus(row, alpha=p["alpha"], beta=p["beta"]).aggregated_value for row in values]),
        }
        for method, prediction in preds.items():
            replacement_rows.append({**p, "method": method, **error_summary(prediction, truth)})
        for tick, error in enumerate(np.abs(preds["full_fpr_owa"] - truth)):
            tick_errors[(path.stem, tick)] = float(error)

    old = pd.read_csv(ablation_path)
    if len(old) != 7636 or old[["scenario", "method"]].duplicated().any():
        raise SystemExit("Historical ablation cardinality/key failure")
    old_idx = old.set_index(["scenario", "method"])
    replacement = pd.DataFrame(replacement_rows).set_index(["scenario", "method"])
    columns = ["mae", "rmse", "median_absolute_error", "p95_absolute_error", "max_absolute_error"]
    old_idx.loc[replacement.index, columns] = replacement[columns]
    corrected = old_idx.reset_index()[old.columns]
    corrected["beta"] = corrected["scenario"].map(lambda x: scenario_parts(x)["beta"])
    corrected.to_csv(args.output / "fpr_ablation.csv", index=False)
    independent = corrected.method.isin(["reliability_only", "fpr_dominance_only"])
    max_independent_diff = float(np.max(np.abs(corrected.loc[independent, columns].to_numpy() - old.loc[independent, columns].to_numpy())))
    plot = corrected.groupby("method", as_index=False)["mae"].mean().sort_values("mae")
    fig, ax = plt.subplots(figsize=(5.6, 3.4)); ax.bar(plot.method, plot.mae); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("MAE"); fig.tight_layout()
    fig.savefig(args.output / "fig_fpr_ablation.pdf"); plt.close(fig)

    chunks = []
    updated_diagnostic_records = 0
    for chunk in pd.read_csv(diagnostic_path, chunksize=500000):
        keys = pd.Series(list(zip(chunk["scenario"], chunk["tick"])), index=chunk.index)
        updates = keys.map(tick_errors)
        mask = updates.notna()
        chunk.loc[mask, "absolute_error"] = updates[mask]
        updated_diagnostic_records += int(mask.sum())
        chunks.append(chunk)
    diagnostic = pd.concat(chunks, ignore_index=True)
    calibration(diagnostic, args.output)
    summary = {
        "ablation_rows": len(corrected), "ablation_scenarios": corrected.scenario.nunique(),
        "replaced_ablation_rows": len(replacement), "independent_method_max_abs_diff": max_independent_diff,
        "diagnostic_records_reused": len(diagnostic), "updated_diagnostic_records": updated_diagnostic_records,
        "replaced_scenario_ticks": len(tick_errors),
        "input_sha256": {**{str(k): v for k, v in historical_hashes.items()}, **extra_inputs},
        "output_sha256": {p.name: sha256_file(p) for p in sorted(args.output.glob("*"))},
    }
    (args.output / "dependent_validation.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
