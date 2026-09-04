#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.artifacts import scenario_parts
from src.revision.replay_manifest import sha256_file


CONFIGS = [(a, m) for a in (0.3, 0.5, 0.7) for m in (2.0, 4.0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    traces_dir = args.original / "results" / "traces"
    all_traces = sorted(traces_dir.glob("full_*.csv"))
    selected = []
    for path in all_traces:
        p = scenario_parts(path.stem)
        affected = p["seed"] in {11, 42, 101} and p["attack_type"] == "bias" and p["beta"] == 0.7
        bias_control = p["seed"] == 11 and p["attack_type"] == "bias" and p["attack_ratio"] == 0.4 and p["sensors_per_group"] == 9 and p["beta"] == 0.8
        nonbias_control = p["seed"] == 11 and p["attack_type"] == "drift" and p["attack_ratio"] == 0.4 and p["sensors_per_group"] == 9 and p["beta"] == 0.7
        if affected or bias_control or nonbias_control:
            selected.append({"path": str(path), "sha256": sha256_file(path), "ticks": 240, "affected": affected})
    affected_count = sum(item["affected"] for item in selected)
    if affected_count != 135 or len(selected) != 141:
        raise SystemExit(f"Unexpected selection: affected={affected_count}, total={len(selected)}")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    historical = [args.original / "results" / "metrics.csv"]
    for alpha_crp, m_z in CONFIGS:
        suffix = f"acrp{alpha_crp}_mz{m_z}"
        historical.extend([
            args.original / "outputs" / f"crp_estimation_per_scenario_{suffix}.csv",
            args.original / "outputs" / f"crp_convergence_{suffix}.csv",
        ])
    historical_inputs = [{"path": str(path), "sha256": sha256_file(path)} for path in historical]
    manifests = args.run_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=False)
    for alpha_crp, m_z in CONFIGS:
        config_id = f"acrp{alpha_crp}_mz{m_z}"
        manifest = {
            "run_id": args.run_dir.name,
            "configuration_id": config_id,
            "source_code_commit": commit,
            "replay_configuration": {"alpha_crp": alpha_crp, "m_z": m_z, "llm_calls": 0},
            "selection": {"main_seeds": [11, 42, 101], "affected": "full/bias/beta=0.7", "controls": "seed11, ratio0.4, s9: three noise levels each for bias/beta=0.8 and drift/beta=0.7"},
            "expected_counts": {"manifest_entries": 141, "affected_scenarios": 135, "control_scenarios": 6, "ticks_per_trace": 240, "total_ticks": 33840},
            "historical_inputs": historical_inputs,
            "traces": selected,
        }
        (manifests / f"{config_id}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
