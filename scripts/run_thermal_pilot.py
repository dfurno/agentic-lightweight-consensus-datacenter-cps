#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.thermal_pilot import exogenous_arrays, run_condition


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenarios(cfg: dict) -> list[tuple[str, str | None]]:
    return [("nominal", None)] + [(attack, level) for attack in cfg["attacks"]["definitions"] for level in ("low", "high")]


def analytical_checks(cfg: dict) -> dict[str, object]:
    p, c = cfg["plant"], cfg["control"]
    coefficient = 1.0 - p["thermal_exchange_per_step"]
    equilibrium_fan = (p["thermal_exchange_per_step"] * (p["ambient_base_c"] - c["target_temperature_c"])
                       + p["load_base_c_per_step"]) / p["fan_gain_c_per_step"]
    reachable = c["fan_min"] <= equilibrium_fan <= c["fan_max"]
    return {"discrete_state_coefficient": coefficient, "stable": abs(coefficient) < 1,
            "nominal_equilibrium_fan": equilibrium_fan, "nominal_target_reachable": reachable,
            "dt_seconds": cfg["time"]["dt_seconds"], "passed": abs(coefficient) < 1 and reachable}


def validate_pairing(cfg: dict) -> None:
    for seed in cfg["seeds"]:
        for index, _ in enumerate(scenarios(cfg)):
            a = exogenous_arrays(seed, index, cfg); b = exogenous_arrays(seed, index, cfg)
            assert np.array_equal(a["process_noise"], b["process_noise"])
            assert np.array_equal(a["sensor_noise"], b["sensor_noise"])
            assert np.array_equal(a["attacked"], b["attacked"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    actual = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != args.commit: raise SystemExit(f"Commit mismatch: {args.commit} != {actual}")
    if args.output.exists(): raise SystemExit("Output destination already exists")
    cfg = yaml.safe_load(args.config.read_text()); manifest = json.loads(args.manifest.read_text())
    expected = len(cfg["seeds"]) * len(scenarios(cfg)) * len(cfg["conditions"])
    if expected != manifest["maximum_condition_runs"] or expected > 135: raise SystemExit("Run budget mismatch")
    checks = analytical_checks(cfg); validate_pairing(cfg)
    args.output.mkdir(parents=True); (args.output / "checkpoints").mkdir()
    (args.output / "validity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    provenance = {"run_id": cfg["run_id"], "source_code_commit": actual, "commit_verified_against_git": True,
                  "config_sha256": sha(args.config), "manifest_sha256": sha(args.manifest),
                  "runner_sha256": sha(Path(__file__)), "python": platform.python_version(),
                  "interpreter": sys.executable, "expected_condition_runs": expected,
                  "llm_calls": 0, "gpu_calls": 0}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not checks["passed"]: raise SystemExit("Analytical validity checks failed")
    if args.validate_only: return

    summaries: list[dict] = []; completed = 0
    for seed_pos, seed in enumerate(cfg["seeds"]):
        for scenario_index, (attack, level) in enumerate(scenarios(cfg)):
            arrays = exogenous_arrays(seed, scenario_index, cfg)
            array_hash = hashlib.sha256(arrays["process_noise"].tobytes() + arrays["sensor_noise"].tobytes()
                                           + arrays["attacked"].tobytes()).hexdigest()
            for condition in cfg["conditions"]:
                rows, summary = run_condition(seed, scenario_index, attack, level, condition, cfg, arrays)
                summary["paired_input_sha256"] = array_hash
                name = f"seed{seed}_{scenario_index:02d}_{attack}_{level or 'none'}_{condition}"
                pd.DataFrame(rows).to_csv(args.output / "checkpoints" / f"{name}.ticks.csv", index=False)
                (args.output / "checkpoints" / f"{name}.summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                summaries.append(summary); completed += 1
                if completed == 1:
                    ctl = cfg["control"]
                    if summary["ticks"] != cfg["time"]["horizon_ticks"] or summary["saturation_ticks"] > summary["ticks"] / 2:
                        pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
                        raise SystemExit("Nominal/oracle validity gate failed")
        if seed_pos == 0:
            first_seed = [s for s in summaries if s["seed"] == seed]
            invalid = any(s["ticks"] != cfg["time"]["horizon_ticks"] for s in first_seed)
            saturated = all(s["saturation_ticks"] > s["ticks"] / 2 for s in first_seed)
            all_rejected = all(s["verifier_accepts"] == 0 for s in first_seed)
            if invalid or saturated or all_rejected:
                pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
                raise SystemExit("First-seed validity gate failed")
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    provenance.update({"completed_condition_runs": completed, "runtime_failures": 0})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
