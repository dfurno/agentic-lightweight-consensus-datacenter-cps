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

from src.consensus.baselines import run_baseline
from src.consensus.fpr_owa import fpr_owa_consensus
from src.data.real_trace import RealDatasetUnavailable, load_project_real_inputs
from src.evaluation.metrics import error_summary
from src.simulation.digital_twin import build_real_dataset_digital_twin, build_synthetic_digital_twin
from src.utils.io import ensure_dir, read_yaml, write_yaml


DEFAULT_SEEDS = [7, 11, 13, 17, 23, 29, 31, 37, 42, 101]
DEFAULT_METHODS = [
    "mean",
    "median",
    "trimmed_mean",
    "hampel_filter_then_mean",
    "huber_location",
    "tukey_biweight",
    "robust_kalman",
    "direct_owa",
    "fpr_owa",
]


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".5f") -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class RobustKalmanLocation:
    def __init__(self, process_noise: float = 0.50, measurement_noise: float = 0.10, huber_c: float = 1.345) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.huber_c = huber_c
        self.estimate: float | None = None
        self.variance = 1.0

    def update(self, values: np.ndarray) -> float:
        clean = np.asarray(values, dtype=float)
        measurement = float(np.median(clean))
        center = measurement if self.estimate is None else self.estimate
        scale = 1.4826 * float(np.median(np.abs(clean - float(np.median(clean))))) + 1e-6
        residual = measurement - center
        innovation_scale = max(scale, 1.0)
        clipped_residual = float(np.clip(residual, -self.huber_c * innovation_scale, self.huber_c * innovation_scale))
        robust_measurement = center + clipped_residual

        if self.estimate is None:
            self.estimate = robust_measurement
            self.variance = max(scale * scale, self.measurement_noise * self.measurement_noise)
            return self.estimate

        pred_variance = self.variance + self.process_noise
        obs_variance = max(scale * scale, self.measurement_noise * self.measurement_noise)
        kalman_gain = pred_variance / (pred_variance + obs_variance)
        self.estimate = float(self.estimate + kalman_gain * (robust_measurement - self.estimate))
        self.variance = float((1.0 - kalman_gain) * pred_variance)
        return self.estimate


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


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


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _build_or_load_trace(
    results: Path,
    scenario: str,
    sim_config: dict,
    sensor_count: int,
    noise: float,
    seed: int,
    data_mode: str,
    real_inputs: object | None,
    resume: bool,
) -> pd.DataFrame:
    trace_path = results / "traces" / f"{scenario}.csv"
    metadata_path = results / "traces" / f"{scenario}.metadata.json"
    if resume and trace_path.exists() and trace_path.stat().st_size > 0:
        return pd.read_csv(trace_path)
    if data_mode == "real":
        try:
            trace = build_real_dataset_digital_twin(sim_config, inputs=real_inputs)
        except RealDatasetUnavailable as exc:
            raise SystemExit(f"Real dataset mode requested but unavailable: {exc}") from exc
    elif data_mode == "synthetic":
        trace = build_synthetic_digital_twin(sim_config)
    else:
        try:
            trace = build_real_dataset_digital_twin(sim_config, inputs=real_inputs)
        except RealDatasetUnavailable:
            trace = build_synthetic_digital_twin(sim_config)
    all_sensor_cols = [col for col in trace.frame.columns if col.startswith("sensor_")]
    sensor_cols = _sensor_subset(all_sensor_cols, sensor_count)
    frame = _add_measurement_noise(trace.frame, sensor_cols, noise, seed)
    ensure_dir(trace_path.parent)
    frame.to_csv(trace_path, index=False)
    pd.Series(trace.metadata).to_json(metadata_path, indent=2)
    return frame


def _evaluate_methods(frame: pd.DataFrame, sensor_cols: list[str], methods: list[str], alpha: float, beta: float) -> tuple[list[dict], list[dict]]:
    truth = frame["temperature_ground_truth"].to_numpy()
    metric_rows: list[dict] = []
    latency_rows: list[dict] = []
    for method in methods:
        predictions: list[float] = []
        anomaly_flags: list[int] = []
        confidences: list[float] = []
        kalman = RobustKalmanLocation()
        start = time.perf_counter()
        for _, row in frame.iterrows():
            values = row[sensor_cols].to_numpy(dtype=float)
            if method == "fpr_owa":
                result = fpr_owa_consensus(values, alpha=alpha, beta=beta, timestamp=str(row.get("timestamp", "")))
                predictions.append(result.aggregated_value)
                anomaly_flags.append(int(result.anomaly_flags.sum()))
                confidences.append(float(result.confidence))
            elif method == "robust_kalman":
                predictions.append(kalman.update(values))
                anomaly_flags.append(0)
                confidences.append(1.0)
            else:
                predictions.append(run_baseline(method, values, alpha=alpha, beta=beta))
                anomaly_flags.append(0)
                confidences.append(1.0)
        elapsed = time.perf_counter() - start
        metric_rows.append(
            {
                "method": method,
                **error_summary(np.asarray(predictions), truth),
                "mean_anomaly_flags": float(np.mean(anomaly_flags)),
                "mean_confidence": float(np.mean(confidences)),
            }
        )
        latency_rows.append(
            {
                "method": method,
                "fuzzy_consensus_latency_seconds": elapsed,
                "deterministic_simulation_latency_seconds": 0.0,
            }
        )
    return metric_rows, latency_rows


def _write_summary(metrics: pd.DataFrame, latency: pd.DataFrame, outputs: Path, seeds: list[int]) -> None:
    global_table = metrics.groupby("method", as_index=False)[["mae", "rmse", "p95_absolute_error"]].mean(numeric_only=True)
    global_table = global_table.sort_values("mae")
    by_attack = metrics.groupby(["attack_type", "method"], as_index=False)["mae"].mean(numeric_only=True)
    by_seed = metrics.groupby(["seed", "method"], as_index=False)["mae"].mean(numeric_only=True)
    latency_table = latency.groupby("method", as_index=False)["fuzzy_consensus_latency_seconds"].mean(numeric_only=True)
    global_table.to_csv(outputs / "seed_extension_global_metrics.csv", index=False)
    by_attack.to_csv(outputs / "seed_extension_by_attack.csv", index=False)
    by_seed.to_csv(outputs / "seed_extension_by_seed.csv", index=False)
    latency_table.to_csv(outputs / "seed_extension_latency.csv", index=False)

    lines = [
        "# Seed and robust-baseline extension",
        "",
        f"- Seeds evaluated: {seeds}",
        f"- Number of seeds: {len(seeds)}",
        f"- Scenarios: {metrics['scenario'].nunique()}",
        f"- Methods: {sorted(metrics['method'].unique())}",
        "",
        "## Global MAE ranking",
        _markdown_table(global_table[["method", "mae", "rmse", "p95_absolute_error"]], ".5f"),
        "",
        "## Mean method runtime per scenario",
        _markdown_table(latency_table, ".6f"),
        "",
        "Interpretation note: this extension is CPU-only and does not rerun the LangGraph/Gemma supervisor. "
        "It addresses reviewer concerns about seed count and robust estimation baselines while preserving the "
        "original agentic safety and realtime experiments as separate artifacts.",
    ]
    (outputs / "seed_extension_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only seed and robust-baseline extension for the major revision.")
    parser.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="real")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--results", default="results")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--max-scenarios", type=int, default=0, help="Debug cap; 0 means the full extended grid.")
    args = parser.parse_args()

    results = ensure_dir(args.results)
    outputs = ensure_dir(args.outputs)
    config = read_yaml("configs/experiments.yaml")
    sim_base = read_yaml("configs/simulation.yaml")
    suite = dict(config["full"])
    suite["seeds"] = _parse_int_list(args.seeds)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    grid = list(_scenario_grid(suite, sim_base))
    if args.max_scenarios > 0:
        grid = grid[: args.max_scenarios]

    real_inputs = None
    if args.data_mode in {"real", "auto"}:
        try:
            real_inputs = load_project_real_inputs(max_rows=int(sim_base.get("time_steps", 240)))
        except RealDatasetUnavailable:
            if args.data_mode == "real":
                raise

    metrics_rows: list[dict] = []
    latency_rows: list[dict] = []
    for index, (seed, attack_ratio, attack_type, noise, sensor_count, alpha, beta) in enumerate(grid, start=1):
        scenario = f"full_{_scenario_suffix(seed, attack_type, attack_ratio, noise, sensor_count, alpha, beta)}"
        print(f"[seed-extension] scenario {index}/{len(grid)}: {scenario}", flush=True)
        sim_config = dict(sim_base)
        sim_config["seed"] = seed
        sim_config["num_sensors"] = sensor_count
        sim_config["sensor_noise_std"] = noise
        sim_config["attacks"] = dict(sim_base["attacks"])
        sim_config["attacks"]["attack_ratio"] = attack_ratio
        sim_config["attacks"]["attack_type"] = attack_type
        frame = _build_or_load_trace(results, scenario, sim_config, sensor_count, noise, seed, args.data_mode, real_inputs, args.resume)
        sensor_cols = _sensor_subset([col for col in frame.columns if col.startswith("sensor_")], sensor_count)
        method_metrics, method_latency = _evaluate_methods(frame, sensor_cols, methods, alpha, beta)
        common = {
            "scenario": scenario,
            "seed": seed,
            "attack_type": attack_type,
            "attack_ratio": attack_ratio,
            "sensor_noise_level": noise,
            "sensors_per_group": sensor_count,
            "alpha": alpha,
            "beta": beta,
        }
        metrics_rows.extend({**common, **row} for row in method_metrics)
        latency_rows.extend({**common, **row} for row in method_latency)
        if index % 25 == 0 or index == len(grid):
            metrics = pd.DataFrame(metrics_rows)
            latency = pd.DataFrame(latency_rows)
            _atomic_write_csv(metrics, outputs / "seed_extension_metrics.csv")
            _atomic_write_csv(latency, outputs / "seed_extension_latency_by_scenario.csv")
            _write_summary(metrics, latency, outputs, suite["seeds"])

    metrics = pd.DataFrame(metrics_rows)
    latency = pd.DataFrame(latency_rows)
    _atomic_write_csv(metrics, outputs / "seed_extension_metrics.csv")
    _atomic_write_csv(latency, outputs / "seed_extension_latency_by_scenario.csv")
    _write_summary(metrics, latency, outputs, suite["seeds"])
    write_yaml(
        outputs / "seed_extension_config.yaml",
        {
            "data_mode": args.data_mode,
            "seeds": suite["seeds"],
            "methods": methods,
            "scenario_count": int(metrics["scenario"].nunique()),
            "row_count": int(len(metrics)),
        },
    )
    print("[seed-extension] DONE", flush=True)


if __name__ == "__main__":
    main()
