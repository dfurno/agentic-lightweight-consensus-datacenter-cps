from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.real_trace import RealDatasetUnavailable, RealTraceInputs, load_bdg2_inputs, load_project_real_inputs
from src.simulation.attacks import apply_attack
from src.simulation.thermal_model import fallback_temperature
from src.utils.seed import set_seed


@dataclass
class TwinTrace:
    frame: pd.DataFrame
    metadata: dict[str, Any]


def build_synthetic_digital_twin(config: dict[str, Any]) -> TwinTrace:
    rng = set_seed(int(config.get("seed", 42)))
    steps = int(config.get("time_steps", 240))
    sensors = int(config.get("num_sensors", 9))
    period = int(config.get("sample_period_seconds", 60))
    t = np.arange(steps)
    power = 120.0 + 35.0 * np.sin(2 * np.pi * t / max(steps // 2, 1)) + rng.normal(0, 8, steps)
    thermal = config.get("thermal", {})
    truth = fallback_temperature(
        power=power,
        ambient=float(config.get("ambient_temperature_c", 22.0)),
        rho=float(thermal.get("rho", 0.82)),
        eta=float(thermal.get("eta", 0.018)),
        lag=int(thermal.get("lag", 2)),
        disturbance_std=float(thermal.get("disturbance_std", 0.08)),
        rng=rng,
    )
    noise_std = float(config.get("sensor_noise_std", 0.25))
    readings = np.vstack([truth + rng.normal(0.0, noise_std, steps) for _ in range(sensors)]).T
    attack_config = config.get("attacks", {})
    labels = np.zeros_like(readings, dtype=bool)
    if attack_config.get("enabled", True):
        attack_ratio = float(attack_config.get("attack_ratio", 0.2))
        compromised = max(0, int(round(sensors * attack_ratio)))
        for sensor_idx in range(compromised):
            readings[:, sensor_idx], labels[:, sensor_idx] = apply_attack(
                readings[:, sensor_idx],
                str(attack_config.get("attack_type", "mixed")),
                rng,
                bias_magnitude=float(attack_config.get("bias_magnitude", 4.0)),
                drift_rate=float(attack_config.get("drift_rate", 0.03)),
                spike_magnitude=float(attack_config.get("spike_magnitude", 8.0)),
                scaling_factor=float(attack_config.get("scaling_factor", 1.15)),
                start_fraction=float(attack_config.get("start_fraction", 0.35)),
                end_fraction=float(attack_config.get("end_fraction", 0.75)),
            )
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=steps, freq=f"{period}s"),
        "power_cu": power,
        "temperature_ground_truth": truth,
    }
    for idx in range(sensors):
        data[f"sensor_{idx}"] = readings[:, idx]
        data[f"attack_label_{idx}"] = labels[:, idx]
    return TwinTrace(
        frame=pd.DataFrame(data),
        metadata={
            "source": "synthetic_fallback",
            "dataset_warning": "Synthetic fallback used because audited real datasets are not required for test execution.",
            "num_sensors": sensors,
            "time_steps": steps,
        },
    )


def build_real_dataset_digital_twin(config: dict[str, Any], inputs: RealTraceInputs | None = None) -> TwinTrace:
    rng = set_seed(int(config.get("seed", 42)))
    inputs = inputs or load_project_real_inputs(max_rows=int(config.get("time_steps", 240)))
    steps = min(int(config.get("time_steps", len(inputs.power_proxy))), len(inputs.power_proxy))
    sensors = int(config.get("num_sensors", 9))
    power = inputs.power_proxy[:steps]
    ambient_series = inputs.ambient_temperature[:steps]
    if inputs.ground_truth_temperature is not None:
        truth = inputs.ground_truth_temperature[:steps]
    else:
        thermal = config.get("thermal", {})
        truth = fallback_temperature(
            power=power,
            ambient=float(np.nanmedian(ambient_series)),
            rho=float(thermal.get("rho", 0.82)),
            eta=float(thermal.get("eta", 0.018)),
            lag=int(thermal.get("lag", 2)),
            disturbance_std=float(thermal.get("disturbance_std", 0.08)),
            rng=rng,
        )
        truth = truth + (ambient_series - float(np.nanmedian(ambient_series)))
    noise_std = float(config.get("sensor_noise_std", 0.25))
    if inputs.sensor_readings is not None:
        base_readings = inputs.sensor_readings[:steps]
        if base_readings.shape[1] >= sensors:
            readings = base_readings[:, :sensors].copy()
        else:
            extra = np.vstack([truth + rng.normal(0.0, noise_std, steps) for _ in range(sensors - base_readings.shape[1])]).T
            readings = np.hstack([base_readings, extra])
    else:
        readings = np.vstack([truth + rng.normal(0.0, noise_std, steps) for _ in range(sensors)]).T
    attack_config = config.get("attacks", {})
    labels = np.zeros_like(readings, dtype=bool)
    if attack_config.get("enabled", True):
        attack_ratio = float(attack_config.get("attack_ratio", 0.2))
        compromised = max(0, int(round(sensors * attack_ratio)))
        for sensor_idx in range(compromised):
            attacked, generated_labels = apply_attack(
                readings[:, sensor_idx],
                str(attack_config.get("attack_type", "mixed")),
                rng,
                bias_magnitude=float(attack_config.get("bias_magnitude", 4.0)),
                drift_rate=float(attack_config.get("drift_rate", 0.03)),
                spike_magnitude=float(attack_config.get("spike_magnitude", 8.0)),
                scaling_factor=float(attack_config.get("scaling_factor", 1.15)),
                start_fraction=float(attack_config.get("start_fraction", 0.35)),
                end_fraction=float(attack_config.get("end_fraction", 0.75)),
            )
            if inputs.attack_template is not None and inputs.attack_template[:steps].any():
                template = inputs.attack_template[:steps].astype(bool)
                readings[template, sensor_idx] = attacked[template]
                labels[:, sensor_idx] = template
            else:
                readings[:, sensor_idx] = attacked
                labels[:, sensor_idx] = generated_labels
    data = {
        "timestamp": pd.to_datetime(inputs.timestamp.iloc[:steps]).reset_index(drop=True),
        "power_cu": power,
        "ambient_temperature": ambient_series,
        "temperature_ground_truth": truth,
    }
    if inputs.cooling_proxy is not None:
        data["cooling_proxy"] = inputs.cooling_proxy[:steps]
    for idx in range(sensors):
        data[f"sensor_{idx}"] = readings[:, idx]
        data[f"attack_label_{idx}"] = labels[:, idx]
    metadata = dict(inputs.metadata)
    metadata.update(
        {
            "num_sensors": sensors,
            "time_steps": steps,
            "dataset_driven": True,
            "temperature_model": (
                "measured TLHC ground truth and measured T_MEAS sensor channels"
                if inputs.ground_truth_temperature is not None and inputs.sensor_readings is not None
                else "transparent fallback thermal model driven by real input traces"
            ),
        }
    )
    return TwinTrace(frame=pd.DataFrame(data), metadata=metadata)


def build_digital_twin_from_available_data(config: dict[str, Any], data_mode: str = "auto") -> TwinTrace:
    if data_mode not in {"auto", "real", "synthetic"}:
        raise ValueError("data_mode must be one of auto, real, synthetic")
    if data_mode != "synthetic":
        try:
            return build_real_dataset_digital_twin(config)
        except RealDatasetUnavailable:
            if data_mode == "real":
                raise
    return build_synthetic_digital_twin(config)
