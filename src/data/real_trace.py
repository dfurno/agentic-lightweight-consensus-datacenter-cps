from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class RealDatasetUnavailable(RuntimeError):
    """Raised when a real dataset was requested but cannot be used."""


@dataclass
class RealTraceInputs:
    timestamp: pd.Series
    power_proxy: np.ndarray
    ambient_temperature: np.ndarray
    cooling_proxy: np.ndarray | None
    ground_truth_temperature: np.ndarray | None
    sensor_readings: np.ndarray | None
    attack_template: np.ndarray | None
    metadata: dict[str, Any]


def is_git_lfs_pointer(path: str | Path) -> bool:
    file = Path(path)
    try:
        with file.open("rb") as handle:
            head = handle.read(128)
    except FileNotFoundError:
        return False
    return b"version https://git-lfs.github.com/spec/v1" in head


def _read_csv_materialized(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise RealDatasetUnavailable(f"Missing required file: {path}")
    if is_git_lfs_pointer(path):
        raise RealDatasetUnavailable(
            f"{path} is a Git LFS pointer, not materialized data. Run git lfs install && git lfs pull in the dataset directory."
        )
    return pd.read_csv(path, nrows=nrows)


def _find_timestamp_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        name = str(column).lower()
        if "timestamp" in name or name in {"time", "datetime", "date"}:
            return str(column)
    first = str(frame.columns[0])
    parsed = pd.to_datetime(frame[first], errors="coerce")
    if parsed.notna().mean() > 0.8:
        return first
    raise RealDatasetUnavailable("Could not infer a timestamp column from real dataset CSV.")


def _wide_meter_series(path: Path, max_rows: int) -> tuple[pd.Series, np.ndarray]:
    frame = _read_csv_materialized(path, nrows=max_rows)
    ts_col = _find_timestamp_column(frame)
    timestamps = pd.to_datetime(frame[ts_col], errors="coerce")
    numeric = frame.drop(columns=[ts_col], errors="ignore").select_dtypes(include=["number"])
    if numeric.empty:
        # Some CSVs store numeric values as strings; coerce all non-time columns.
        candidates = frame.drop(columns=[ts_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
        numeric = candidates.dropna(axis=1, how="all")
    if numeric.empty:
        raise RealDatasetUnavailable(f"No numeric meter columns found in {path}")
    series = numeric.mean(axis=1, skipna=True).interpolate(limit_direction="both").to_numpy(dtype=float)
    valid = timestamps.notna() & np.isfinite(series)
    if valid.sum() < 10:
        raise RealDatasetUnavailable(f"Not enough valid timestamp/numeric rows in {path}")
    return timestamps[valid].reset_index(drop=True), series[valid.to_numpy()]


def _weather_temperature(path: Path, timestamps: pd.Series, max_rows: int) -> np.ndarray:
    frame = _read_csv_materialized(path, nrows=max_rows)
    ts_col = _find_timestamp_column(frame)
    temp_cols = [col for col in frame.columns if "air_temperature" in str(col).lower() or "temperature" in str(col).lower()]
    if not temp_cols:
        return np.full(len(timestamps), 22.0)
    weather = frame[[ts_col, temp_cols[0]]].copy()
    weather[ts_col] = pd.to_datetime(weather[ts_col], errors="coerce")
    weather[temp_cols[0]] = pd.to_numeric(weather[temp_cols[0]], errors="coerce")
    weather = weather.dropna().sort_values(ts_col)
    if weather.empty:
        return np.full(len(timestamps), 22.0)
    aligned = (
        weather.set_index(ts_col)[temp_cols[0]]
        .reindex(pd.DatetimeIndex(timestamps))
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    return aligned.to_numpy(dtype=float)


def load_bdg2_inputs(
    root: str | Path = "data/raw/kaggle_hot_corridor/building_data_genome_2",
    max_rows: int = 5000,
) -> RealTraceInputs:
    base = Path(root)
    electricity = base / "data" / "meters" / "cleaned" / "electricity_cleaned.csv"
    chilled = base / "data" / "meters" / "cleaned" / "chilledwater_cleaned.csv"
    weather = base / "data" / "weather" / "weather.csv"
    timestamps, power = _wide_meter_series(electricity, max_rows)
    cooling = None
    if chilled.exists() and not is_git_lfs_pointer(chilled):
        chilled_ts, chilled_values = _wide_meter_series(chilled, max_rows)
        cooling = (
            pd.Series(chilled_values, index=pd.DatetimeIndex(chilled_ts))
            .reindex(pd.DatetimeIndex(timestamps))
            .interpolate(method="time", limit_direction="both")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    ambient = _weather_temperature(weather, timestamps, max_rows) if weather.exists() else np.full(len(timestamps), 22.0)
    power = np.nan_to_num(power, nan=np.nanmedian(power))
    if np.nanstd(power) > 0:
        power = 80.0 + 80.0 * (power - np.nanmin(power)) / (np.nanmax(power) - np.nanmin(power) + 1e-9)
    else:
        power = np.full_like(power, 120.0)
    return RealTraceInputs(
        timestamp=timestamps,
        power_proxy=power,
        ambient_temperature=np.nan_to_num(ambient, nan=np.nanmedian(ambient)),
        cooling_proxy=cooling,
        ground_truth_temperature=None,
        sensor_readings=None,
        attack_template=None,
        metadata={
            "source": "building_data_genome_2_substitute",
            "source_url": "https://github.com/buds-lab/building-data-genome-project-2",
            "role": "thermal_calibration_substitute",
            "limitations": (
                "BDG2 is a building energy/weather/cooling-meter dataset, not direct data-center hot-corridor telemetry. "
                "Electricity is used as a computing-load proxy and weather air temperature as ambient input."
            ),
            "rows_loaded": int(len(timestamps)),
            "electricity_file": str(electricity),
            "weather_file": str(weather),
            "chilledwater_file": str(chilled) if chilled.exists() else None,
        },
    )


def _read_hot_corridor(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise RealDatasetUnavailable(f"Missing required file: {path}")
    frame = pd.read_csv(path, sep=";", nrows=max_rows)
    if frame.shape[1] <= 1:
        frame = pd.read_csv(path, sep=None, engine="python", nrows=max_rows)
    return frame.apply(pd.to_numeric, errors="coerce")


def _optional_enact_power(root: Path, steps: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates = [
        root / "node_telemetry_pods_on.csv",
        root / "node_telemetry_pods_off.csv",
        root / "pod_telemetry_pods_on.csv",
        root / "pod_telemetry_pods_off.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "timestamp" not in frame.columns or "Energy (watts)" not in frame.columns:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["Energy (watts)"] = pd.to_numeric(frame["Energy (watts)"], errors="coerce")
        series = frame.dropna(subset=["timestamp", "Energy (watts)"]).groupby("timestamp")["Energy (watts)"].mean()
        if series.empty:
            continue
        values = series.interpolate(limit_direction="both").to_numpy(dtype=float)
        if values.size < steps:
            values = np.resize(values, steps)
        values = values[:steps]
        values = 80.0 + 80.0 * (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-9)
        return values, {"enact_file": str(path), "enact_rows_used": int(min(len(series), steps))}
    return None, {"enact_warning": "No usable ENACT Energy (watts) CSV found; Kaggle P_cu columns used as power proxy."}


def _optional_cucd_attack_template(root: Path, steps: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates = list(root.rglob("consolidated_dataset_raw.csv")) + list(root.rglob("noised_dataset.csv"))
    for path in candidates:
        frame = pd.read_csv(path, usecols=lambda col: col == "Label")
        if "Label" not in frame.columns or frame.empty:
            continue
        labels = pd.to_numeric(frame["Label"], errors="coerce").fillna(3).to_numpy()
        attack = labels != 3
        if attack.size < steps:
            attack = np.resize(attack, steps)
        return attack[:steps], {"cucd_file": str(path), "cucd_label_policy": "Label != 3 mapped to attack timing only"}
    return None, {"cucd_warning": "No usable CuCD-ID Label CSV found; configured attack windows used."}


def load_project_real_inputs(
    hot_corridor_path: str | Path = "data/raw/kaggle_hot_corridor/final_dataset_std.csv",
    enact_root: str | Path = "data/raw/enact",
    cucd_root: str | Path = "data/raw/cucd_id",
    max_rows: int = 5000,
) -> RealTraceInputs:
    hot = _read_hot_corridor(Path(hot_corridor_path), max_rows)
    truth_cols = [col for col in hot.columns if str(col).upper() == "TLHC"]
    sensor_cols = [col for col in hot.columns if str(col).startswith("T_MEAS-")]
    power_cols = [col for col in hot.columns if str(col).startswith("P_cu-")]
    ambient_cols = [col for col in hot.columns if str(col).startswith("T_out-")]
    if not truth_cols:
        raise RealDatasetUnavailable("Kaggle hot-corridor CSV does not contain TLHC column.")
    if not sensor_cols:
        raise RealDatasetUnavailable("Kaggle hot-corridor CSV does not contain T_MEAS-* sensor columns.")
    steps = len(hot)
    kaggle_power = hot[power_cols].mean(axis=1).interpolate(limit_direction="both").to_numpy(dtype=float) if power_cols else np.full(steps, 120.0)
    enact_power, enact_meta = _optional_enact_power(Path(enact_root), steps)
    power = enact_power if enact_power is not None else kaggle_power
    ambient = hot[ambient_cols].mean(axis=1).interpolate(limit_direction="both").to_numpy(dtype=float) if ambient_cols else np.full(steps, 0.0)
    sensor_readings = hot[sensor_cols].interpolate(limit_direction="both").to_numpy(dtype=float)
    truth = hot[truth_cols[0]].interpolate(limit_direction="both").to_numpy(dtype=float)
    attack_template, cucd_meta = _optional_cucd_attack_template(Path(cucd_root), steps)
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=steps, freq="60s"))
    return RealTraceInputs(
        timestamp=timestamps,
        power_proxy=np.nan_to_num(power, nan=np.nanmedian(power)),
        ambient_temperature=np.nan_to_num(ambient, nan=np.nanmedian(ambient)),
        cooling_proxy=None,
        ground_truth_temperature=np.nan_to_num(truth, nan=np.nanmedian(truth)),
        sensor_readings=np.nan_to_num(sensor_readings, nan=np.nanmedian(sensor_readings)),
        attack_template=attack_template,
        metadata={
            "source": "kaggle_hot_corridor_enact_cucd",
            "thermal_source_url": "https://www.kaggle.com/datasets/mbjunior/data-centre-hot-corridor-temperature-prediction",
            "enact_reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13091772/",
            "cucd_source_url": "https://data.mendeley.com/datasets/7n2d42pm3n/3",
            "role": "real hot-corridor thermal data with ENACT power proxy and CuCD-ID pattern-level attacks",
            "limitations": (
                "Kaggle hot-corridor values appear standardized; metrics are reported in dataset units unless inverse scaling metadata is provided. "
                "CuCD-ID is spacecraft cybersecurity telemetry and is used only for attack timing labels, not thermal measurements."
            ),
            "rows_loaded": int(steps),
            "sensor_columns": [str(col) for col in sensor_cols],
            "power_columns": [str(col) for col in power_cols],
            **enact_meta,
            **cucd_meta,
        },
    )
