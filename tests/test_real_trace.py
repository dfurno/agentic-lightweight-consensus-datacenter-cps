from pathlib import Path

import pandas as pd

from src.data.real_trace import is_git_lfs_pointer, load_bdg2_inputs
from src.data.real_trace import load_project_real_inputs
from src.simulation.digital_twin import build_real_dataset_digital_twin


def test_lfs_pointer_detection(tmp_path):
    pointer = tmp_path / "data.csv"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 10\n",
        encoding="utf-8",
    )
    assert is_git_lfs_pointer(pointer)


def test_bdg2_adapter_builds_real_trace_from_wide_csv(tmp_path):
    root = tmp_path / "building_data_genome_2"
    meters = root / "data" / "meters" / "cleaned"
    weather = root / "data" / "weather"
    meters.mkdir(parents=True)
    weather.mkdir(parents=True)
    timestamps = pd.date_range("2026-01-01", periods=24, freq="h")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "meter_a": range(24),
            "meter_b": range(10, 34),
        }
    ).to_csv(meters / "electricity_cleaned.csv", index=False)
    pd.DataFrame({"timestamp": timestamps, "air_temperature": [20 + i * 0.1 for i in range(24)]}).to_csv(
        weather / "weather.csv", index=False
    )
    inputs = load_bdg2_inputs(root, max_rows=24)
    trace = build_real_dataset_digital_twin({"seed": 1, "time_steps": 12, "num_sensors": 3, "attacks": {"enabled": False}}, inputs)
    assert trace.metadata["source"] == "building_data_genome_2_substitute"
    assert trace.metadata["dataset_driven"]
    assert len(trace.frame) == 12
    assert {"sensor_0", "sensor_1", "sensor_2", "temperature_ground_truth"}.issubset(trace.frame.columns)


def test_project_adapter_uses_hot_corridor_enact_and_cucd(tmp_path):
    hot = tmp_path / "hot.csv"
    enact = tmp_path / "enact"
    cucd = tmp_path / "cucd" / "Data" / "Raw"
    enact.mkdir()
    cucd.mkdir(parents=True)
    pd.DataFrame(
        {
            "P_cu-0": [1.0, 2.0, 3.0, 4.0],
            "T_out-0": [0.1, 0.2, 0.3, 0.4],
            "T_MEAS-0": [10.0, 11.0, 12.0, 13.0],
            "T_MEAS-1": [10.1, 11.1, 12.1, 13.1],
            "TLHC": [9.9, 10.9, 11.9, 12.9],
        }
    ).to_csv(hot, sep=";", index=False)
    pd.DataFrame(
        {
            "node_name": ["n"] * 4,
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="s"),
            "Energy (watts)": [80, 90, 100, 110],
        }
    ).to_csv(enact / "node_telemetry_pods_on.csv", index=False)
    pd.DataFrame({"Label": [3, 0, 3, 1]}).to_csv(cucd / "consolidated_dataset_raw.csv", index=False)
    inputs = load_project_real_inputs(hot, enact, tmp_path / "cucd", max_rows=4)
    assert inputs.metadata["source"] == "kaggle_hot_corridor_enact_cucd"
    assert inputs.ground_truth_temperature is not None
    assert inputs.sensor_readings is not None
    assert inputs.attack_template is not None
    assert inputs.attack_template.tolist() == [False, True, False, True]
