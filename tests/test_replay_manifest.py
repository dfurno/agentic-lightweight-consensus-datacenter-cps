import hashlib
import json

import pandas as pd
import pytest

from src.revision.replay_manifest import load_and_validate_manifest


def _trace(tmp_path, name="full_seed11_bias_0p4_noise0p25_s2_a0p2_b0p7.csv"):
    path = tmp_path / name
    pd.DataFrame({
        "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"],
        "temperature_ground_truth": [1.0, 1.1],
        "sensor_0": [1.0, 1.1], "sensor_1": [1.2, 1.3],
        "attack_label_0": [False, False], "attack_label_1": [False, True],
    }).to_csv(path, index=False)
    return path


def _manifest(tmp_path, trace):
    value = {
        "replay_configuration": {"alpha_crp": 0.5, "m_z": 4.0},
        "expected_counts": {"manifest_entries": 1, "total_ticks": 2},
        "traces": [{"path": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(), "ticks": 2}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_valid_manifest(tmp_path):
    trace = _trace(tmp_path)
    manifest, _ = _manifest(tmp_path, trace)
    _, paths = load_and_validate_manifest(manifest, alpha_crp=0.5, m_z=4.0, output_dir=tmp_path / "new")
    assert paths == [trace]


@pytest.mark.parametrize("mutation", ["hash", "config", "duplicate", "schema"])
def test_invalid_manifest_conditions(tmp_path, mutation):
    trace = _trace(tmp_path)
    manifest, value = _manifest(tmp_path, trace)
    if mutation == "hash":
        value["traces"][0]["sha256"] = "0" * 64
    elif mutation == "config":
        value["replay_configuration"]["m_z"] = 2.0
    elif mutation == "duplicate":
        value["traces"].append(dict(value["traces"][0]))
        value["expected_counts"]["manifest_entries"] = 2
        value["expected_counts"]["total_ticks"] = 4
    else:
        pd.read_csv(trace).drop(columns="attack_label_1").to_csv(trace, index=False)
        value["traces"][0]["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_and_validate_manifest(manifest, alpha_crp=0.5, m_z=4.0, output_dir=tmp_path / "new")
