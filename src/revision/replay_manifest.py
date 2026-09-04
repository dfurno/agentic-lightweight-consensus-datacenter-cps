from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.revision.artifacts import label_columns, scenario_parts, sensor_columns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_manifest(
    path: Path,
    *,
    alpha_crp: float,
    m_z: float,
    output_dir: Path,
) -> tuple[dict, list[Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = manifest.get("replay_configuration", {})
    if float(config.get("alpha_crp", np.nan)) != alpha_crp or float(config.get("m_z", np.nan)) != m_z:
        raise ValueError("Manifest replay_configuration disagrees with CLI alpha_crp/M_z")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not new/empty: {output_dir}")
    items = manifest.get("traces")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest traces must be a non-empty list")
    expected = manifest.get("expected_counts", {})
    if int(expected.get("manifest_entries", -1)) != len(items):
        raise ValueError("Manifest entry count disagrees with expected_counts")

    paths: list[Path] = []
    seen_paths: set[Path] = set()
    seen_scenarios: set[str] = set()
    total_ticks = 0
    for item in items:
        trace_path = Path(item["path"])
        scenario = trace_path.stem
        if trace_path in seen_paths or scenario in seen_scenarios:
            raise ValueError(f"Duplicate path or scenario in manifest: {scenario}")
        seen_paths.add(trace_path)
        seen_scenarios.add(scenario)
        if sha256_file(trace_path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {trace_path}")
        parts = scenario_parts(scenario)
        frame = pd.read_csv(trace_path)
        sensors = sensor_columns(frame)
        labels = label_columns(frame)
        required = {"temperature_ground_truth", "timestamp"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns in {scenario}: {sorted(missing)}")
        if len(sensors) != int(parts["sensors_per_group"]) or len(labels) != len(sensors):
            raise ValueError(f"Sensor/label schema mismatch in {scenario}")
        numeric = frame[sensors + ["temperature_ground_truth"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Non-finite numeric values in {scenario}")
        expected_ticks = int(item.get("ticks", -1))
        if len(frame) != expected_ticks:
            raise ValueError(f"Tick count mismatch in {scenario}: {len(frame)} != {expected_ticks}")
        total_ticks += len(frame)
        paths.append(trace_path)
    if int(expected.get("total_ticks", -1)) != total_ticks:
        raise ValueError("Total tick count disagrees with expected_counts")
    return manifest, paths
