from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.manifest import load_manifest
from src.utils.io import ensure_dir, read_yaml, write_json


def infer_columns(columns: list[str], candidates: list[str]) -> list[str]:
    lowered = {column: column.lower() for column in columns}
    matches = []
    for column, lower in lowered.items():
        if any(candidate in lower for candidate in candidates):
            matches.append(column)
    return matches


def inspect_tabular_file(path: Path, data_config: dict[str, Any]) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, nrows=5000)
            if frame.shape[1] == 1:
                frame = pd.read_csv(path, sep=";", nrows=5000)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif path.suffix.lower() in {".json", ".jsonl"}:
            frame = pd.read_json(path, lines=path.suffix.lower() == ".jsonl", nrows=5000)
        else:
            return {"path": str(path), "supported": False, "reason": "Unsupported file extension"}
    except Exception as exc:
        return {"path": str(path), "supported": False, "reason": str(exc)}
    columns = list(frame.columns)
    missingness = frame.isna().mean().round(4).to_dict()
    return {
        "path": str(path),
        "supported": True,
        "rows_sampled": int(len(frame)),
        "columns": columns,
        "timestamp_columns": infer_columns(columns, data_config["timestamp_column_candidates"]),
        "power_columns": infer_columns(columns, data_config["power_column_candidates"]),
        "temperature_columns": infer_columns(columns, data_config["temperature_column_candidates"]),
        "attack_label_columns": infer_columns(columns, data_config["attack_label_column_candidates"]),
        "missingness": missingness,
    }


def audit_datasets(
    data_config_path: str | Path = "configs/data.yaml",
    manifest_path: str | Path = "configs/datasets.yaml",
    results_dir: str | Path = "results",
) -> list[dict[str, Any]]:
    data_config = read_yaml(data_config_path)
    records: list[dict[str, Any]] = []
    for spec in load_manifest(manifest_path):
        files = []
        if spec.target_dir.exists():
            for suffix in ("*.csv", "*.parquet", "*.pq", "*.json", "*.jsonl"):
                files.extend(spec.target_dir.rglob(suffix))
        inspected = [inspect_tabular_file(path, data_config) for path in sorted(files)]
        columns = [col for item in inspected if item.get("supported") for col in item.get("columns", [])]
        power = any(item.get("power_columns") for item in inspected)
        temp = any(item.get("temperature_columns") for item in inspected)
        attack = any(item.get("attack_label_columns") for item in inspected)
        usable_for_power = power and spec.role == "workload_power"
        usable_for_temperature = temp and spec.role == "thermal_calibration"
        usable_for_attack_patterns = attack and spec.role == "anomaly_timing"
        not_usable = None
        if not files:
            not_usable = "No local tabular files found under target directory."
        elif spec.role == "anomaly_timing" and temp:
            not_usable = None
        elif spec.role == "anomaly_timing" and not attack:
            not_usable = "No attack label/category columns detected."
        elif spec.role == "thermal_calibration" and not temp:
            not_usable = "No temperature-like columns detected."
        elif spec.role == "workload_power" and not power:
            not_usable = "No workload/power-like columns detected."
        records.append(
            {
                "id": spec.id,
                "name": spec.name,
                "role": spec.role,
                "manifest_status": spec.status,
                "source_url": spec.source_url,
                "license": spec.license,
                "file_count": len(files),
                "columns_detected": sorted(set(columns)),
                "files": inspected,
                "usable_for_power": bool(usable_for_power),
                "usable_for_temperature": bool(usable_for_temperature),
                "usable_for_attack_patterns": bool(usable_for_attack_patterns),
                "not_usable_reason": not_usable,
                "mapping_warning": (
                    "Cyber/IoT datasets may only be used for pattern-level attack timing, not thermal measurements."
                    if spec.role == "anomaly_timing"
                    else None
                ),
            }
        )
    write_audit_report(records, results_dir)
    return records


def write_audit_report(records: list[dict[str, Any]], results_dir: str | Path) -> None:
    results = ensure_dir(results_dir)
    write_json(results / "data_audit.json", records)
    lines = ["# Data Audit", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['name']} ({record['id']})",
                f"- Role: `{record['role']}`",
                f"- Source: {record['source_url'] or 'not provided'}",
                f"- License/terms: {record['license']}",
                f"- Files inspected: {record['file_count']}",
                f"- Usable for power: `{record['usable_for_power']}`",
                f"- Usable for temperature: `{record['usable_for_temperature']}`",
                f"- Usable for attack patterns: `{record['usable_for_attack_patterns']}`",
                f"- Not usable reason: {record['not_usable_reason'] or 'none'}",
            ]
        )
        if record.get("mapping_warning"):
            lines.append(f"- Mapping warning: {record['mapping_warning']}")
        lines.append("")
    (results / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
