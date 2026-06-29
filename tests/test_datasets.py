from pathlib import Path

from src.data.audit import audit_datasets
from src.data.discovery import discover_datasets
from src.data.manifest import load_manifest
from src.utils.io import write_yaml


def test_manifest_contains_manual_required_and_substitutes():
    specs = load_manifest("configs/datasets.yaml")
    statuses = {spec.status for spec in specs}
    assert "primary" in statuses
    assert "substitute" in statuses


def test_discovery_primary_records_without_remote(tmp_path):
    records = discover_datasets("configs/datasets.yaml", tmp_path, check_remote=False)
    enact = next(record for record in records if record["id"] == "enact")
    assert enact["status"] == "primary"
    assert enact["validation_state"] in {"unreachable", "candidate_validated"}
    assert (tmp_path / "dataset_discovery.md").exists()


def test_audit_does_not_accept_cyber_as_temperature(tmp_path):
    raw = tmp_path / "raw" / "optional_iot_anomaly" / "ciciot2023"
    raw.mkdir(parents=True)
    (raw / "flows.csv").write_text("timestamp,label,flow_duration\n2026-01-01,spoofing,1.0\n", encoding="utf-8")
    manifest = tmp_path / "datasets.yaml"
    write_yaml(
        manifest,
        {
            "datasets": [
                {
                    "id": "ciciot2023",
                    "name": "CICIoT2023",
                    "role": "anomaly_timing",
                    "status": "substitute",
                    "source_url": "https://example.test",
                    "license": "terms",
                    "download_method": "manual",
                    "target_dir": str(raw),
                    "checksum": None,
                    "notes": "test",
                }
            ]
        },
    )
    data_config = tmp_path / "data.yaml"
    write_yaml(
        data_config,
        {
            "timestamp_column_candidates": ["timestamp"],
            "power_column_candidates": ["power"],
            "temperature_column_candidates": ["temperature", "temp"],
            "attack_label_column_candidates": ["label", "attack"],
        },
    )
    records = audit_datasets(data_config, manifest, tmp_path)
    assert records[0]["usable_for_attack_patterns"]
    assert not records[0]["usable_for_temperature"]
    assert records[0]["mapping_warning"]
