from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.data.manifest import DatasetSpec, load_manifest
from src.utils.io import ensure_dir, write_json


def check_url(url: str, timeout: int = 10) -> dict[str, Any]:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code >= 400 or response.status_code == 405:
            response = requests.get(url, stream=True, timeout=timeout)
        return {
            "reachable": 200 <= response.status_code < 400,
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("content-type"),
        }
    except requests.RequestException as exc:
        return {
            "reachable": False,
            "status_code": None,
            "final_url": url,
            "content_type": None,
            "error": str(exc),
        }


def discover_datasets(
    manifest_path: str | Path = "configs/datasets.yaml",
    results_dir: str | Path = "results",
    check_remote: bool = True,
) -> list[dict[str, Any]]:
    specs = load_manifest(manifest_path)
    discovered: list[dict[str, Any]] = []
    for spec in specs:
        remote = None
        if check_remote and spec.source_url:
            remote = check_url(spec.source_url)
        if spec.status == "manual_required":
            validation_state = "manual_required"
        elif remote and remote.get("reachable"):
            validation_state = "candidate_validated"
        elif spec.source_url:
            validation_state = "unreachable"
        else:
            validation_state = "missing_source"
        discovered.append(
            {
                "id": spec.id,
                "name": spec.name,
                "role": spec.role,
                "status": spec.status,
                "source_url": spec.source_url,
                "license": spec.license,
                "download_method": spec.download_method,
                "target_dir": str(spec.target_dir),
                "notes": spec.notes,
                "remote": remote,
                "validation_state": validation_state,
                "accessed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    write_discovery_report(discovered, results_dir)
    return discovered


def write_discovery_report(records: list[dict[str, Any]], results_dir: str | Path) -> None:
    results = ensure_dir(results_dir)
    write_json(results / "dataset_discovery.json", records)
    lines = ["# Dataset Discovery", ""]
    for record in records:
        lines.extend(
            [
                f"## {record['name']} ({record['id']})",
                f"- Role: `{record['role']}`",
                f"- Manifest status: `{record['status']}`",
                f"- Validation state: `{record['validation_state']}`",
                f"- Source URL: {record['source_url'] or 'not provided'}",
                f"- License/terms: {record['license']}",
                f"- Download method: `{record['download_method']}`",
                f"- Target directory: `{record['target_dir']}`",
                f"- Notes: {record['notes']}",
                "",
            ]
        )
    (results / "dataset_discovery.md").write_text("\n".join(lines), encoding="utf-8")
