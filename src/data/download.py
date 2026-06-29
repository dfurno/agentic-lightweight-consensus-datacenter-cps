from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.data.manifest import DatasetSpec, load_manifest
from src.utils.io import ensure_dir, write_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_manifest_datasets(
    manifest_path: str | Path = "configs/datasets.yaml",
    results_dir: str | Path = "results",
    allow_downloads: bool | None = None,
) -> list[dict[str, Any]]:
    if allow_downloads is None:
        allow_downloads = os.getenv("ALLOW_DATASET_DOWNLOADS", "false").lower() == "true"
    records: list[dict[str, Any]] = []
    for spec in load_manifest(manifest_path):
        records.append(download_dataset(spec, allow_downloads))
    write_json(Path(results_dir) / "dataset_downloads.json", records)
    return records


def download_dataset(spec: DatasetSpec, allow_downloads: bool) -> dict[str, Any]:
    base = {
        "id": spec.id,
        "name": spec.name,
        "source_url": spec.source_url,
        "target_dir": str(spec.target_dir),
        "accessed_at": datetime.now(timezone.utc).isoformat(),
    }
    if spec.status == "manual_required":
        return base | {
            "downloaded": False,
            "state": "manual_required",
            "reason": "Dataset source/version/license must be supplied before automated use.",
        }
    local_zip_states = {
        "local_zip_or_kaggle_api": ("archive.zip",),
        "local_zip_or_zenodo": ("18920397.zip",),
        "local_zip_or_mendeley": ("CubeSat Cybersecurity Dataset for Intrusion Detect.zip",),
    }
    if spec.download_method in local_zip_states:
        found = [name for name in local_zip_states[spec.download_method] if (Path("data/raw") / name).exists()]
        if found:
            return base | {
                "downloaded": True,
                "state": "local_archive_present",
                "path": str(Path("data/raw") / found[0]),
                "reason": "Local archive is present; extraction/audit can proceed without network download.",
            }
        if spec.download_method == "local_zip_or_kaggle_api" and (
            not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY")
        ):
            return base | {
                "downloaded": False,
                "state": "credentials_required",
                "reason": "Kaggle archive missing locally and KAGGLE_USERNAME/KAGGLE_KEY are required for API download.",
            }
    if spec.download_method == "kaggle_api":
        if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
            return base | {
                "downloaded": False,
                "state": "credentials_required",
                "reason": "KAGGLE_USERNAME and KAGGLE_KEY are required.",
            }
    if not allow_downloads:
        return base | {
            "downloaded": False,
            "state": "download_disabled",
            "reason": "Set ALLOW_DATASET_DOWNLOADS=true to permit network downloads.",
        }
    if not spec.source_url:
        return base | {"downloaded": False, "state": "missing_source", "reason": "No source URL."}
    if spec.download_method == "git_lfs":
        return clone_git_lfs_dataset(spec, base)
    if spec.download_method == "git_or_manual":
        return clone_git_dataset(spec, base)
    if spec.download_method not in {"manual_or_web"}:
        return base | {
            "downloaded": False,
            "state": "manual_or_git_required",
            "reason": f"Download method {spec.download_method!r} is not a direct file download.",
        }
    ensure_dir(spec.target_dir)
    out = spec.target_dir / "source.html"
    response = requests.get(spec.source_url, timeout=30)
    response.raise_for_status()
    out.write_bytes(response.content)
    return base | {
        "downloaded": True,
        "state": "downloaded",
        "path": str(out),
        "sha256": file_sha256(out),
        "bytes": out.stat().st_size,
    }


def clone_git_dataset(spec: DatasetSpec, base: dict[str, Any]) -> dict[str, Any]:
    if not shutil.which("git"):
        return base | {"downloaded": False, "state": "git_missing", "reason": "git executable not found."}
    if spec.target_dir.exists() and any(spec.target_dir.iterdir()):
        return base | {"downloaded": True, "state": "already_present", "path": str(spec.target_dir)}
    spec.target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", str(spec.source_url), str(spec.target_dir)])
    return base | {"downloaded": True, "state": "git_cloned", "path": str(spec.target_dir)}


def clone_git_lfs_dataset(spec: DatasetSpec, base: dict[str, Any]) -> dict[str, Any]:
    cloned = clone_git_dataset(spec, base)
    if not cloned.get("downloaded"):
        return cloned
    if not shutil.which("git-lfs") and not shutil.which("git"):
        return cloned | {"lfs_state": "git_lfs_missing", "lfs_reason": "Install git-lfs and run git lfs pull in the dataset directory."}
    if shutil.which("git-lfs"):
        subprocess.call(["git", "lfs", "install"], cwd=spec.target_dir)
        rc = subprocess.call(["git", "lfs", "pull"], cwd=spec.target_dir)
        if rc != 0:
            return cloned | {"lfs_state": "git_lfs_pull_failed", "lfs_reason": "git lfs pull returned non-zero."}
        return cloned | {"lfs_state": "git_lfs_pulled"}
    return cloned | {"lfs_state": "git_lfs_missing", "lfs_reason": "Install git-lfs and run git lfs pull in the dataset directory."}
