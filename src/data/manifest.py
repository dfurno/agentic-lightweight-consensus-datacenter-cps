from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import read_yaml


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    role: str
    status: str
    source_url: str | None
    license: str
    download_method: str
    target_dir: Path
    checksum: str | None
    notes: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DatasetSpec":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            role=str(data["role"]),
            status=str(data["status"]),
            source_url=data.get("source_url"),
            license=str(data.get("license", "unknown")),
            download_method=str(data.get("download_method", "manual")),
            target_dir=Path(str(data["target_dir"])),
            checksum=data.get("checksum"),
            notes=str(data.get("notes", "")),
        )


def load_manifest(path: str | Path = "configs/datasets.yaml") -> list[DatasetSpec]:
    raw = read_yaml(path)
    datasets = raw.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("configs/datasets.yaml must contain a datasets list")
    return [DatasetSpec.from_mapping(item) for item in datasets]
