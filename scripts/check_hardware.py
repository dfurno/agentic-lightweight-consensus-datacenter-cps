#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import ensure_dir, write_json


def optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def nvidia_smi() -> dict[str, object]:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        )
        lines = [line.strip() for line in query.splitlines() if line.strip()]
        return {"available": True, "gpus": lines, "gpu_count": len(lines)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def main() -> None:
    report = {
        "os": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "nvidia_smi": nvidia_smi(),
        "torch_version": optional_version("torch"),
        "vllm_version": optional_version("vllm"),
        "gpu_environment": {
            key: os.getenv(key)
            for key in ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"]
            if os.getenv(key) is not None
        },
    }
    try:
        import psutil  # type: ignore

        report["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        report["ram_bytes"] = None
    try:
        import torch  # type: ignore

        report["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        report["torch_cuda_available"] = False
    ensure_dir("results")
    write_json(Path("results") / "hardware_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
