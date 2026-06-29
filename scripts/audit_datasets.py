#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.audit import audit_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--manifest", default="configs/datasets.yaml")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    audit_datasets(args.data_config, args.manifest, args.results_dir)


if __name__ == "__main__":
    main()
