#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download import download_manifest_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/datasets.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--allow", action="store_true")
    args = parser.parse_args()
    download_manifest_datasets(args.manifest, args.results_dir, allow_downloads=args.allow)


if __name__ == "__main__":
    main()
