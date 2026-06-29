#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.artifacts import RevisionPaths, bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble major-revision reproducibility bundle.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    bundle(RevisionPaths(results=Path(args.results), outputs=Path(args.outputs)))


if __name__ == "__main__":
    main()
