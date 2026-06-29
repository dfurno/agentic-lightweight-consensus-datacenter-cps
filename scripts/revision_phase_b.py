#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.artifacts import RevisionPaths, phase_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase B post-processing from existing traces and CSV artifacts.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional development limit; omit on the remote full run.")
    args = parser.parse_args()
    phase_b(RevisionPaths(results=Path(args.results), outputs=Path(args.outputs)), max_traces=args.max_traces)


if __name__ == "__main__":
    main()
