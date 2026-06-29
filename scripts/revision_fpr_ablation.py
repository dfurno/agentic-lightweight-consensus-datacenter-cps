#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.closing_experiments import run_fpr_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C6 FPR component ablation.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--max-traces", type=int, default=None)
    args = parser.parse_args()
    run_fpr_ablation(Path(args.results), Path(args.outputs), max_traces=args.max_traces)


if __name__ == "__main__":
    main()
