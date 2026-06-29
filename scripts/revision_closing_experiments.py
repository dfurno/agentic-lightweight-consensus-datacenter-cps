#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.closing_experiments import run_closing_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closing major-revision experiments: C6, C8, and C4.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional development limit for FPR ablation.")
    parser.add_argument("--repeats", type=int, default=1000, help="Repeats per micro-benchmark method/size.")
    args = parser.parse_args()
    run_closing_experiments(Path(args.results), Path(args.outputs), max_traces=args.max_traces, repeats=args.repeats)


if __name__ == "__main__":
    main()
