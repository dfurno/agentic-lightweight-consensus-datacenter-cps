#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.closing_experiments import run_compute_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C8 lightweight compute-cost micro-benchmark.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run_compute_cost(Path(args.outputs), repeats=args.repeats)


if __name__ == "__main__":
    main()
