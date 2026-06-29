#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.closing_experiments import run_safety_adversarial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C4 verifier adversarial and failure-mode checks.")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    run_safety_adversarial(Path(args.outputs))


if __name__ == "__main__":
    main()
