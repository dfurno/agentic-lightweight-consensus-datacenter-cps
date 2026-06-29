#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.supervisor_comparison import run_rule_supervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C3 rule-based supervisor vs LangGraph/Gemma comparison.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional development limit; omit for the remote full C3 run.")
    args = parser.parse_args()
    run_rule_supervisor(results_dir=Path(args.results), outputs_dir=Path(args.outputs), max_traces=args.max_traces)


if __name__ == "__main__":
    main()
