#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revision.artifacts import RevisionPaths, write_text


CPU_PLAN = """# CPU extension plan

This script is intentionally conservative in this release. It stages the commands
for new seeded CPU campaigns, but it does not silently alter the existing scenario
grid or architecture.

Recommended remote execution after reproduction gate passes:

1. Add at least 9 new seeds to `configs/experiments.yaml` under the `full` suite.
2. Run `make run-real-full-resume PYTHON=.venv/bin/python`.
3. Re-run `make revision-phase-b PYTHON=.venv/bin/python`.

For additional deterministic baselines requested by reviewers, implement them as
new consensus methods inside `src/consensus/baselines.py` and add them to
`configs/experiments.yaml::methods`, then run the same real full-resume target.
"""


AGENTIC_PLAN = """# Agentic extension plan

Run this only after the LLM server responds at `http://127.0.0.1:8000/v1/models`
and the reproduction gate passes.

Recommended budget-safe path:

1. Keep the current realtime event-trigger policy.
2. Add a representative stratified suite covering every attack type, beta_a in
   {0, 0.2, 0.4}, and sensor counts {5, 13}.
3. Run a deterministic rule-based supervisor on the same triggers as the LLM.
4. Compare against the existing LangGraph/Gemma results in:
   `outputs/supervisor_comparison.csv`.

This placeholder exists to prevent accidental expensive GPU runs before the
review-gate artifacts have been inspected.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage major-revision extension-run instructions.")
    parser.add_argument("--mode", choices=["cpu", "agentic-subset"], required=True)
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    out = Path(args.outputs)
    if args.mode == "cpu":
        write_text(out / "revision_cpu_extension_plan.md", CPU_PLAN)
    else:
        write_text(out / "revision_agentic_extension_plan.md", AGENTIC_PLAN)


if __name__ == "__main__":
    main()
