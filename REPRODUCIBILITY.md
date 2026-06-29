# Reproducibility Instructions

This repository is organized so that reported manuscript quantities are derived
from CSV/JSON artifacts and can be recomputed by scripts.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make test
```

For CPU-only reproduction of reported tables and diagnostic checks:

```bash
make revision-inventory PYTHON=.venv/bin/python
make revision-reproduce PYTHON=.venv/bin/python
make revision-phase-b PYTHON=.venv/bin/python
```

The reproduction gate writes:

```text
outputs/reproduction_check.csv
```

Every row should have `status=pass` for manuscript values that are already
encoded in the gate.

## Dataset-Aware Full Run

Raw datasets are not redistributed. After obtaining the datasets from their
original sources and placing them under `data/raw/`, run:

```bash
make discover-data PYTHON=.venv/bin/python
make audit-data PYTHON=.venv/bin/python
make run-real-full PYTHON=.venv/bin/python
make paper PYTHON=.venv/bin/python
```

This recomputes the full experiment from materialized data. The LLM supervisor
requires an OpenAI-compatible local endpoint; see `README.md` and
`configs/llm.yaml`.

## Event-Triggered Realtime Run

Start the model server first, then run:

```bash
make run-real-realtime-resume PYTHON=.venv/bin/python
make paper PYTHON=.venv/bin/python
```

The realtime design keeps the safety-critical path deterministic and invokes
the LLM supervisor sparsely. LLM proposals are never executed directly; they
must pass the deterministic verifier.

## Revision/Robustness Analyses

The following scripts generate additional reviewer-response artifacts:

```bash
make revision-supervisor-comparison PYTHON=.venv/bin/python
make revision-closing-experiments PYTHON=.venv/bin/python
make revision-seed-baselines PYTHON=.venv/bin/python
python scripts/eval_crp.py results outputs 0.5 4.0
```

`revision-seed-baselines` is CPU-only and evaluates ten seeds together with
robust estimation baselines: Huber location, Tukey biweight, and robust Kalman.

## Large Trace Artifacts

The committed CSV result files are sufficient for the reproduction gate and
reported aggregate quantities. Per-tick trace CSV archives may be distributed
through GitHub Releases or Zenodo to avoid bloating the Git history.
