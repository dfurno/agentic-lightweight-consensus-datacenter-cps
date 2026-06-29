# GitHub Release Manifest

This directory is a cleaned, GitHub-ready reproducibility repository.

Included:

- `src/`: source code for consensus, agents, simulation, data audit and
  evaluation.
- `scripts/`: experiment, reproduction, revision and serving scripts.
- `configs/`: dataset, experiment, LLM, SLA and simulation configuration.
- `tests/`: unit tests.
- `paper/`: LaTeX snippets, generated tables/figures and report material.
- `results/`: aggregate CSV/JSON result files and trace metadata.
- `docs/`: prompt/schema, verifier-policy and trace-schema documentation.
- `REPRODUCIBILITY.md`: commands for reproducing reported quantities.
- `DATA_AVAILABILITY.md`: source and redistribution policy for datasets.
- `LICENSE`: MIT license for original software.

Excluded:

- raw third-party datasets under `data/raw/`;
- local virtual environments;
- Python and LaTeX build caches;
- intermediate archives and remote execution logs;
- large trace CSV archives unless attached separately as release assets.

Before publishing, update `CITATION.cff` with the final GitHub URL and, if
available, DOI or manuscript citation details.
