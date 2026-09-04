# Smart Cities Round 1 Release Manifest

Release identifier: `smartcities-round1-2026-09-04`. The initial commit `a935bb0` remains unchanged as historical provenance; its ASOC-oriented label and draft are superseded by this release.

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
- `docs/round1/`: control contract, formal ordering note and frozen protocol.
- `paper/smartcities-round1/`: revised MDPI manuscript source and compiled PDF.
- `results/round1_evidence/`: curated Round 1 manifests, hashes, tables and adverse outcomes.
- `REPRODUCIBILITY.md`: commands for reproducing reported quantities.
- `DATA_AVAILABILITY.md`: source and redistribution policy for datasets.
- `LICENSE`: MIT license for original software.

Excluded:

- raw third-party datasets under `data/raw/`;
- local virtual environments;
- Python and LaTeX build caches;
- intermediate archives and remote execution logs;
- large trace CSV archives unless attached separately as release assets.
- reviewer reports, response letters, editorial correspondence and private workflow files.

`CITATION.cff` contains the public repository URL and Round 1 release identifier. Add the article DOI when it becomes available.
