# Agentic Lightweight Consensus — LaTeX project

> Historical pre-Round-1 draft retained for provenance. The current Smart Cities manuscript is in `paper/smartcities-round1/`.

Self-contained LaTeX project for the journal paper draft
*Agentic Lightweight Consensus for Resilient Monitoring and Control of
Data-Centre Cyber–Physical Systems*.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Complete manuscript (single file, embedded bibliography) |
| `elsarticle.cls` | Elsevier class (bundled, LPPL licence) so the project compiles even without `texlive-publishers` |
| `figures/architecture_overview.pdf` | Working vector version of the architecture figure (Fig. 1) |
| `figures/arch_source_reference.drawio` | Editable draw.io source from the companion FHIR paper, to be adapted for the final figure |
| `Makefile` | `make` builds the PDF, `make clean` removes build artefacts |

## Build

```bash
pdflatex main && pdflatex main
# or simply
make
```

No BibTeX run is required: the bibliography is embedded as
`thebibliography` for maximum portability across journal templates.
The project also compiles on Overleaf out of the box.

## Status and TODOs

All scientific sections are complete and populated from the real
experimental campaign: Introduction, Related Work, Architecture/System
Model, Consensus Methodology, Verifier-Gated Agentic Control,
Experimental Setup, Results (RQ1–RQ4), Discussion, Limitations, and
Conclusions. Only administrative front/back-matter items remain as red
`[TODO: …]` markers (CRediT roles, repository/DOI links, acknowledgements,
funding). Search for `\todo{` to enumerate them.

Tables (`tables/*.tex`) and figures (`figures/*.pdf`) are generated from
the verified result CSVs; regenerate with the scripts noted in the paper
header. Numbers in the prose were cross-checked against
`metrics.csv`, `safety_metrics.csv`, `latency_metrics.csv`, and the two
`realtime_metrics.csv` runs.

Floats follow a strict no-redundancy rule: each result is shown once,
either as a table or a figure, never both. The paper keeps three figures
(architecture; attack$\times$method MAE heatmap; per-decision latency)
and seven tables (positioning, attack taxonomy, datasets, consensus
performance, sensor redundancy, agentic safety, real-time comparison).
A few superseded `figures/*.pdf` from earlier drafts remain on disk but
are no longer referenced by `main.tex`.

Compile with the `review` class option removed (and `preprint` kept)
for single-spaced output; switch to the target journal's template by
replacing the `\documentclass` line and the frontmatter only — the body
uses no Elsevier-specific commands.

## Reference policy

Every bibliography entry was validated against Crossref/arXiv (title,
authors, venue, volume/pages, DOI) at the time of writing. No entry is
invented; conference papers without DOIs (PBFT/OSDI, ICLR/NeurIPS) are
cited by their official proceedings.
