# CRP evaluation — ready-to-run package (to execute on the host that holds the traces)

**Why this is here and not already run:** the `major_revision_final_outputs_*.tar.gz`
bundle ships the per-tick **`*.metadata.json`** and the *aggregate* Phase-B metrics, but
**not** the per-tick trace CSVs (`results/traces/*.csv`) nor the raw datasets
(`data/raw/...`). The CRP replay needs the per-tick sensor readings + attack labels, so it
must run where those CSVs exist (the GPU/experimentation host). The code below is verified
to run (synthetic self-test passed: it flags/excludes a bias sensor and tracks the truth).

## What it does
Faithful implementation of Methodology Sec. 4.5 (CRP) wrapping the single-round FPR-OWA:
social proximity / zone consensus degree `C_z`, temporal self-consistency `k_i` from an
EWMA short-horizon prediction, combined reliability `r_i = α_crp·p_i + (1−α_crp)·k_i`, and
feedback by **delayed inclusion** (exclude reporters with `r_i < τ_r`, up to `max_rounds`,
forwarding a degraded-confidence snapshot otherwise). Re-sampling feedback is approximated by
delayed inclusion/down-weighting (documented replay assumption).

It replays every trace and emits, for **CRP vs single-round FPR-OWA** on the *same* scenarios:
- `crp_estimation_per_scenario.csv` — MAE/RMSE/median/P95/max AE per scenario per method.
- `crp_anomaly_detection_metrics.csv` — AUROC/AUPRC/precision/recall/F1/FPR/detection-delay
  per method × attack × ratio (directly comparable to `anomaly_detection_metrics.csv`).
- `crp_convergence.csv` — rounds-to-converge, converged rate, mean `C_z`, feedback exclusions.
- console summary (global + per-attack MAE; mean AUROC over attacked ratios; convergence).

## How to run (on the host with `results/traces/*.csv`)
```bash
# place crp.py at:        <repo>/src/consensus/crp.py
# place eval_crp.py at:   <repo>/scripts/eval_crp.py
cd <repo>
# main run (default alpha_crp=0.5, M_z=4.0) -> writes crp_*.csv (no tag)
python3 scripts/eval_crp.py results outputs 0.5 4.0
#                              ^results ^outputs ^alpha_crp ^M_z
```
Sanity gate: the recomputed **fpr_owa** global MAE must match the manuscript (≈1.6171);
the script prints it. If it matches, CRP numbers are apples-to-apples.

## Recommended sweep (cheap, CPU, strengthens the Sec. 4.5 study)
A 5th argument is an output **tag** so runs do not overwrite each other:
```bash
for ac in 0.3 0.5 0.7; do for mz in 2.0 4.0; do
  python3 scripts/eval_crp.py results outputs $ac $mz acrp${ac}_mz${mz}
done; done
```
This yields `crp_*_acrp{ac}_mz{mz}.csv` for each config and lets us pick a principled default
and report the sensitivity reviewers expect.

## Parameters (defaults; document final choices in the paper / params table)
`α_crp=0.5` (social/temporal blend), `M_z=4.0` (zone normalization), `τ_c=0.7` (zone
consensus accept), `τ_r=0.5` (reporter reliability), `L=3` (persistence), EWMA `λ=0.3`,
`max_rounds=3`, `min_sensors=3`. (Quantifier `(a,b)` is read per-scenario from the trace name.)

## What to send back
The three CSVs above + the console summary. I will then (a) validate the fpr_owa sanity gate,
(b) decide CRP positioning vs FPR-OWA on the measured numbers, and (c) integrate into the
revised Methodology/Results/Discussion and the repro bundle.
