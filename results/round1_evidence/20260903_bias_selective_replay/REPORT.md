# Selective bias replay and historical-output recomposition

## Scope and provenance

This run corrects the parser-affected `full/bias/beta=0.7` slice of the original three-seed campaign without generating traces or invoking an LLM. Six unique historical CRP configurations were replayed: alpha_crp in {0.3, 0.5, 0.7} and M_z in {2.0, 4.0}. The unsuffixed historical files were verified as byte-identical duplicates of alpha_crp=0.5, M_z=4.0 and were reused as aliases.

Each configuration used a hash-locked manifest containing 135 affected scenarios and six controls, for 141 scenarios and 33,840 ticks. The affected design is seed {11,42,101} x five attack ratios x three noise levels x three sensor counts. The ten-seed extension was not replayed.

## Validation and recomposition

Manifest mode now rejects configuration disagreement, duplicate scenarios or paths, invalid hashes, incomplete sensor/label schema, non-finite values, unexpected tick counts, non-empty destinations, and output-cardinality mismatch. The repository suite passes 31 tests.

For every CRP configuration, all 141 replayed FPR-OWA rows matched `results/metrics.csv` across five error metrics with maximum absolute difference 8.88e-16. The four convergence quantities matched their historical rows exactly (maximum difference 0.0), confirming that OWA beta changes the final estimate but not CRP convergence, exclusions, or diagnostic state in the current implementation.

The recomposition replaced exactly 270 estimation rows per configuration, corrected beta metadata from scenario names, and preserved 3,780 estimation rows and 1,890 convergence rows. No historical row was added or removed.

## Products and manuscript-facing values

The table reports the alpha_crp=0.5, M_z=4.0 configuration used by the submitted CRP table. Manuscript edits are intentionally deferred.

All campaign-wide error values retain the historical aggregation convention: they are means of scenario-level summaries. Thus P95 AE is the mean of 1,890 within-scenario 95th percentiles and max AE is the mean of 1,890 within-scenario maxima; neither is a pooled-tick percentile or the absolute campaign maximum. RMSE is likewise averaged across scenario-level RMSE values.

| Product/value | Historical | Corrected | Rounded manuscript value |
|---|---:|---:|---:|
| FPR-OWA MAE in CRP table | 1.616633 | 1.617131 | 1.6166 -> 1.6171 |
| FPR-OWA P95 AE | 2.480700 | 2.484669 | 2.4807 -> 2.4847 |
| FPR-OWA max AE | 2.828682 | 2.834537 | 2.8287 -> 2.8345 |
| CRP MAE | 1.611088 | 1.611571 | 1.6111 -> 1.6116 |
| CRP P95 AE | 2.419366 | 2.422970 | 2.4194 -> 2.4230 |
| CRP max AE | 2.770965 | 2.775261 | 2.7710 -> 2.7753 |
| Mean rounds / convergence / exclusions | unchanged | unchanged | no numeric change |

The main six-aggregator `metrics.csv` already used the correct beta and therefore its headline FPR-OWA MAE 1.6171 does not change. Only the parser-derived CRP replay copy moves from 1.616633 to the same 1.617131 value.

For the 1,909-scenario ablation, exactly 270 beta-dependent rows were replaced. Reliability-only and dominance-only are numerically unchanged. Mean MAE changes from 1.580206 to 1.567739 for OWA-only and from 1.615675 to 1.616169 for full FPR-OWA. The historical text rounds these to 1.580 and 1.616; OWA-only must therefore change to 1.568, while full FPR-OWA remains 1.616 at three decimals.

Calibration was recomposed from all 16,528,320 historical diagnostic records while updating 1,166,400 records associated with 32,400 affected scenario-ticks. FPR-OWA global Spearman confidence/error correlation changes from -0.228320 to -0.227195 (both round to -0.23). Diagnostic scores and flags were not reclassified as beta-dependent observations because they are invariant to OWA weights.

## Files deliberately unchanged

`results/metrics.csv`, `owa_ab_split.csv`, `mae_vs_beta.csv`, anomaly-score/flag summaries, and convergence numeric fields remain historical evidence. Their computation either used the correct parameters originally or is independent of OWA beta. No ten-seed extension, LLM, safety, realtime, or manuscript artifact was modified.

## Limitations and unavailable historical information

The historical project has no Git metadata, so an original source-code commit cannot be recovered; its relevant exported source was previously verified byte-for-byte against the 91-file baseline. Exact historical shell commands and runtime environment for every CRP sweep are not recoverable from the CSVs. Input files, current commands, current code commits, parameters, sample counts, and hashes are recorded here. The complete 232 MB compressed diagnostic record file remains in backed-up server storage and is referenced by hash rather than duplicated in Git.
