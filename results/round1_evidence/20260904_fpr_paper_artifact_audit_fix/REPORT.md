# Extended paper-facing stable-ID audit

This package supersedes the paper-facing conclusions in `20260904_fpr_tie_closeout`. It reuses historical traces only; no physical trajectory, LLM inference, GPU work, sweep, or tuning was run. The raw `all_derived_metrics.csv` remains under `runs/round1/20260904_fpr_paper_artifact_audit_final/`; the versioned package contains curated mappings and changed-case/tick records.

`paper_artifact_mapping.csv` maps artifact, cell/series, source family, sample count, before/after values, displayed precision, and whether the displayed value changes. Coverage includes the 1,890-scenario main campaign, corrected single-round/CRP rows, 1,909-scenario ablation, and the FPR-OWA dependency in the ten-seed/6,300-scenario robustness extension.

The main FPR-OWA global row changes at four decimals: MAE `1.6171 -> 1.6173`, RMSE `1.7071 -> 1.7073`, median AE `1.5437 -> 1.5438`, P95 AE `2.4847 -> 2.4849`, and max AE `2.8345 -> 2.8347`. The spike heatmap MAE changes at three decimals (`1.564 -> 1.565`); the attack-ratio MAE series changes at 0.3 (`1.6255 -> 1.6259`) and 0.4 (`1.6942 -> 1.6946`); and the five-sensor table MAE changes (`1.8079 -> 1.8084`). The corrected CRP P95 cell changes `2.4230 -> 2.4229`; its MAE remains `1.6116` at four decimals.

For the ten-seed robustness dependency, FPR-OWA MAE `1.614117 -> 1.614190`, RMSE `1.704658 -> 1.704732`, P95 `2.474742 -> 2.474725`, and between-seed MAE SD `0.007421 -> 0.007337`; none changes at the table's three-decimal precision. The full-FPR ablation MAE `1.616169 -> 1.616337` also remains `1.616` at three decimals. Rankings and scientific conclusions do not change, but the affected displayed values listed above require regeneration during the later manuscript phase.

Exact-tie counts use exact computed-score equality. Stable reporter IDs deterministically extend those exact ties. This does not establish cross-platform identity for near-equal floating-point dominance scores: the separate R2.1 legacy gate retains 50 mismatches confined to gaps `<=1e-12`. Reproducibility claims must preserve that boundary.
