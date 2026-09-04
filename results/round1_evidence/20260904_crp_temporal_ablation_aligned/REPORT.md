# Aligned temporal-CRP diagnostic replay

This package supersedes `20260904_crp_temporal_ablation` for R2.1. It reuses the same 45 frozen FPR+CRP trajectories and executed actions (10,800 ticks) and compares four estimator diagnostics with the production stable-ID tie policy. It does not rerun or alter the physical plant.

For tick zero the dynamic predictor applies no transition. For tick `t>0`, it predicts observation `t` from estimate `t-1` with ambient, load, and executed fan from row `t-1`. True temperature is never an input. All 45 trajectories satisfy `fan_after[t-1] == fan_before[t]` within `1e-12`.

| Variant | Mean MAE (degC) | Non-converged ticks | Reporter exclusions | Total rounds |
|---|---:|---:|---:|---:|
| temporal only | 0.084507 | 637 | 1,142 | 11,181 |
| social only | 0.079962 | 57 | 2,826 | 11,763 |
| published blend | 0.084417 | 627 | 1,171 | 11,197 |
| social + aligned dynamic innovation | 0.079996 | 22 | 2,973 | 11,796 |

The aligned dynamic variant improves non-convergence relative to the published blend but has slightly worse MAE than social-only and produces more exclusions and rounds. These are mixed diagnostic outcomes, not evidence of universally better closed-loop control.

The legacy-order gate is executed separately only to compare `published_blend` with recoverable saved raw fields. Confidence, excluded counts and rounds match. Exactly 50 estimate mismatches remain, all confined to dominance gaps `<=1e-12`; they are near-equal floating-point cases, not exact cross-platform reproduction. Saved observations and complete flags were unavailable and were not imputed.
