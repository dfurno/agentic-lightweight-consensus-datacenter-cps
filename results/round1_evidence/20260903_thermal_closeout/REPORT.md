# Thermal evidence closeout and hot-start appendix

The original 135 condition-runs were not repeated. Their 135 checkpoint files (32,400 ticks) were audited in place. The only new simulation comprised the pre-specified 18 hot-start condition-runs on frozen commit `d7b1eee49ce1205ace618b43a600ec4e64e0f391`; all completed without runtime failure, LLM, network, or GPU use.

## Audit of existing pilot raw data

For every original checkpoint, the audit verified 240 unique ticks, `next_temperature[k] == temperature[k+1]`, fan continuity, finite required state, fan bounds/slew, the frozen plant equation using innovations deterministically reconstructed from seed/configuration, and exact agreement of IAE, SLA integral, and fan effort with the published summary. All checks passed. `raw_file_manifest.csv` records SHA-256 and size for each of the 135 raw tick files.

The original logs did not store full clean/observed sensor arrays or per-sensor CRP flags; these fields cannot be recovered as original records. Five complete original tick extracts are published (nominal, drift-high, freeze-low, spike-high, and the run with most rejections). The audit reconstructs only plant innovations, explicitly labeled as reconstruction.

The original controller was periodic, not event-triggered. Target-band return onset differed despite universal binary return: means were 1.156 min oracle, 6.178 FPR+CRP, and 7.267 mean. None was recovery from an SLA violation. Freeze-low injection ended at tick 104 exclusive; corrected effective-window diagnostics do not alter frozen primary outcomes. Window alarm is not attacked-sensor localization recall.

## Exploratory 18-run hot-start appendix

All runs start at 33 degC. Clean runs recovered sustainably below the 32 degC SLA at minute 3 for every system, with mean exposure duration 3 min, peak excess 1 degC, and integrated excess 1.919 degC min. The oracle common-bias case is identical to its clean paired run because the oracle controller sees true state.

With a -3 degC common bias on all sensors during ticks 0--59, mean and FPR+CRP systems recovered sustainably at minute 68, eight minutes after bias removal. Mean exposure duration was 61.333 min for both; mean peak excess was 1.375 degC for mean and 1.428 for FPR+CRP; mean integrated excess was 50.275 and 50.073 degC min. There were no censored recoveries or subsequent violations after confirmed recovery.

The common-bias fault greatly worsened tracking: mean IAE was 242.286 degC min for mean and 242.678 for FPR+CRP, versus 111.664 for oracle. It also reduced mean normalized fan effort to 119.3 and 119.6, compared with oracle 140.833, because corrupted measurements delayed cooling. This is adverse evidence about dependence on common measurement quality, not energy savings. FPR+CRP had two total verifier rejections across the three common-bias runs; CRP had no non-convergent ticks in this appendix. Equivalent or adverse results were retained.

Six complete new tick extracts (both cases, seed 11, all systems) include clean/observed sensor values, effective mask, CRP flags/exclusions, runtime payloads and IDs, verifier result, plant state, and persistent fan. All 18 raw logs remain on the server. Pairing between cases/systems, finite state, completeness, and actuator contract passed.

## Interpretation boundary

The appendix is a deliberately narrow stress boundary outside the original 3/9-sensor attack regime. It shows that a common-mode measurement error can delay safety recovery for measurement-dependent controllers, while an oracle unaffected by measurement error recovers promptly. It does not establish representative attack prevalence, estimator-only causality, general robustness, hardware realism, or statistical significance. No tuning, CRP modification, original-pilot rerun, or extra scenario was performed.
