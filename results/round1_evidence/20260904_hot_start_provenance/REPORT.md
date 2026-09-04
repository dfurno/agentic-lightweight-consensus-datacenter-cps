# Hot-start provenance audit

All 18 existing hot-start condition runs (4,320 ticks) were audited in place; none was rerun. Temperature and fan continuity, the frozen plant equation with seed-reconstructed process noise, finite state, terminal state, summary metrics, sustained recovery convention, pairing, actuator bounds/slew, and runtime record completeness all pass. Derived counts are zero actuator-contract violations and zero runtime failures.

The future closeout runner now derives these two counts from tick logs instead of writing literal zeros. Raw outputs and the already published closeout evidence remain unchanged. `raw_manifest.csv` contains SHA-256 and byte size for all 36 raw tick/summary artifacts. `provenance.json` records the command, commit, configuration hashes, interpreter and dependencies.

The audit does not turn the three-seed illustrative hot-start appendix into hardware validation or a statistical superiority study. Under common-mode bias, mean and FPR+CRP remain practically similar and both recover later than the oracle.
