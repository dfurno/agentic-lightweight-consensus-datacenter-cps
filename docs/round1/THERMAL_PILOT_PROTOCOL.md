# Frozen compact closed-loop thermal pilot protocol

This document consolidates the executed protocol and the binding decisions in `THERMAL_PILOT_REVIEW_DECISIONS.md`. Historical drafts remain recoverable in Git. The experiment compares complete systems; it is not an estimator-only causal study. Parameters are illustrative, not calibrated from standardized historical temperatures or hardware.

## Plant and control

The one-zone simulation has 240 ticks at 60 seconds, true temperature in degC, a persistent normalized fan, and fixed 24 degC setpoint. The plant is `T_(k+1)=T_k+0.08*(T_amb-T_k)+q-0.80*u_f+epsilon`. It starts at 30 degC with ambient 26 degC, load 0.70 degC/step, and process-noise SD 0.02 degC. Ticks 60--119 add 1 degC ambient and 0.15 degC/step load. The state coefficient is 0.92; nominal 30 degC equilibrium fan is 0.475.

The controller is periodic (`always_supervise=True`): it is invoked every tick, not triggered by anomaly count. It targets 30 degC with 0.30 degC hysteresis, changes fan by 0.1, and decreases fan only after a three-tick low-temperature dwell. Fan bounds are [0,1] and maximum allowed slew is 0.2. `anomaly_count` is logged but does not schedule control; confidence remains an admission input.

The pilot-only verifier preserves freshness, confidence, zone, binding, emergency, and actuator checks but changes the historical outcome rule. At/below the 32 degC SLA, predicted next temperature must remain at/below SLA. Above SLA, cooling must improve predicted next temperature over maintaining current fan by at least 0.01 degC. At/above 38 degC an alarm is required and does not cool. Prediction uses post-command fan and gain 0.88 degC/step, a fixed +10% mismatch from the plant. This counterfactual rule is not a physical-safety guarantee and does not change historical verifier claims.

## Systems, attacks, and pairing

The systems are oracle true-state estimate plus baseline diagnostics, arithmetic mean plus the same baseline diagnostics, and FPR+CRP using actual CRP aggregate, confidence, and persistent flags. Nine homogeneous sensors have paired 0.20 degC noise; three reproducibly selected channels are attacked. Baseline flags use deviation from median above 1.5 degC. CRP runs once per branch tick with persistent state: alpha=0.3, beta=0.8, alpha_crp=0.5, M_z=4 degC, tau_c=0.7, tau_r=0.5, EWMA lambda=0.3, persistence=3, max rounds=3, and min sensors=3. Non-convergence forwards degraded confidence and is retained.

Seeds 11, 42, and 101 each contain nominal plus bias, drift, replay, freeze, repeated alternating spike, scaling, and stuck attacks at two frozen intensities: 45 scenarios and 135 condition-runs. The nominal attack window is ticks 84--143. Effective injection differs only for freeze-low, which changes readings at ticks 84--103 and ends at tick 104 exclusive; diagnostic timing must use this effective mask. Exogenous innovations and attacked-channel draws are paired, while each system evolves its own action-dependent trajectory.

An “attack detected” result means at least one tick with `anomaly_count>0` during the effective injection window. It is a window-level alarm, not attacked-sensor localization recall. Pre-attack and legitimate-transition alarms have separate denominators.

## Outcomes and boundaries

IAE is relative to 30 degC in degC min. Report true-temperature duration, peak, and positive integral above SLA; sustained ten-tick return within 30 +/- 0.5 degC after the effective attack end; normalized fan-command integral; total variation; estimator error; alarm windows and missed alarms; verifier decisions/fallback; actuator invariants; CRP rounds, exclusions, convergence, confidence, and measured CPU callback latency. Fan effort is not energy.

Target-band return is distinct from safety recovery after an SLA violation. In the executed 135-run pilot no true state exceeded SLA: binary target return was universal but its onset time varied, while safety recovery was unobserved. The hot-start appendix is specified separately in `THERMAL_CLOSEOUT_PROTOCOL.md`.

The unit of comparison is a paired seed/scenario/system run. Shared innovations and repeated scenarios induce dependence; ticks are not independent replicates. Results are descriptive feasibility evidence, not hardware validation, production safety, statistical significance, estimator-only causality, or LLM value. Runtime failure stops execution; CRP non-convergence, missed detection, non-recovery, saturation, equivalence, and worse results are retained outcomes.
