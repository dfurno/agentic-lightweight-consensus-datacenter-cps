# Runtime contract correction: CPU evidence report

Run ID: `20260903_runtime_cpu_fix`. Source commit: `60c06e277191eda4d3cd78a5503de7e8eb240b22`, verified against `git rev-parse HEAD` by the matrix runner. This is a CPU software-validation run; it contains no thermal-pilot execution and no LLM inference.

## Reproduction and correction

Before modification, the six supplied regression tests failed: three invalid but actuating snapshots changed fan state; a pending request was overwritten five times; a consumed authorization became valid after re-emission; and a planner mutated the caller's snapshot. After correction, the complete repository suite passed 52/52.

The runtime now copies and hashes the authoritative snapshot before callbacks, blocks observation-dependent actuation for invalid snapshots, binds proposals to that snapshot, preserves a pending request and its original deadline, distinguishes timeout and supersession, and uses a monotonic authorization-emission identity. Events record request/snapshot/proposal identities and payloads, explicit verifier acceptance and rejection reasons, actual action source, before/after actuator state, and planner/verifier/step timings.

## Updated matrix

The updated controlled-time matrix passed 24/24 cases and emitted 34 runtime events. It includes high-temperature future, stale, wrong-zone, and non-finite-confidence snapshots; malformed planner parsing inside the invoked callback; proposal timestamp mismatch; delayed completion while the planner remains available; supersession; token re-emission; callback mutation; timeout and exception paths; and persistent actuation limits. The independent oracle checks actuator bounds and per-event slew, not only final state.

All invalid actuating snapshots produced a local `raise_alarm` with unchanged actuator state and without verifier acceptance. The delayed proposal was called once, completed on its original request ID, and changed fan speed from 0.5 to 0.6. The malformed planner response continued through the verified local fast path. The timestamp-mismatched proposal and planner-mutated proposal were rejected and replaced by a local alarm. No matrix case failed.

## One-trace integration smoke test

The explicitly selected input is `realtime_seed42_bias_0p0_noise0p25_s13_a0p3_b0p8.csv` (SHA-256 `8def20740657b9a3579ff55af59d18d5aae0a941907d16eb32c585565ce8d801`). Its 240 ticks produced nine deterministic-supervisor calls and acceptances, 231 accepted fast-path actions, no fallback, and zero independently observed actuator-contract violations. Mean measured planner callback latency was `1.70555e-05 s`; mean complete loop latency was `0.311610 ms` in this run.

Thermal safety was not measured by this fixed-trace smoke test and is stored as missing with the label `not_measured_thermal_safety`, rather than as an observed zero. The historical LLM comparison was explicitly disabled. This trace is a wiring and logging smoke test only; it is not paired with the 18 historical LLM traces and supports no LLM equivalence or superiority claim.

## Commands

```text
/root/research/iot-agentic-lightweight-consensus/.venv/bin/python -m pytest -q -p no:cacheprovider
/root/research/iot-agentic-lightweight-consensus/.venv/bin/python scripts/run_runtime_cpu_matrix.py --matrix revision/runtime/runtime_cpu_matrix_fix.json --output runs/round1/20260903_runtime_cpu_fix/matrix --commit 60c06e277191eda4d3cd78a5503de7e8eb240b22
MPLCONFIGDIR=/tmp/matplotlib-runtime-fix /root/research/iot-agentic-lightweight-consensus/.venv/bin/python scripts/revision_supervisor_comparison.py --results /root/research/iot-agentic-lightweight-consensus/results --outputs runs/round1/20260903_runtime_cpu_fix/smoke --max-traces 1 --no-llm-comparison
```

## Limitations and superseded conclusions

This run supersedes the earlier runtime run's claims that invalid-snapshot handling, pending requests, token re-emission, supervisor latency, fallback attribution, and `unsafe_actions_executed=0` had been validated. The earlier files remain unchanged as evidence of the earlier implementation. Controlled delay adapters do not establish live asynchronous model behavior. The smoke trace has attack ratio zero, uses standardized historical data, and does not provide physical temperature, SLA, closed-loop, or LLM evidence. The thermal pilot protocol was prepared separately but not executed.
