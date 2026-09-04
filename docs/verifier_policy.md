# Deterministic Actuation-Verifier Policy

The verifier is implemented in `src/agents/verifier.py` and configured by
`configs/sla.yaml`.

In the tested shared runtime, the verifier is the process-local command gate for supervisory actions. It checks:

- consensus freshness;
- minimum confidence;
- thermal SLA and emergency thresholds;
- actuator bounds;
- slew-rate constraints;
- predicted cooling effect where applicable;
- policy constraints encoded in the verifier configuration.

If a proposed action is late, unsafe, invalid, or inconsistent with the policy,
the verifier rejects it and the deterministic fallback path remains active.

The 24-case controlled-time matrix under `results/round1_evidence/20260903_runtime_cpu_fix/` validates this path with a simulated actuator. It does not prove cross-process or system-wide non-bypassability and does not provide PKI, identity authentication, RBAC/IAM, operating-system isolation, live asynchronous LLM validation, or hardware safety certification. See `docs/round1/CONTROL_CONTRACT.md`.
