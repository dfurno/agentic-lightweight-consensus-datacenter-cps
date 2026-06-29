# Deterministic Verifier Policy

The verifier is implemented in `src/agents/verifier.py` and configured by
`configs/sla.yaml`.

The verifier is non-bypassable for supervisory actions. It checks:

- consensus freshness;
- minimum confidence;
- thermal SLA and emergency thresholds;
- actuator bounds;
- slew-rate constraints;
- predicted cooling effect where applicable;
- policy constraints encoded in the verifier configuration.

If a proposed action is late, unsafe, invalid, or inconsistent with the policy,
the verifier rejects it and the deterministic fallback path remains active.

The verifier is the source of the reported safety guarantee that unsafe LLM
actions are not executed in the verifier-gated experiments.
