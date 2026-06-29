# LLM Prompt Template and JSON Schema

The LangGraph/Gemma supervisor is implemented in
`src/agents/react_planner.py`. It is a sparse, event-triggered supervisory
component and not part of the safety-critical per-tick control path.

## Prompt Template

The planner receives a consensus snapshot containing:

- temperature estimate;
- confidence;
- anomaly count;
- timestamp;
- target zone;
- agentic variant;
- refinement cycle;
- verifier feedback.

The system instruction requires a strict JSON-only control-action proposal. The
planner cannot execute physical actions.

## Control Action Schema

The machine-readable output is defined in `src/agents/schemas.py`:

```json
{
  "action_type": "increase_fan_speed | decrease_setpoint | maintain | raise_alarm",
  "target_zone": "string",
  "magnitude": "float >= 0",
  "duration": "integer > 0",
  "reasoning_summary": "short string",
  "consensus_timestamp": "string",
  "used_consensus": "boolean"
}
```

Invalid or non-JSON responses are rejected before actuation. In realtime mode,
invalid LLM output triggers deterministic fallback and the fast path continues.

## Action-Only Compatibility

The reported safety guarantees do not depend on verbose LLM reasoning. The same
architecture is compatible with action-only or tool-only invocation, where the
LLM is constrained to emit only structured tool calls or machine-readable
control proposals. This is a deployment optimization rather than a requirement:
all proposals must pass the deterministic, non-bypassable verifier.
