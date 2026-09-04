# Shared control-step contract

One step validates the current zone snapshot and evaluates trigger reasons. A ready, correctly bound supervisory proposal may replace the fast-path action; otherwise exactly one deterministic fast-path action is selected. The verifier evaluates that selected action against the same immutable snapshot and current persistent actuator state. Acceptance creates a process-local, payload-bound, single-use capability; only successful consumption can reach the simulated actuator.

Planner work may remain pending while later ticks execute the fast path. A late proposal retains its source snapshot and is rejected when that snapshot is superseded. Planner/verifier deadline and exception paths use local fast-path or hold/alarm behavior without calling the failed component again. Hold/alarm is procedural fallback under the in-process trust model; it is not a claim of physical thermal safety.

The snapshot interface accepts the current FPR-derived fields and can later accept equivalent CRP-derived fields. The present runtime integration does not execute CRP.
