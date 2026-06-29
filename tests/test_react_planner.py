from src.agents.react_planner import ReactPlanner


def test_extracts_fenced_json_after_reasoning_text() -> None:
    raw = """thought
```json
{
  "action_type": "increase_fan_speed",
  "target_zone": "zone-0",
  "magnitude": 0.25,
  "duration": 300,
  "reasoning_summary": "Cooling required.",
  "consensus_timestamp": "2026-06-10T15:57:07.224744+00:00",
  "used_consensus": true
}
```"""

    extracted = ReactPlanner._extract_json_object(raw)

    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert '"action_type": "increase_fan_speed"' in extracted


def test_extracts_balanced_json_with_braces_inside_string() -> None:
    raw = """Planner output:
{
  "action_type": "maintain",
  "target_zone": "zone-0",
  "magnitude": 0.0,
  "duration": 300,
  "reasoning_summary": "Value contains {braces} but is still valid.",
  "consensus_timestamp": "2026-06-10T15:57:07.224744+00:00",
  "used_consensus": true
}
done"""

    extracted = ReactPlanner._extract_json_object(raw)

    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert "done" not in extracted
