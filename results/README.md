# Result Files

This directory contains reproducibility result files used by the manuscript and
the revision analyses.

Core files:

- `metrics.csv`: scenario-level estimation metrics for consensus and baseline
  methods.
- `metrics_by_scenario.csv`: grouped scenario summaries.
- `safety_metrics.csv`: verifier and agentic-safety outcomes.
- `latency_metrics.csv`: consensus and planner/verifier timing metrics.
- `token_usage.csv`: LLM and verifier call counts.
- `realtime_metrics.csv`: event-triggered realtime metrics.
- `realtime_safety_metrics.csv`: realtime verifier/safety outcomes.
- `dataset_discovery.*`, `data_audit.*`, `dataset_downloads.json`: dataset
  discovery, download and audit records.
- `config_snapshot.yaml`, `realtime_config_snapshot.yaml`: configuration
  snapshots used for the reported runs.

Per-tick trace CSV files can be regenerated from the public datasets and the
scenario generator. Metadata JSON files document the trace source and run
configuration. Large trace archives should be attached as GitHub Releases or
Zenodo artifacts rather than committed directly to Git.
