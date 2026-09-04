# Data Availability

The experiments use public datasets whose sources and roles are declared in
`configs/datasets.yaml`.

## Dataset Roles

| Dataset | Source | Experimental role | Redistribution |
| --- | --- | --- | --- |
| Data Centre Warm/Hot Corridor Temperature Prediction | Kaggle | Thermal calibration and warm/hot corridor measurements | Not redistributed; obtain from Kaggle under dataset-specific terms |
| ENACT Edge-Cloud Continuum Telemetry | Zenodo record 18920397 | Workload, CPU/memory and energy telemetry | Not redistributed here; obtain from Zenodo under record-specific terms |
| CuCD-ID | Mendeley Data, version 3 | Attack timing/pattern labels only | Not redistributed here; original dataset is CC BY 4.0 |

CuCD-ID is not used as thermal telemetry. It is used only for anomaly timing and
pattern-level injection.

## Included Artifacts

This repository includes:

- source code and configuration files;
- dataset discovery, download and audit scripts;
- LLM prompt/schema documentation;
- deterministic verifier policy;
- scenario generator and experiment runner;
- execution-trace schema documentation;
- parameter/configuration tables;
- aggregate result files used by the reproduction gate;
- scripts that recompute reported quantities from CSV/JSON outputs.
- the authoritative curated Round 1 evidence under `results/round1_evidence/`;
- the revised Smart Cities manuscript source under `paper/smartcities-round1/`.

The closed-loop additions use a declared first-order illustrative thermal model driven by recorded context. They do not constitute hardware or instrumented-facility validation.

## Not Included

This repository intentionally does not include raw third-party dataset files.
Users must download them from the original providers and comply with their
licenses or terms.

Large per-tick trace archives may be attached to a release or deposited in a
research-data repository. If omitted from Git, they can be regenerated from the
scenario generator once the raw datasets are materialized.
