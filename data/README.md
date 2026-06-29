# Data Policy

This repository does not redistribute third-party raw datasets.

Place locally obtained dataset files under `data/raw/` only after accepting the
terms of the original providers. The expected sources are documented in
`configs/datasets.yaml` and audited by:

```bash
make discover-data
make audit-data
```

Primary sources:

- Kaggle Data Centre Warm/Hot Corridor Temperature Prediction: thermal
  calibration and hot/warm corridor measurements.
- ENACT Edge-Cloud Continuum Telemetry, Zenodo record 18920397: workload,
  CPU/memory and energy telemetry.
- CuCD-ID, Mendeley Data, CC BY 4.0: attack timing/pattern labels only, not
  thermal measurements.

The pipeline is intentionally strict: cyber datasets are never treated as
thermal telemetry, and substitutions must be explicitly marked in the dataset
manifest and paper.
