# Dataset Discovery

## ENACT (enact)
- Role: `workload_power`
- Manifest status: `primary`
- Validation state: `candidate_validated`
- Source URL: https://zenodo.org/records/18920397
- License/terms: record_specific
- Download method: `local_zip_or_zenodo`
- Target directory: `data/raw/enact`
- Notes: ENACT Edge-Cloud Continuum Telemetry; node/pod telemetry includes CPU, memory, and Energy (watts).

## Data Centre Warm Channel Temperature Prediction (kaggle_hot_corridor)
- Role: `thermal_calibration`
- Manifest status: `primary`
- Validation state: `candidate_validated`
- Source URL: https://www.kaggle.com/datasets/mbjunior/data-centre-hot-corridor-temperature-prediction
- License/terms: kaggle_dataset_specific
- Download method: `local_zip_or_kaggle_api`
- Target directory: `data/raw/kaggle_hot_corridor`
- Notes: Primary thermal dataset. final_dataset_std.csv contains P_cu-*, T_MEAS-* and TLHC columns.

## CuCD-ID (cucd_id)
- Role: `anomaly_timing`
- Manifest status: `primary`
- Validation state: `candidate_validated`
- Source URL: https://data.mendeley.com/datasets/7n2d42pm3n/3
- License/terms: CC BY 4.0
- Download method: `local_zip_or_mendeley`
- Target directory: `data/raw/cucd_id`
- Notes: CubeSat intrusion dataset. Use labels only for attack timing/pattern injection, not as thermal telemetry.

## Google Cluster Data (google_cluster_data)
- Role: `workload_power`
- Manifest status: `substitute`
- Validation state: `candidate_validated`
- Source URL: https://github.com/google/cluster-data
- License/terms: CC-BY
- Download method: `git_or_manual`
- Target directory: `data/raw/enact/google_cluster_data`
- Notes: Official repository documents Borg workload traces and May 2019 power traces.

## Building Data Genome Project 2 / ASHRAE GEPIII (building_data_genome_2)
- Role: `thermal_calibration`
- Manifest status: `substitute`
- Validation state: `candidate_validated`
- Source URL: https://github.com/buds-lab/building-data-genome-project-2
- License/terms: repository_license
- Download method: `git_lfs`
- Target directory: `data/raw/kaggle_hot_corridor/building_data_genome_2`
- Notes: Open building energy/weather/cooling-meter dataset; not a direct data-center hot-corridor source.

## CICIoT2023 (ciciot2023)
- Role: `anomaly_timing`
- Manifest status: `substitute`
- Validation state: `candidate_validated`
- Source URL: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- License/terms: dataset_terms
- Download method: `manual_or_web`
- Target directory: `data/raw/optional_iot_anomaly/ciciot2023`
- Notes: IoT network attack dataset. It must not be treated as thermal telemetry.
