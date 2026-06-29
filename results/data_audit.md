# Data Audit

## ENACT (enact)
- Role: `workload_power`
- Source: https://zenodo.org/records/18920397
- License/terms: record_specific
- Files inspected: 4
- Usable for power: `True`
- Usable for temperature: `False`
- Usable for attack patterns: `False`
- Not usable reason: none

## Data Centre Warm Channel Temperature Prediction (kaggle_hot_corridor)
- Role: `thermal_calibration`
- Source: https://www.kaggle.com/datasets/mbjunior/data-centre-hot-corridor-temperature-prediction
- License/terms: kaggle_dataset_specific
- Files inspected: 20
- Usable for power: `False`
- Usable for temperature: `True`
- Usable for attack patterns: `False`
- Not usable reason: none

## CuCD-ID (cucd_id)
- Role: `anomaly_timing`
- Source: https://data.mendeley.com/datasets/7n2d42pm3n/3
- License/terms: CC BY 4.0
- Files inspected: 3
- Usable for power: `False`
- Usable for temperature: `False`
- Usable for attack patterns: `True`
- Not usable reason: none
- Mapping warning: Cyber/IoT datasets may only be used for pattern-level attack timing, not thermal measurements.

## Google Cluster Data (google_cluster_data)
- Role: `workload_power`
- Source: https://github.com/google/cluster-data
- License/terms: CC-BY
- Files inspected: 0
- Usable for power: `False`
- Usable for temperature: `False`
- Usable for attack patterns: `False`
- Not usable reason: No local tabular files found under target directory.

## Building Data Genome Project 2 / ASHRAE GEPIII (building_data_genome_2)
- Role: `thermal_calibration`
- Source: https://github.com/buds-lab/building-data-genome-project-2
- License/terms: repository_license
- Files inspected: 19
- Usable for power: `False`
- Usable for temperature: `False`
- Usable for attack patterns: `False`
- Not usable reason: No temperature-like columns detected.

## CICIoT2023 (ciciot2023)
- Role: `anomaly_timing`
- Source: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- License/terms: dataset_terms
- Files inspected: 0
- Usable for power: `False`
- Usable for temperature: `False`
- Usable for attack patterns: `False`
- Not usable reason: No local tabular files found under target directory.
- Mapping warning: Cyber/IoT datasets may only be used for pattern-level attack timing, not thermal measurements.
