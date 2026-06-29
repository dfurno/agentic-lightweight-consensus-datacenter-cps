# Execution Trace Schema

Per-tick trace files are CSV files generated under `results/traces/`.

Typical columns:

- `timestamp`: tick timestamp;
- `power_cu`: workload/power proxy;
- `ambient_temperature`: ambient temperature when available;
- `temperature_ground_truth`: target thermal state;
- `cooling_proxy`: cooling/heating proxy when available;
- `sensor_0`, `sensor_1`, ...: sensor readings;
- `attack_label_0`, `attack_label_1`, ...: boolean labels indicating attacked
  sensors/ticks.

Metadata JSON files next to traces record the dataset source, number of sensors,
number of time steps, and whether the trace is dataset-driven.

Large trace CSV archives are intentionally suitable for external release assets.
The aggregate result files in `results/` and the scripts in `scripts/` are the
primary committed reproduction artifacts.
