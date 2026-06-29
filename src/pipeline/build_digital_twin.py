from __future__ import annotations

from pathlib import Path

from src.simulation.digital_twin import TwinTrace, build_synthetic_digital_twin
from src.utils.io import ensure_dir, read_yaml, write_json


def build_digital_twin(
    simulation_config_path: str | Path = "configs/simulation.yaml",
    output_path: str | Path = "data/processed/digital_twin.csv",
) -> TwinTrace:
    config = read_yaml(simulation_config_path)
    trace = build_synthetic_digital_twin(config)
    ensure_dir(Path(output_path).parent)
    trace.frame.to_csv(output_path, index=False)
    write_json(Path(output_path).with_suffix(".metadata.json"), trace.metadata)
    return trace
