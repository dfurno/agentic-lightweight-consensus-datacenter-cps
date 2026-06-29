from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_first_table(path: str | Path) -> pd.DataFrame:
    root = Path(path)
    for suffix in ("*.csv", "*.parquet", "*.json", "*.jsonl"):
        files = sorted(root.rglob(suffix))
        if not files:
            continue
        file = files[0]
        if file.suffix == ".csv":
            return pd.read_csv(file)
        if file.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(file)
        return pd.read_json(file, lines=file.suffix == ".jsonl")
    raise FileNotFoundError(f"No supported table found under {path}")
