from __future__ import annotations

import pandas as pd


def resample_time_grid(frame: pd.DataFrame, timestamp_column: str, period_seconds: int) -> pd.DataFrame:
    data = frame.copy()
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data = data.sort_values(timestamp_column).set_index(timestamp_column)
    return data.resample(f"{period_seconds}s").mean(numeric_only=True).interpolate().reset_index()
