from __future__ import annotations

import numpy as np


def fallback_temperature(
    power: np.ndarray,
    ambient: float,
    rho: float,
    eta: float,
    lag: int,
    disturbance_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    power = np.asarray(power, dtype=float)
    temp = np.zeros_like(power, dtype=float)
    temp[0] = ambient + eta * power[0]
    for t in range(1, len(power)):
        lagged_idx = max(0, t - lag)
        temp[t] = (
            ambient
            + rho * (temp[t - 1] - ambient)
            + eta * power[lagged_idx]
            + rng.normal(0.0, disturbance_std)
        )
    return temp
