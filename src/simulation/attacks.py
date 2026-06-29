from __future__ import annotations

import numpy as np


def attack_window(length: int, start_fraction: float, end_fraction: float) -> slice:
    start = int(length * start_fraction)
    end = max(start + 1, int(length * end_fraction))
    return slice(start, min(end, length))


def apply_attack(
    signal: np.ndarray,
    attack_type: str,
    rng: np.random.Generator,
    bias_magnitude: float = 4.0,
    drift_rate: float = 0.03,
    spike_magnitude: float = 8.0,
    scaling_factor: float = 1.15,
    start_fraction: float = 0.35,
    end_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    attacked = np.asarray(signal, dtype=float).copy()
    labels = np.zeros(attacked.shape, dtype=bool)
    window = attack_window(len(attacked), start_fraction, end_fraction)
    labels[window] = True
    idx = np.arange(len(attacked[window]), dtype=float)
    if attack_type == "bias":
        attacked[window] += bias_magnitude
    elif attack_type == "drift":
        attacked[window] += idx * drift_rate
    elif attack_type == "replay":
        source_start = max(0, window.start - (window.stop - window.start))
        replay = attacked[source_start : source_start + (window.stop - window.start)]
        if replay.size < window.stop - window.start:
            replay = np.resize(replay, window.stop - window.start)
        attacked[window] = replay
    elif attack_type == "freeze":
        attacked[window] = attacked[window.start]
    elif attack_type == "spike":
        attacked[window] += rng.choice([-1.0, 1.0], size=idx.size) * spike_magnitude
    elif attack_type == "scaling":
        attacked[window] *= scaling_factor
    elif attack_type == "mixed":
        choices = ["bias", "drift", "freeze", "spike", "scaling"]
        current = signal.copy()
        labels = np.zeros(attacked.shape, dtype=bool)
        segments = np.array_split(np.arange(window.start, window.stop), len(choices))
        for attack, segment in zip(choices, segments):
            if segment.size == 0:
                continue
            tmp, tmp_labels = apply_attack(
                current,
                attack,
                rng,
                bias_magnitude,
                drift_rate,
                spike_magnitude,
                scaling_factor,
                segment[0] / len(signal),
                (segment[-1] + 1) / len(signal),
            )
            current[tmp_labels] = tmp[tmp_labels]
            labels |= tmp_labels
        attacked = current
    else:
        raise ValueError(f"Unknown attack type {attack_type!r}")
    return attacked, labels
