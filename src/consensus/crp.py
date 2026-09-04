"""Consensus-Reaching Process (CRP) operator faithful to Methodology Sec. 4.5.

This module implements the zone-level consensus-reaching loop that wraps the
single-round FPR-OWA operator with:
  - social proximity / zone consensus degree (Eq. similarity/consensusdegree),
  - temporal self-consistency from an EWMA short-horizon prediction (Eq. consistency),
  - combined reliability r_i = alpha_crp * p_i + (1 - alpha_crp) * k_i (Eq. combined),
  - feedback by delayed inclusion (exclude low-reliability reporters) iterated for
    up to `max_rounds`, forwarding a degraded-confidence snapshot otherwise.

It is evaluated offline by replaying recorded traces; "re-sampling" feedback is
approximated by delayed inclusion / down-weighting (documented as a replay
assumption). State (the per-sensor EWMA) is carried across ticks by `CRPState`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.consensus.fpr import reliability_scores
from src.consensus.fpr_owa import fpr_owa_consensus


@dataclass
class CRPState:
    """Per-zone temporal state carried across control ticks."""
    ewma: np.ndarray | None = None          # short-horizon prediction s_tilde_i
    below_streak: np.ndarray | None = None   # consecutive rounds with r_i < tau_r
    lam: float = 0.3                          # EWMA smoothing factor

    def update_ewma(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.ewma is None or self.ewma.shape != values.shape:
            self.ewma = values.copy()
            self.below_streak = np.zeros_like(values)
            return self.ewma
        prediction = self.ewma  # prediction for current tick = previous EWMA
        self.ewma = self.lam * values + (1.0 - self.lam) * self.ewma
        return prediction


@dataclass
class CRPResult:
    aggregated_value: float
    confidence: float                 # mean combined reliability of retained reporters
    consensus_degree: float           # C_z
    combined_reliability: np.ndarray  # r_i over all reporters
    anomaly_flags: np.ndarray         # per-tick flag r_i < tau_r
    persistent_flags: np.ndarray      # r_i < tau_r for >= L consecutive rounds
    rounds: int
    converged: bool
    excluded: np.ndarray              # reporters excluded by feedback


def _proximity(values: np.ndarray, m_z: float) -> tuple[np.ndarray, float]:
    n = values.size
    if n == 1:
        return np.array([1.0]), 1.0
    diff = np.abs(values[:, None] - values[None, :])
    sim = np.maximum(0.0, 1.0 - diff / m_z)
    np.fill_diagonal(sim, 0.0)
    p = sim.sum(axis=1) / (n - 1)
    return p, float(np.mean(p))


def crp_consensus(
    values: np.ndarray,
    state: CRPState,
    *,
    alpha: float = 0.3,      # OWA quantifier lower bound a (matches scenario grid)
    beta: float = 0.8,       # OWA quantifier upper bound b
    alpha_crp: float = 0.5,  # social/temporal blend
    m_z: float = 4.0,
    tau_c: float = 0.7,
    tau_r: float = 0.5,
    persistence_l: int = 3,
    max_rounds: int = 3,
    min_sensors: int = 3,
    timestamp: str | None = None,
) -> CRPResult:
    values = np.asarray(values, dtype=float)
    n = values.size
    prediction = state.update_ewma(values)
    # temporal self-consistency (Eq. consistency)
    k = np.maximum(0.0, 1.0 - np.abs(values - prediction) / m_z)

    active = np.ones(n, dtype=bool)
    rounds = 0
    converged = False
    last_p = np.zeros(n)
    last_C = 0.0
    for rounds in range(1, max_rounds + 1):
        idx = np.where(active)[0]
        sub = values[idx]
        p_sub, C = _proximity(sub, m_z)
        last_C = C
        p_full = np.zeros(n)
        p_full[idx] = p_sub
        last_p = p_full
        if C >= tau_c or idx.size <= min_sensors:
            converged = C >= tau_c
            break
        # combined reliability on active set; exclude the weakest below tau_r (delayed inclusion)
        r_sub = alpha_crp * p_sub + (1.0 - alpha_crp) * k[idx]
        weak = idx[r_sub < tau_r]
        if weak.size == 0 or idx.size - weak.size < min_sensors:
            break
        active[weak] = False

    # final combined reliability over all reporters (active ones aggregated)
    r = alpha_crp * last_p + (1.0 - alpha_crp) * k
    idx = np.where(active)[0]
    # estimate via FPR-OWA over retained reporters (degraded confidence if not converged)
    # ``idx`` contains the stable reporter identifiers from the original input;
    # preserve them when feedback excludes reporters rather than renumbering the
    # retained subset by its compact array position.
    est = fpr_owa_consensus(
        values[idx], alpha=alpha, beta=beta, timestamp=timestamp, sensor_ids=idx
    ).aggregated_value

    flags = r < tau_r
    # update persistence streak
    if state.below_streak is None or state.below_streak.shape != flags.shape:
        state.below_streak = np.zeros(n)
    state.below_streak = np.where(flags, state.below_streak + 1, 0.0)
    persistent = state.below_streak >= persistence_l

    confidence = float(np.mean(r[idx])) if idx.size else 0.0
    if not converged:
        confidence *= 0.5  # degraded-confidence forwarding
    return CRPResult(
        aggregated_value=float(est),
        confidence=confidence,
        consensus_degree=float(last_C),
        combined_reliability=r,
        anomaly_flags=flags,
        persistent_flags=persistent,
        rounds=int(rounds),
        converged=bool(converged),
        excluded=~active,
    )
