from __future__ import annotations

from src.agents.schemas import ConsensusSnapshot
from src.consensus.fpr_owa import ConsensusResult


def consensus_to_snapshot(result: ConsensusResult) -> ConsensusSnapshot:
    return ConsensusSnapshot(
        temperature=result.aggregated_value,
        timestamp=result.timestamp,
        confidence=result.confidence,
        anomaly_count=int(result.anomaly_flags.sum()),
    )
