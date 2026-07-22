"""Source/Target CTC eligibility gate for WAIT/WRITE decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PolicyDecision(str, Enum):
    WAIT = "wait"
    WRITE = "write"


@dataclass
class PolicyGate:
    min_write_tokens: int = 1
    confidence_threshold: float = 0.0
    previous_source_count: int = 0

    def eligible(
        self,
        *,
        source_count: int,
        target_supported_count: int,
        target_committed_count: int,
        target_confidence: float = 1.0,
        is_final: bool = False,
    ) -> bool:
        if is_final:
            self.previous_source_count = max(self.previous_source_count, source_count)
            return True
        source_grew = source_count > self.previous_source_count
        self.previous_source_count = max(self.previous_source_count, source_count)
        target_growth = target_supported_count - target_committed_count
        return (
            source_grew
            and target_growth >= self.min_write_tokens
            and target_confidence >= self.confidence_threshold
        )

    def decide(self, **kwargs) -> PolicyDecision:
        return PolicyDecision.WRITE if self.eligible(**kwargs) else PolicyDecision.WAIT
