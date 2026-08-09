"""Stability-driven wait-k for Student v2 (Step 3).

Counts cumulative source tokens whose stability probability ≥ ``threshold``.
When that count reaches ``k``, the policy emits WRITE; otherwise WAIT.
This is the Student-v2 analogue of Stage05 ``lagging_k`` on CTC tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaitKDecision:
    action: str  # "WAIT" | "WRITE"
    stable_count: int
    total_tokens: int
    newly_stable: int


@dataclass
class StabilityWaitKPolicy:
    k: int
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._stable = 0
        self._seen = 0
        self._written_through = 0

    def reset(self) -> None:
        self._stable = 0
        self._seen = 0
        self._written_through = 0

    def observe(self, stability_probabilities: list[float]) -> WaitKDecision:
        """Ingest the full prefix of stability probs seen so far (idempotent on prefix)."""

        if len(stability_probabilities) < self._seen:
            raise ValueError("stability prefix shrank")
        newly = 0
        for value in stability_probabilities[self._seen :]:
            self._seen += 1
            if float(value) >= self.threshold:
                self._stable += 1
                newly += 1
        action = "WAIT"
        if self._stable - self._written_through >= self.k:
            action = "WRITE"
            self._written_through = self._stable
        return WaitKDecision(
            action=action,
            stable_count=self._stable,
            total_tokens=self._seen,
            newly_stable=newly,
        )
