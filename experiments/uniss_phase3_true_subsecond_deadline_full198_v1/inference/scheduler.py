"""640/800 ms safe-commit scheduler with explicit forced-write accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeadlineDecision:
    action: str
    natural_write: bool
    deadline_forced: bool
    commit_tokens: int
    reason: str


class DeadlineScheduler:
    def __init__(
        self,
        *,
        soft_deadline_ms: int = 640,
        hard_deadline_ms: int = 800,
        write_threshold: float = 0.5,
        minimum_commit_tokens: int = 1,
        maximum_commit_tokens: int = 8,
    ) -> None:
        if not 0 < soft_deadline_ms <= hard_deadline_ms:
            raise ValueError("invalid soft/hard deadline")
        if not 0.0 < write_threshold < 1.0:
            raise ValueError("write_threshold must be in (0,1)")
        if not 0 < minimum_commit_tokens <= maximum_commit_tokens:
            raise ValueError("invalid commit token limits")
        self.soft_deadline_ms = soft_deadline_ms
        self.hard_deadline_ms = hard_deadline_ms
        self.write_threshold = write_threshold
        self.minimum_commit_tokens = minimum_commit_tokens
        self.maximum_commit_tokens = maximum_commit_tokens

    def decide(
        self,
        *,
        elapsed_speech_ms: int,
        write_probability: float,
        supported_tokens: int,
        speech_active: bool,
        final: bool = False,
    ) -> DeadlineDecision:
        supported = max(0, min(int(supported_tokens), self.maximum_commit_tokens))
        natural = supported >= self.minimum_commit_tokens and write_probability >= self.write_threshold
        if natural:
            return DeadlineDecision("WRITE", True, False, supported, "model_safe_commit")
        if final and supported > 0:
            return DeadlineDecision("WRITE", False, False, supported, "final_safe_flush")
        if not speech_active:
            return DeadlineDecision("READ", False, False, 0, "silence")
        if elapsed_speech_ms >= self.hard_deadline_ms:
            # The hard scheduler guarantees an action boundary, but it must not
            # invent a hard reference token. Anticipation decoding supplies a
            # soft candidate downstream when supported==0.
            return DeadlineDecision(
                "WRITE",
                False,
                supported == 0,
                supported,
                "hard_deadline_safe" if supported else "hard_deadline_anticipation",
            )
        if elapsed_speech_ms >= self.soft_deadline_ms and supported > 0:
            return DeadlineDecision("WRITE", False, False, supported, "soft_deadline_supported")
        return DeadlineDecision("READ", False, False, 0, "insufficient_support")
