"""Model-agnostic append-only Simul-UniSS controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from uniss.streaming.bicodec_streamer import StreamingBiCodecDecoder
from uniss.streaming.policy import PolicyDecision, PolicyGate
from uniss.streaming.stable_prefix import StablePrefixCommitter


@dataclass(frozen=True)
class FrontendStep:
    candidate_glm: Sequence[int]
    source_count: int
    target_supported_count: int
    target_confidence: float = 1.0


@dataclass(frozen=True)
class WriteResult:
    target_text_ids: Sequence[int]
    semantic_tokens: Sequence[int]


class ModelAdapter(Protocol):
    def append_source(self, glm_tokens: Sequence[int]) -> None: ...

    def choose_action(self, eligible: bool, is_final: bool) -> PolicyDecision: ...

    def commit_wait(self) -> None: ...

    def generate_write(self, is_final: bool) -> WriteResult: ...


@dataclass
class StreamingController:
    model: ModelAdapter
    codec: StreamingBiCodecDecoder
    policy: PolicyGate = field(default_factory=PolicyGate)
    prefix_committer: StablePrefixCommitter = field(default_factory=StablePrefixCommitter)
    committed_target_tokens: int = 0
    wait_count: int = 0
    write_count: int = 0

    def process_step(
        self,
        frontend: FrontendStep,
        *,
        speaker_tokens: Sequence[int],
        is_final: bool = False,
    ) -> tuple[PolicyDecision, np.ndarray]:
        new_glm = self.prefix_committer.update(frontend.candidate_glm, is_final=is_final)
        if new_glm:
            self.model.append_source(new_glm)
        eligible = self.policy.eligible(
            source_count=frontend.source_count,
            target_supported_count=frontend.target_supported_count,
            target_committed_count=self.committed_target_tokens,
            target_confidence=frontend.target_confidence,
            is_final=is_final,
        )
        action = self.model.choose_action(eligible=eligible, is_final=is_final)
        if is_final and action == PolicyDecision.WAIT:
            action = PolicyDecision.WRITE
        if action == PolicyDecision.WAIT:
            self.wait_count += 1
            self.model.commit_wait()
            return action, np.zeros(0, dtype=np.float32)

        self.write_count += 1
        result = self.model.generate_write(is_final=is_final)
        self.committed_target_tokens += len(result.target_text_ids)
        waveform = self.codec.push(
            result.semantic_tokens,
            speaker_tokens=speaker_tokens,
            is_final=is_final,
        )
        return action, waveform
