"""Strict, isolated schema for repaired history-conditioned trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "uniss_true_subsecond_pilot15_trajectory_v2"


class Action(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


def canonical_checksum(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("checksum", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    shard: int
    row_index: int
    src_lang: str
    tgt_lang: str
    source_duration_ms: int
    chunk_end_ms: int
    future_1_end_ms: int
    future_2_end_ms: int
    trajectory_kind: str
    causal_source_glm: tuple[int, ...]
    future_1_source_glm: tuple[int, ...]
    future_2_source_glm: tuple[int, ...]
    frontend_token_cache: str
    translation_ids: tuple[int, ...]
    teacher_prefix_topk_path: str
    teacher_future_1_topk_path: str
    teacher_future_2_topk_path: str
    teacher_full_topk_path: str
    previous_committed_length: int
    stable_target_length: int
    new_supported_count: int
    support_bucket: int
    safe_commit_mask: tuple[bool, ...]
    natural_action_target: Action
    deadline_action_target: Action
    deadline_forced_target: bool
    deadline_loss_enabled: bool
    target_text_delta_ids: tuple[int, ...]
    semantic_history_start: int
    semantic_history_end: int
    semantic_target_start: int
    semantic_target_end: int
    speaker_global: tuple[int, ...]
    soft_deadline_ms: int = 640
    hard_deadline_ms: int = 800
    history_context_version: str = "committed_text_semantic_v1"
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    checksum: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported pilot15 trajectory schema")
        if not self.sample_id or self.shard < 0 or self.row_index < 0:
            raise ValueError("invalid trajectory identity")
        if self.src_lang not in {"eng", "cmn"} or self.tgt_lang not in {"eng", "cmn"}:
            raise ValueError("only eng/cmn directions are supported")
        if self.src_lang == self.tgt_lang:
            raise ValueError("source and target languages must differ")
        if not 0 < self.chunk_end_ms <= self.future_1_end_ms <= self.future_2_end_ms:
            raise ValueError("trajectory times must be positive and monotonic")
        if self.future_2_end_ms > self.source_duration_ms:
            raise ValueError("future prefix exceeds source duration")
        for name in (
            "causal_source_glm",
            "future_1_source_glm",
            "future_2_source_glm",
            "translation_ids",
            "speaker_global",
        ):
            values = tuple(int(value) for value in getattr(self, name))
            if not values or any(value < 0 for value in values):
                raise ValueError(f"{name} must contain non-negative tokens")
        if len(self.speaker_global) != 32:
            raise ValueError("speaker_global must contain exactly 32 tokens")
        if not (
            len(self.causal_source_glm)
            <= len(self.future_1_source_glm)
            <= len(self.future_2_source_glm)
        ):
            raise ValueError("source prefixes must be monotonic")
        if not 0 <= self.previous_committed_length <= self.stable_target_length <= len(
            self.translation_ids
        ):
            raise ValueError("invalid committed/stable target lengths")
        expected = self.stable_target_length - self.previous_committed_length
        if self.new_supported_count != expected or self.support_bucket != min(expected, 4):
            raise ValueError("support metadata is inconsistent")
        if tuple(self.target_text_delta_ids) != tuple(
            self.translation_ids[self.previous_committed_length : self.stable_target_length]
        ):
            raise ValueError("target text delta is inconsistent")
        if len(self.safe_commit_mask) != len(self.translation_ids):
            raise ValueError("safe commit mask length mismatch")
        if not (
            0
            <= self.semantic_history_start
            <= self.semantic_history_end
            == self.semantic_target_start
            <= self.semantic_target_end
        ):
            raise ValueError("semantic history/target cursor is not contiguous")
        semantic_count = self.semantic_target_end - self.semantic_target_start
        if semantic_count > 16:
            raise ValueError("semantic target block exceeds 16 tokens")
        if self.natural_action_target is Action.WRITE:
            if expected <= 0 or semantic_count <= 0:
                raise ValueError("natural WRITE requires text and semantic deltas")
        elif expected > 0 or semantic_count > 0:
            raise ValueError("natural READ cannot hide a supported delta")
        expected_forced = (
            self.chunk_end_ms == self.hard_deadline_ms
            and self.deadline_action_target is Action.WRITE
            and self.natural_action_target is Action.READ
            and expected == 0
        )
        if self.deadline_forced_target != expected_forced:
            raise ValueError("forced WRITE must occur only at the exact hard deadline")
        if not self.deadline_forced_target and self.deadline_action_target is not self.natural_action_target:
            raise ValueError("non-forced deadline action must equal the natural action")
        if self.deadline_loss_enabled and self.source_duration_ms < self.hard_deadline_ms:
            raise ValueError("short utterance cannot enable grouped deadline loss")
        if self.checksum and self.checksum != canonical_checksum(asdict(self)):
            raise ValueError("trajectory checksum mismatch")

    def with_checksum(self) -> "TrajectoryRecord":
        payload = asdict(self)
        payload["checksum"] = canonical_checksum(payload)
        return TrajectoryRecord.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["natural_action_target"] = self.natural_action_target.value
        payload["deadline_action_target"] = self.deadline_action_target.value
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        payload = dict(value)
        for name in (
            "causal_source_glm",
            "future_1_source_glm",
            "future_2_source_glm",
            "translation_ids",
            "safe_commit_mask",
            "target_text_delta_ids",
            "speaker_global",
            "quality_flags",
        ):
            payload[name] = tuple(payload.get(name, ()))
        payload["natural_action_target"] = Action(payload["natural_action_target"])
        payload["deadline_action_target"] = Action(payload["deadline_action_target"])
        return cls(**payload)


__all__ = ["Action", "SCHEMA_VERSION", "TrajectoryRecord", "canonical_checksum"]
