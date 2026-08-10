"""Versioned schemas and strict validation for trajectory sidecars."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "uniss_true_subsecond_trajectory_v2"
PLAN_SCHEMA_VERSION = "uniss_true_subsecond_trajectory_plan_v1"


class Action(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


def _ints(values: Sequence[object], name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} contains a negative token")
    return result


def canonical_checksum(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("checksum", None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrajectoryPlan:
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
    source_glm_length: int
    source_bicodec_length: int
    target_bicodec_length: int
    schema_version: str = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.shard < 0 or self.row_index < 0:
            raise ValueError("shard and row_index must be non-negative")
        if self.src_lang not in {"eng", "cmn"} or self.tgt_lang not in {"eng", "cmn"}:
            raise ValueError("only eng/cmn directions are supported")
        if self.src_lang == self.tgt_lang:
            raise ValueError("source and target language must differ")
        if self.trajectory_kind not in {"early", "middle_late"}:
            raise ValueError("unsupported trajectory_kind")
        if not 0 < self.chunk_end_ms <= self.future_1_end_ms <= self.future_2_end_ms:
            raise ValueError("trajectory times must be positive and monotonic")
        if self.future_2_end_ms > self.source_duration_ms:
            raise ValueError("future prefix exceeds source duration")
        for name in ("source_glm_length", "source_bicodec_length", "target_bicodec_length"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    target_text_delta_ids: tuple[int, ...]
    semantic_history_start: int
    semantic_history_end: int
    semantic_target_start: int
    semantic_target_end: int
    speaker_global: tuple[int, ...]
    soft_deadline_ms: int = 640
    hard_deadline_ms: int = 800
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    checksum: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory schema")
        if "::causal:" not in self.frontend_token_cache:
            raise ValueError("frontend_token_cache must reference a causal-token row")
        if len(self.speaker_global) != 32:
            raise ValueError("speaker_global must contain exactly 32 tokens")
        _ints(self.causal_source_glm, "causal_source_glm")
        _ints(self.future_1_source_glm, "future_1_source_glm")
        _ints(self.future_2_source_glm, "future_2_source_glm")
        _ints(self.translation_ids, "translation_ids")
        _ints(self.speaker_global, "speaker_global")
        _ints(self.target_text_delta_ids, "target_text_delta_ids", allow_empty=True)
        if not (
            len(self.causal_source_glm)
            <= len(self.future_1_source_glm)
            <= len(self.future_2_source_glm)
        ):
            raise ValueError("source prefixes must be monotonic")
        if not 0 <= self.previous_committed_length <= self.stable_target_length <= len(self.translation_ids):
            raise ValueError("invalid committed/stable target lengths")
        expected = self.stable_target_length - self.previous_committed_length
        if self.new_supported_count != expected:
            raise ValueError("new_supported_count does not match stable delta")
        if self.support_bucket != min(expected, 4):
            raise ValueError("support_bucket is inconsistent")
        if len(self.safe_commit_mask) != len(self.translation_ids):
            raise ValueError("safe_commit_mask length mismatch")
        if not 0 <= self.semantic_history_start <= self.semantic_history_end:
            raise ValueError("invalid semantic history span")
        if not self.semantic_history_end <= self.semantic_target_start < self.semantic_target_end:
            raise ValueError("invalid semantic target span")
        if self.semantic_target_end - self.semantic_target_start not in {8, 12, 16}:
            raise ValueError("semantic target block must contain 8, 12, or 16 tokens")
        if self.natural_action_target is Action.WRITE and expected <= 0:
            raise ValueError("natural WRITE requires supported target content")
        if self.natural_action_target is Action.READ and expected > 0:
            raise ValueError("natural READ cannot hide supported target content")
        expected_forced = self.deadline_action_target is Action.WRITE and expected == 0
        if self.deadline_forced_target != expected_forced:
            raise ValueError("deadline_forced_target is inconsistent")
        if self.checksum and self.checksum != canonical_checksum(asdict(self)):
            raise ValueError("trajectory checksum mismatch")

    def with_checksum(self) -> "TrajectoryRecord":
        values = asdict(self)
        values["checksum"] = canonical_checksum(values)
        return TrajectoryRecord.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["natural_action_target"] = self.natural_action_target.value
        value["deadline_action_target"] = self.deadline_action_target.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        payload = dict(value)
        tuple_fields = (
            "causal_source_glm",
            "future_1_source_glm",
            "future_2_source_glm",
            "translation_ids",
            "safe_commit_mask",
            "target_text_delta_ids",
            "speaker_global",
            "quality_flags",
        )
        for name in tuple_fields:
            payload[name] = tuple(payload.get(name, ()))
        payload["natural_action_target"] = Action(payload["natural_action_target"])
        payload["deadline_action_target"] = Action(payload["deadline_action_target"])
        return cls(**payload)
