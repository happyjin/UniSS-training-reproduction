"""Immutable V1 rollout sidecar schema and append-only validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Mapping

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import append_text
from training import constants_uniss as c


ROLLOUT_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_v1_asr_rollout_v1"


@dataclass(frozen=True)
class V1RolloutEvent:
    event_index: int
    source_end_ms: int
    visible_glm_tokens: int
    generated_tokens: tuple[int, ...]
    content_tokens: tuple[int, ...]
    v1_source_delta: str
    v1_source_prefix: str
    reached_content_stop: bool
    write_structure_valid: bool
    early_eos: bool
    noise_severity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "V1RolloutEvent":
        fields = dict(value)
        fields["generated_tokens"] = tuple(int(item) for item in fields["generated_tokens"])  # type: ignore[index]
        fields["content_tokens"] = tuple(int(item) for item in fields["content_tokens"])  # type: ignore[index]
        return cls(**fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class V1Rollout:
    sample_id: str
    split: str
    src_lang: str
    source_manifest_record: int
    v1_checkpoint_sha256: str
    v1_hf_sha256: str
    runtime_sha256: str
    source_audio_sha256: str
    events: tuple[V1RolloutEvent, ...]
    final_generated_tokens: tuple[int, ...]
    final_reached_eos: bool
    full_text: str
    metric: str
    errors: int
    reference_units: int
    error_rate: float
    empty_events: int
    early_eos_events: int
    malformed_write_events: int
    final_visible_glm_tokens: int
    elapsed_seconds: float
    schema_version: str = ROLLOUT_SCHEMA

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "V1Rollout":
        fields = dict(value)
        fields["events"] = tuple(
            V1RolloutEvent.from_mapping(item) for item in fields["events"]  # type: ignore[index]
        )
        fields["final_generated_tokens"] = tuple(
            int(item) for item in fields["final_generated_tokens"]  # type: ignore[index]
        )
        return cls(**fields)  # type: ignore[arg-type]


def validate_rollout(rollout: V1Rollout, *, expected_events: int | None = None) -> None:
    if rollout.schema_version != ROLLOUT_SCHEMA:
        raise ValueError("unexpected V1 rollout schema")
    if not rollout.sample_id or not rollout.events:
        raise ValueError("V1 rollout is missing sample ID or events")
    for label, value in (
        ("V1 checkpoint SHA256", rollout.v1_checkpoint_sha256),
        ("V1 HF SHA256", rollout.v1_hf_sha256),
        ("runtime SHA256", rollout.runtime_sha256),
        ("source audio SHA256", rollout.source_audio_sha256),
    ):
        if len(value) != 64:
            raise ValueError(f"{label} is not a SHA256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"{label} is not hexadecimal") from exc
    if expected_events is not None and len(rollout.events) != int(expected_events):
        raise ValueError("V1 rollout event count differs from gold trajectory")
    prefix = ""
    previous_ms = 0
    previous_glm = 0
    for index, event in enumerate(rollout.events):
        if event.event_index != index:
            raise ValueError("V1 rollout event indices are not contiguous")
        if event.source_end_ms <= previous_ms:
            raise ValueError("V1 rollout source times are not strictly increasing")
        if event.visible_glm_tokens < previous_glm:
            raise ValueError("V1 rollout visible GLM count rolled back")
        prefix = append_text(prefix, event.v1_source_delta, rollout.src_lang)
        if prefix != event.v1_source_prefix:
            raise ValueError("V1 rollout text prefix is not append-only")
        if not event.generated_tokens and event.content_tokens:
            raise ValueError("V1 rollout content exists without generated tokens")
        if event.early_eos and c.TOKEN_EOS not in event.generated_tokens:
            raise ValueError("V1 rollout early-EOS flag has no EOS token")
        if event.reached_content_stop and event.generated_tokens and (
            event.generated_tokens[-1] != c.TOKEN_END_CONTENT
        ):
            raise ValueError("V1 rollout content-stop flag differs from generated tokens")
        previous_ms = event.source_end_ms
        previous_glm = event.visible_glm_tokens
    if prefix != rollout.full_text:
        raise ValueError("V1 rollout events do not reconstruct full text")
    if rollout.reference_units <= 0 or rollout.errors < 0:
        raise ValueError("V1 rollout error counts are invalid")
    if rollout.final_visible_glm_tokens < previous_glm:
        raise ValueError("V1 rollout final visible GLM count rolled back")
    if rollout.elapsed_seconds < 0:
        raise ValueError("V1 rollout elapsed time is negative")
    expected_rate = rollout.errors / rollout.reference_units
    if abs(expected_rate - rollout.error_rate) > 1e-12:
        raise ValueError("V1 rollout error rate differs from counts")
    if rollout.empty_events != sum(not event.v1_source_delta for event in rollout.events):
        raise ValueError("V1 rollout empty-event count differs")
    if rollout.early_eos_events != sum(event.early_eos for event in rollout.events):
        raise ValueError("V1 rollout early-EOS count differs")
    if rollout.malformed_write_events != sum(
        bool(event.generated_tokens) and not event.write_structure_valid
        for event in rollout.events
    ):
        raise ValueError("V1 rollout malformed-WRITE count differs")


__all__ = ["ROLLOUT_SCHEMA", "V1Rollout", "V1RolloutEvent", "validate_rollout"]
