from __future__ import annotations

import hashlib

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    TrajectoryEvent,
    hash_int_sequence,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
    V1RolloutEvent,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.requests import (
    build_phase3_requests,
)
from training import constants_uniss as c


DIGEST = hashlib.sha256(b"teacher").hexdigest()


def _encode(text: str) -> list[int]:
    return [1000 + ord(value) for value in text]


def _trajectory() -> E2ETrajectory:
    events = (
        TrajectoryEvent(
            event_index=0,
            source_start_ms=0,
            source_end_ms=500,
            source_pcm_start=0,
            source_pcm_end=8000,
            source_glm_start=0,
            source_glm_end=2,
            source_glm_delta=(1, 2),
            gold_source_word_start=0,
            gold_source_word_end=1,
            gold_source_delta="hello",
            gold_source_prefix="hello",
            v1_source_delta=None,
            v1_source_prefix=None,
            target_text_delta="你",
            target_text_prefix="你",
            target_semantic_start=0,
            target_semantic_end=2,
            target_semantic_delta=(10, 11),
            target_support_end_ms=400,
            source_final=False,
            target_final=False,
            alignment_confidence=1.0,
            noise_severity="pending",
        ),
        TrajectoryEvent(
            event_index=1,
            source_start_ms=500,
            source_end_ms=1000,
            source_pcm_start=8000,
            source_pcm_end=16000,
            source_glm_start=2,
            source_glm_end=4,
            source_glm_delta=(3, 4),
            gold_source_word_start=1,
            gold_source_word_end=2,
            gold_source_delta="world",
            gold_source_prefix="hello world",
            v1_source_delta=None,
            v1_source_prefix=None,
            target_text_delta="好",
            target_text_prefix="你好",
            target_semantic_start=2,
            target_semantic_end=3,
            target_semantic_delta=(12,),
            target_support_end_ms=900,
            source_final=True,
            target_final=True,
            alignment_confidence=1.0,
            noise_severity="pending",
        ),
    )
    return E2ETrajectory(
        sample_id="sample-1",
        split="valid",
        src_lang="eng",
        tgt_lang="cmn",
        source_audio="/tmp/source.wav",
        source_audio_sha256=DIGEST,
        source_audio_hash_status="complete",
        source_audio_frames=16000,
        source_audio_channels=1,
        source_audio_finite=True,
        source_audio_audit_status="complete",
        source_sample_rate=16000,
        source_duration_ms=1000,
        speaker_global=tuple(range(32)),
        full_transcription="hello world",
        normalized_transcription="hello world",
        full_translation="你好",
        normalized_translation="你好",
        source_glm_length=4,
        source_glm_sha256=hash_int_sequence((1, 2, 3, 4)),
        target_semantic_length=3,
        target_semantic_sha256=hash_int_sequence((10, 11, 12)),
        source_manifest="/tmp/source.jsonl",
        source_manifest_record=0,
        v1_checkpoint_sha256=DIGEST,
        phase3_teacher_sha256=DIGEST,
        v1_rollout_status="pending",
        events=events,
    )


def _rollout() -> V1Rollout:
    rows = (
        V1RolloutEvent(
            event_index=0,
            source_end_ms=500,
            visible_glm_tokens=2,
            generated_tokens=(
                c.TOKEN_WRITE_GENERATE,
                c.TOKEN_ENG,
                c.TOKEN_START_CONTENT,
                *_encode("hello"),
                c.TOKEN_END_CONTENT,
            ),
            content_tokens=tuple(_encode("hello")),
            v1_source_delta="hello",
            v1_source_prefix="hello",
            reached_content_stop=True,
            write_structure_valid=True,
            early_eos=False,
            noise_severity="exact",
        ),
        V1RolloutEvent(
            event_index=1,
            source_end_ms=1000,
            visible_glm_tokens=4,
            generated_tokens=(
                c.TOKEN_WRITE_GENERATE,
                c.TOKEN_ENG,
                c.TOKEN_START_CONTENT,
                *_encode("world"),
                c.TOKEN_END_CONTENT,
            ),
            content_tokens=tuple(_encode("world")),
            v1_source_delta="world",
            v1_source_prefix="hello world",
            reached_content_stop=True,
            write_structure_valid=True,
            early_eos=False,
            noise_severity="exact",
        ),
    )
    return V1Rollout(
        sample_id="sample-1",
        split="valid",
        src_lang="eng",
        source_manifest_record=0,
        v1_checkpoint_sha256=DIGEST,
        v1_hf_sha256=DIGEST,
        runtime_sha256=DIGEST,
        source_audio_sha256=DIGEST,
        events=rows,
        final_generated_tokens=(c.TOKEN_EOS,),
        final_reached_eos=True,
        full_text="hello world",
        metric="wer",
        errors=0,
        reference_units=2,
        error_rate=0.0,
        empty_events=0,
        early_eos_events=0,
        malformed_write_events=0,
        final_visible_glm_tokens=4,
        elapsed_seconds=1.0,
    )


def test_phase3_requests_cover_gold_v1_mt_and_semantic_without_future_text() -> None:
    requests = build_phase3_requests(_trajectory(), _rollout(), encode_text=_encode)
    assert len(requests) == 6
    assert sum(value.family == "phase3_mt" for value in requests) == 4
    assert sum(value.family == "phase3_semantic" for value in requests) == 2
    assert {value.history_kind for value in requests} == {
        "gold_source",
        "v1_source",
        "gold_target",
    }
    first_semantic = next(
        value
        for value in requests
        if value.family == "phase3_semantic" and value.event_index == 0
    )
    assert _encode("你")[0] in first_semantic.prompt_ids
    assert _encode("好")[0] not in first_semantic.prompt_ids
    assert first_semantic.visible_semantic_tokens == 0
    final_semantic = next(
        value
        for value in requests
        if value.family == "phase3_semantic" and value.event_index == 1
    )
    assert final_semantic.reference_labels[-1] == c.TOKEN_EOS
    assert all(value.content_selected_tokens > 0 for value in requests)
