from __future__ import annotations

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.tests.test_teacher_requests import (
    _encode,
    _rollout,
    _trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    FAMILY_INCREMENTAL_MT,
    FAMILY_INTERLEAVED,
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    FAMILY_STREAMING_ASR,
    LOSS_ASR,
    LOSS_EOS,
    LOSS_MT,
    LOSS_NONE,
    LOSS_REPLAY,
    LOSS_SEMANTIC,
    build_incremental_mt_tasks,
    build_interleaved_task,
    build_phase3_replay_tasks,
    build_streaming_asr_task,
)
from training import constants_uniss as c


def test_streaming_asr_task_matches_gold_history_teacher_geometry() -> None:
    trajectory = _trajectory()
    sample = build_streaming_asr_task(
        trajectory, _rollout(), encode_text=_encode
    )
    assert sample.family == FAMILY_STREAMING_ASR
    assert [value for value in sample.speech_indices if value is not None] == list(
        range(trajectory.source_glm_length)
    )
    assert sum(value == LOSS_ASR for value in sample.loss_kinds) == len(
        _encode("hello")
    ) + len(_encode("world"))
    assert sum(value == LOSS_EOS for value in sample.loss_kinds) == 1
    assert len(sample.teacher_bindings) == 3
    assert all(value.cache_kind == "v1_asr" for value in sample.teacher_bindings)
    mapping = sample.to_mapping()
    assert mapping["family"] == FAMILY_STREAMING_ASR
    assert mapping["teacher_bindings"][0]["cache_position_start"] == 0


def test_incremental_mt_tasks_cover_gold_and_v1_histories_without_semantic() -> None:
    tasks = build_incremental_mt_tasks(
        _trajectory(), _rollout(), encode_text=_encode
    )
    assert len(tasks) == 4
    assert all(value.family == FAMILY_INCREMENTAL_MT for value in tasks)
    assert {value.sequence_id.rsplit(":", 1)[-1] for value in tasks} == {
        "gold_source",
        "v1_source",
    }
    for sample in tasks:
        assert any(value == LOSS_MT for value in sample.loss_kinds)
        assert all(value != LOSS_SEMANTIC for value in sample.loss_kinds)
        assert sample.source_audio is None
        assert sum(
            binding.cache_position_stop - binding.cache_position_start
            for binding in sample.teacher_bindings
        ) == sum(value != LOSS_NONE for value in sample.loss_kinds)


def test_interleaved_task_has_one_causal_audio_path_and_all_three_outputs() -> None:
    trajectory = _trajectory()
    sample = build_interleaved_task(trajectory, encode_text=_encode)
    assert sample.family == FAMILY_INTERLEAVED
    assert sample.token_ids[-1] == c.TOKEN_EOS
    assert sample.token_ids.count(c.TOKEN_EOS) == 1
    assert [value for value in sample.speech_indices if value is not None] == list(
        range(trajectory.source_glm_length)
    )
    assert sum(value == LOSS_ASR for value in sample.loss_kinds) == len(
        _encode("hello")
    ) + len(_encode("world"))
    assert sum(value == LOSS_MT for value in sample.loss_kinds) == len(
        _encode("你")
    ) + len(_encode("好"))
    assert sum(value == LOSS_SEMANTIC for value in sample.loss_kinds) == (
        trajectory.target_semantic_length
    )
    assert sum(value == LOSS_EOS for value in sample.loss_kinds) == 1


def test_phase3_replay_tasks_reconstruct_quality_and_performance() -> None:
    quality, performance = build_phase3_replay_tasks(
        _trajectory(), encode_text=_encode
    )
    assert quality.family == FAMILY_PHASE3_QUALITY
    assert performance.family == FAMILY_PHASE3_PERFORMANCE
    assert quality.token_ids[0] == performance.token_ids[0] == c.TOKEN_TASK_S2S_TRANSLATION
    assert c.TOKEN_SLOW_MODE in quality.token_ids
    assert c.TOKEN_BALANCE_MODE in performance.token_ids
    for sample in (quality, performance):
        assert sample.source_audio is None
        assert all(
            value == LOSS_NONE or value == LOSS_REPLAY
            for value in sample.loss_kinds
        )
        assert sample.loss_kinds[-1] == LOSS_REPLAY
        semantic = [
            token
            for token, kind in zip(sample.token_ids, sample.loss_kinds)
            if kind == LOSS_REPLAY
            and c.BICODEC_SEMANTIC_OFFSET
            <= token
            <= c.BICODEC_SEMANTIC_SPAN.last_id
        ]
        assert len(semantic) == _trajectory().target_semantic_length
