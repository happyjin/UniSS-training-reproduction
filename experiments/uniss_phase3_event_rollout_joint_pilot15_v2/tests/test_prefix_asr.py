from __future__ import annotations

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.evaluate_checkpoint import (
    oracle_target_prefixes,
)
from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation.score_prefix_asr import (
    score,
)


def test_oracle_target_prefixes_are_language_aware() -> None:
    assert oracle_target_prefixes(
        {"tgt_lang": "cmn", "target_words": [{"text": "我"}, {"text": "同意"}]}
    ) == ["我", "我同意"]
    assert oracle_target_prefixes(
        {"tgt_lang": "eng", "target_words": [{"text": "I"}, {"text": "agree"}]}
    ) == ["I", "I agree"]


def test_prefix_asr_uses_earliest_contentful_matching_prefix() -> None:
    base = {
        "mode": "exact_runtime_prefix_asr",
        "parent_sample_id": "sample",
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "oracle_target_text_prefixes": ["我", "我完全同意"],
    }
    rows = [
        {**base, "id": "sample:0", "candidate_index": 0, "source_end_ms": 320, "wall_end_ms": 600.0, "asr_text": "我"},
        {**base, "id": "sample:1", "candidate_index": 1, "source_end_ms": 480, "wall_end_ms": 850.0, "asr_text": "我同意"},
    ]
    report = score(rows, minimum_similarity=0.5, minimum_content_units=2)
    sample = report["samples"][0]
    assert sample["first_useful_audio_candidate_id"] == "sample:1"
    assert report["groups"]["all"]["first_useful_audio_wall_ms"]["p50"] == 850.0


def test_unrelated_asr_never_becomes_useful_audio() -> None:
    rows = [
        {
            "id": "sample:0",
            "parent_sample_id": "sample",
            "candidate_index": 0,
            "src_lang": "cmn",
            "tgt_lang": "eng",
            "source_end_ms": 320,
            "wall_end_ms": 700.0,
            "oracle_target_text_prefixes": ["the weather is nice"],
            "asr_text": "random noise",
        }
    ]
    report = score(rows, minimum_similarity=0.5, minimum_content_units=2)
    assert report["groups"]["all"]["useful_audio_recall"] == 0.0
    assert report["groups"]["all"]["first_useful_audio_wall_ms"]["p50"] is None


def test_runtime_failures_remain_in_useful_audio_recall_denominator() -> None:
    report = score(
        [],
        minimum_similarity=0.5,
        minimum_content_units=2,
        expected_samples=[
            {
                "sample_id": "failed",
                "src_lang": "eng",
                "tgt_lang": "cmn",
                "error": "runtime_error:semantic_safety_ceiling",
            }
        ],
    )
    assert report["groups"]["all"]["samples"] == 1
    assert report["groups"]["all"]["useful_audio_recall"] == 0.0
    assert report["samples"][0]["runtime_error"]
