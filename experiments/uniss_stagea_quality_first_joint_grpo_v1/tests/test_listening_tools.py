import numpy as np

from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.aggregate_listening import (
    summarize,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.bounded_longform import (
    SAMPLE_RATE,
    _plan_complete_windows,
    _silence_metrics,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.compare_arms import (
    _arm,
    ranking_key,
)


def _row(sample_id: str, src: str, tgt: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "src_lang": src,
        "tgt_lang": tgt,
        "source_duration_ms": 2_000,
        "processing_seconds": 1.0,
        "reference_translation": "你好" if tgt == "cmn" else "hello",
        "generated_streaming_translation": "你好" if tgt == "cmn" else "hello",
        "asr_errors": 1,
        "asr_reference_units": 10,
        "strict_streaming_runtime_passed": True,
        "prefinal_audio_emitted": True,
        "audio_audit": {"healthy": True},
        "first_audio_source_ms": 640,
        "audio_writes": 2,
        "semantic_tokens": 100,
    }


def test_listening_summary_is_directional_and_weighted() -> None:
    value = summarize([_row("en", "eng", "cmn"), _row("zh", "cmn", "eng")])
    assert value["weighted_asr_error_rate"] == 0.1
    assert value["strict_streaming_pass_rate"] == 1.0
    assert set(value["translation"]) == {"cmn->eng", "eng->cmn"}


def test_longform_silence_audit_tracks_internal_gap() -> None:
    waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    waveform[: SAMPLE_RATE // 5] = 0.1
    waveform[-SAMPLE_RATE // 5 :] = 0.1
    value = _silence_metrics(waveform)
    assert value["first_non_silent_ms"] == 0.0
    assert value["maximum_internal_silence_ms"] >= 500.0


def test_longform_window_plan_relaxes_impossible_minimum_without_exceeding_cap() -> None:
    # 34 seconds cannot be split into windows that are both >=18 and <=30
    # seconds.  The evaluator must preserve all samples and the hard 30-second
    # frontend cap instead of rejecting the recording.
    waveform = np.ones(34 * SAMPLE_RATE, dtype=np.float32)
    spans, mode = _plan_complete_windows(
        waveform,
        SAMPLE_RATE,
        target_seconds=25.0,
        minimum_seconds=18.0,
        maximum_seconds=30.0,
    )
    assert mode == "equal_partition_relaxed_minimum"
    assert spans[0].start_sample == 0
    assert spans[-1].end_sample == len(waveform)
    assert all(span.samples <= 30 * SAMPLE_RATE for span in spans)
    assert all(left.end_sample == right.start_sample for left, right in zip(spans, spans[1:]))


def _metric_block(score: float) -> dict[str, object]:
    path = {
        "directions": {
            "cmn->eng": {
                "candidate_bleu": score,
                "candidate_chrf": score,
            },
            "eng->cmn": {
                "candidate_bleu": score,
                "candidate_chrf": score,
            },
        },
        "target_coverage_min": 1.0,
        "target_coverage_mean": 1.0,
        "target_rollback_events": 0,
        "commit_conflicts": 0,
        "unterminated_generations": 0,
    }
    return {"gold_source": path, "free_running_source": path}


def _comparison_summary(score: float, malformed: int = 0) -> dict[str, object]:
    metrics = {
        "e_mt": _metric_block(score),
        "e_s2s_free": {
            "samples": 4,
            "semantic_coverage_mean": 1.0,
            "non_silent_pcm": 4,
            "target_text_before_source_eos": 4,
            "target_semantic_before_source_eos": 4,
            "malformed_segments": malformed,
            "invalid_semantic_tokens": 0,
        },
        "latency": {
            "first_semantic_write_ms": {"p50": 640.0},
        },
    }
    baseline = {
        **metrics,
        "e_mt": _metric_block(10.0),
    }
    return {"candidate": metrics, "stage_a": baseline}


def test_quality_first_ranking_penalizes_structure_errors() -> None:
    clean = _arm(_comparison_summary(9.0, malformed=0))
    malformed = _arm(_comparison_summary(20.0, malformed=1))
    assert ranking_key(clean) > ranking_key(malformed)
