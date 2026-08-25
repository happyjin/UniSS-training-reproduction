import numpy as np

from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.aggregate_listening import (
    summarize,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.bounded_longform import (
    SAMPLE_RATE,
    _silence_metrics,
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

