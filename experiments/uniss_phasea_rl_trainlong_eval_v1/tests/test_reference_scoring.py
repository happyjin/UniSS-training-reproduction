from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_phasea_rl_trainlong_eval_v1.evaluation.score_results import (
    audit_wav,
    edit_distance,
    lcs_length,
    ngram_repetition,
    percentile,
    score_row,
    summarize,
)


def test_reference_metric_primitives() -> None:
    assert edit_distance(list("abcd"), list("abxd")) == 1
    assert lcs_length(list("abcdef"), list("abqdef")) == 5
    value = ngram_repetition("abcabcabc".split(), order=4)
    assert value["total"] == 0
    repeated = ngram_repetition(list("abcdabcd"), order=4)
    assert repeated["repeated_occurrences"] == 1
    assert percentile([0.0, 10.0, 20.0], 0.95) == 19.0


def test_independent_wav_audit(tmp_path: Path) -> None:
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    wave = np.full(16_000, 0.1, dtype=np.float32)
    sf.write(mono, wave, 16_000)
    sf.write(stereo, np.stack([wave, wave], axis=1), 16_000)
    assert audit_wav(str(mono), 1)["healthy"]
    assert audit_wav(str(stereo), 2)["healthy"]
    assert not audit_wav(str(stereo), 1)["healthy"]


def test_score_row_uses_installed_sacrebleu_api(tmp_path: Path) -> None:
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    wave = np.full(16_000, 0.1, dtype=np.float32)
    sf.write(mono, wave, 16_000)
    sf.write(stereo, np.stack([wave, wave], axis=1), 16_000)
    result = {
        "sample_id": "example",
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "generated_streaming_transcription": "hello world",
        "generated_streaming_translation": "你好世界",
        "continuous_audio_path": str(mono),
        "timeline_audio_path": str(mono),
        "stereo_audio_path": str(stereo),
        "playback_schedule": [
            {"source_available_ms": 640},
            {"source_available_ms": 1280},
        ],
    }
    reference = {
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "reference_transcription": "hello world",
        "reference_translation": "你好世界",
        "rl_train_seen": True,
        "formal_rollout_seen": True,
        "validation_overlap": False,
        "component_count": 1,
        "source_audio_sha256": "hash",
    }
    value = score_row(result, reference)["reference_metrics"]
    assert value["asr_error_rate"] == 0.0
    assert value["mt_sentence_chrf"] == 100.0
    assert value["final_translation_lcs_coverage"] == 1.0


def test_summary_falls_back_to_runtime_gap_summary() -> None:
    row = {
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "source_duration_ms": 2_000,
        "generated_streaming_translation": "你好",
        "reference_translation": "你好",
        "reference_metrics": {
            "asr_metric": "wer",
            "asr_errors": 0,
            "asr_reference_units": 2,
            "asr_normalized_similarity": 1.0,
            "final_translation_lcs_coverage": 1.0,
            "translation_length_ratio": 1.0,
            "translation_4gram_repetition": {
                "repetition_rate": 0.0,
                "maximum_frequency": 0,
            },
            "write_gaps_ms": [],
            "independent_wav_audit": {
                key: {"healthy": True} for key in ("continuous", "timeline", "stereo")
            },
        },
        "first_audio_source_ms": 640,
        "inter_write_gap_ms": {"mean": 640, "p50": 640, "p95": 900, "maximum": 960},
        "maximum_internal_timeline_silence_ms": 0,
        "translation_audio_to_source_duration_ratio": 1.0,
        "audio_writes": 3,
        "prefinal_audio_emitted": True,
        "tts_pending_unspoken_items": 0,
        "tts_failures": 0,
        "rejected_early_end": 0,
        "semantic_continuations": 0,
        "stateful_runtime_passed": True,
        "rtf": 1.0,
    }
    value = summarize([row])
    assert value["write_gap_ms"]["observed"] == 2
    assert value["write_gap_ms"]["p95"] == 900
    assert value["write_gap_ms"]["maximum"] == 960
