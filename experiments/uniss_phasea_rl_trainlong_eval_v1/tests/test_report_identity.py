from pathlib import Path

from experiments.uniss_phasea_rl_trainlong_eval_v1.evaluation.write_report import (
    LABELS,
    paired_identity_audit,
)


def _row(sample_id: str, text: str, audio: Path) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "generated_streaming_transcription": "fixed asr",
        "generated_streaming_translation": text,
        "continuous_audio_path": str(audio),
        "timeline_audio_path": str(audio),
        "stereo_audio_path": str(audio),
    }


def test_paired_identity_audit_distinguishes_text_and_audio(tmp_path: Path) -> None:
    base_audio = tmp_path / "base.wav"
    changed_audio = tmp_path / "changed.wav"
    base_audio.write_bytes(b"base")
    changed_audio.write_bytes(b"changed")
    scored = {}
    for run_id in LABELS:
        changed = run_id == "rl_iter45_runtime_v2"
        scored[run_id] = {
            "results": [
                _row(
                    "sample",
                    "changed mt" if changed else "base mt",
                    changed_audio if changed else base_audio,
                )
            ]
        }

    samples, paired = paired_identity_audit(scored)

    assert samples == [
        {
            "sample_id": "sample",
            "asr_unique": 1,
            "mt_unique": 2,
            "continuous_unique": 2,
            "timeline_unique": 2,
            "stereo_unique": 2,
        }
    ]
    assert paired["rl_iter15_runtime_v2"]["mt_same"] == 1
    assert paired["rl_iter45_runtime_v2"] == {
        "asr_same": 1,
        "mt_same": 0,
        "continuous_same": 0,
        "timeline_same": 0,
        "stereo_same": 0,
    }
