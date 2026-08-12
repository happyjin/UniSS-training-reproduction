from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.uniss_phase3_event_rollout_joint_pilot15_v2.evaluation import (
    evaluate_checkpoint,
)


def test_v2_evaluator_uses_repaired_provenance() -> None:
    source = Path(evaluate_checkpoint.__file__).read_text(encoding="utf-8")
    assert '"version": "uniss_phase3_event_rollout_joint_pilot15_v2"' in source
    assert '"repair": "trainable_causal_frontend"' in source
    assert '"forced_write": False' in source


def test_v2_shell_wrapper_is_non_overwriting_and_configurable() -> None:
    wrapper = Path(evaluate_checkpoint.__file__).with_suffix(".sh")
    source = wrapper.read_text(encoding="utf-8")
    assert 'RUN_NAME="${RUN_NAME:-' in source
    assert '[[ ! -e "${OUTPUT}" ]]' in source
    assert "uniss_phase3_event_rollout_joint_pilot15_v2.evaluation" in source
    assert 'FUSE_TICKS="${FUSE_TICKS:-1}"' in source
    assert 'STATIC_CACHE="${STATIC_CACHE:-1}"' in source
    assert 'REPO_ROOT="$(cd "${EVAL_DIR}/../../.." && pwd)"' in source


def test_audio_audit_distinguishes_playable_pcm_from_silence(tmp_path: Path) -> None:
    silence = tmp_path / "silence.wav"
    speech = tmp_path / "speech.wav"
    sf.write(silence, np.zeros(1600, dtype=np.float32), 16000)
    sf.write(speech, np.full(1600, 0.1, dtype=np.float32), 16000)
    silent_audit = evaluate_checkpoint.audio_audit(silence)
    speech_audit = evaluate_checkpoint.audio_audit(speech)
    assert silent_audit["severe_semantic_collapse"] is True
    assert speech_audit["severe_semantic_collapse"] is False
    assert speech_audit["translation_audio_finite"] is True
    assert speech_audit["translation_audio_samples"] == 1600
