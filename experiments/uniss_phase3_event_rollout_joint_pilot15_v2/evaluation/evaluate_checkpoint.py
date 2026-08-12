#!/usr/bin/env python3
"""Strict real-PCM evaluation for repaired fixed15 event-rollout checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

import web_demo.runtime_parity_streaming_v2.evaluate_checkpoint as runtime_eval
from experiments.uniss_phase3_event_rollout_joint_pilot15_v1.evaluation.model_loader import (
    load_runtime_models,
)
from experiments.uniss_phase3_runtime_parity_streaming_v2.generalize15_action_eos_calibration.inference import (
    CalibratedMicroblockRuntimeGenerator,
)
from web_demo.runtime_parity_streaming_v5.evaluate_checkpoint import (
    WarmedBiCodecTokenizer,
)


def audio_audit(path: Path) -> dict[str, object]:
    """Record objective PCM evidence instead of treating file creation as success."""

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 2:
        value = value.mean(axis=1)
    value = value.reshape(-1)
    finite = bool(np.isfinite(value).all())
    rms = float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))) if len(value) else 0.0
    peak = float(np.max(np.abs(value))) if len(value) else 0.0
    non_silent_fraction = float(np.mean(np.abs(value) >= 1.0e-4)) if len(value) else 0.0
    severe_collapse = (
        not finite
        or len(value) == 0
        or rms < 1.0e-5
        or non_silent_fraction < 0.01
    )
    return {
        "translation_audio_samples": len(value),
        "translation_audio_sample_rate": int(sample_rate),
        "translation_audio_finite": finite,
        "translation_audio_rms": rms,
        "translation_audio_peak": peak,
        "translation_audio_non_silent_fraction": non_silent_fraction,
        "severe_semantic_collapse": severe_collapse,
    }


def evaluate(args):
    """Run the shared exact runtime while recording the repaired v2 provenance."""

    runtime_eval.load_runtime_models = load_runtime_models
    runtime_eval.NaturalRuntimeParityGenerator = CalibratedMicroblockRuntimeGenerator
    runtime_eval.BiCodecTokenizer = WarmedBiCodecTokenizer
    summary = runtime_eval.evaluate(args)
    formal_path = Path(summary["formal_manifest"])
    source_metadata: dict[str, dict[str, object]] = {}
    with formal_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row["id"])
            source_metadata[sample_id] = {
                "src_lang": str(row["src_lang"]),
                "tgt_lang": str(row["tgt_lang"]),
                "transcription": str(row.get("transcription", "")),
                "source_audio_path": str(Path(str(row["source_audio"])).resolve()),
                "target_audio_path": str(Path(str(row["target_audio"])).resolve()),
                "evaluation_split": row.get("_evaluation_split"),
                "evaluation_source_index": row.get("_evaluation_source_index"),
                "evaluation_shard_index": row.get("_evaluation_shard_index"),
            }
    output = Path(args.output)
    for row_index, sample in enumerate(summary["samples"]):
        sample.update(source_metadata[str(sample["sample_id"])])
        sample_root = output / f"{row_index:04d}_{sample['sample_id']}"
        sample.update(audio_audit(sample_root / "translation.wav"))
        sample["audio_path"] = str((sample_root / "translation.wav").resolve())
        sample["timeline_audio_path"] = str(
            (sample_root / "translation_timeline.wav").resolve()
        )
        sample["stereo_audio_path"] = str(
            (sample_root / "stereo_left_source_right_translation.wav").resolve()
        )
        (sample_root / "result.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary["schema_version"] = "uniss_event_rollout_fixed15_pcm_evaluation_v2"
    summary["runtime_training"] = {
        "version": "uniss_phase3_event_rollout_joint_pilot15_v2",
        "repair": "trainable_causal_frontend",
        "initial_checkpoint": "phase3_v4_iter_0009075",
        "data_scope": "unist_train_shards_00000_00014_only",
        "timing_classification": "pseudo_oracle_alignment",
        "natural_exact_timing": False,
        "persistent_kv": True,
        "event_rollout_recovery": True,
        "learned_wait_write": True,
        "learned_text": True,
        "learned_semantic_microblocks": True,
        "learned_continuation_eos": True,
        "forced_write": False,
        "forced_eos": False,
        "oracle_semantic_length": False,
        "forced_semantic_length": False,
        "safety_ceiling_is_failure": True,
    }
    (Path(args.output) / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    evaluate(runtime_eval.parse_args())
