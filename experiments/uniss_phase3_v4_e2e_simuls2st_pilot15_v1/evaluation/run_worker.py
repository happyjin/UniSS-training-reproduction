#!/usr/bin/env python3
"""One-GPU worker for E-ASR, E-MT and E-S2S free-running validation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    validate_trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.gate import (
    WORKER_SCHEMA,
    incremental_text_metrics,
    write_new_json,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.evaluation.runtime import (
    PersistentInterleavedSession,
    incremental_mt_rollout,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.persistent_runtime import (
    _speech_embeddings,
    rollout_trajectory,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from training.simul_uniss.jsonl_index import load_index
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from uniss.streaming.bicodec_streamer import (
    StreamingBiCodecDecoder,
    bicodec_decode_function,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--candidate-hf", type=Path, required=True)
    parser.add_argument("--phase3-hf", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--candidate-hf-sha256", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-asr-event-tokens", type=int, default=96)
    parser.add_argument("--max-asr-final-tokens", type=int, default=8)
    parser.add_argument("--max-mt-tokens", type=int, default=192)
    parser.add_argument("--max-s2s-fragments", type=int, default=4)
    parser.add_argument("--max-s2s-text-tokens", type=int, default=48)
    parser.add_argument("--max-s2s-semantic-tokens", type=int, default=64)
    return parser.parse_args()


def _load_trajectory(
    path: Path, offsets: Sequence[int], record_index: int
) -> E2ETrajectory:
    if not 0 <= int(record_index) < len(offsets):
        raise IndexError("free-running selection record is outside gold trajectories")
    with path.open("rb") as handle:
        handle.seek(int(offsets[int(record_index)]))
        value = E2ETrajectory.from_mapping(json.loads(handle.readline()))
    validate_trajectory(value, require_audio_hash=True, require_audio_audit=True)
    return value


def _load_models(args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("free-running E2E validation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(
        args.candidate_hf, local_files_only=True
    )
    candidate = AutoModelForCausalLM.from_pretrained(
        args.candidate_hf,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    phase3 = AutoModelForCausalLM.from_pretrained(
        args.phase3_hf,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    if int(candidate.config.vocab_size) < len(tokenizer) or int(
        phase3.config.vocab_size
    ) < len(tokenizer):
        raise ValueError("free-running model vocabulary is smaller than tokenizer")
    objective = stage_a_eval.load_objective(
        args.v1_checkpoint, args.whispervq_model, device
    ).eval().requires_grad_(False)
    frontend = make_cached_frontend(objective, device)
    return device, tokenizer, candidate, phase3, objective, frontend


def _mt_value(
    rollout: Mapping[str, object], reference: str, language: str
) -> dict[str, object]:
    hypotheses = [str(value) for value in rollout["hypotheses"]]  # type: ignore[index]
    value = incremental_text_metrics(hypotheses, reference, language)
    value.update(
        {
            "raw_hypotheses": list(rollout["raw_hypotheses"]),
            "commit_conflicts": int(rollout["commit_conflicts"]),
            "unterminated_generations": int(
                rollout["unterminated_generations"]
            ),
        }
    )
    return value


def _decode_semantic_fragments(
    codec: BiCodecTokenizer,
    speaker_global: Sequence[int],
    fragments: Sequence[Sequence[int]],
) -> np.ndarray:
    streamer = StreamingBiCodecDecoder(bicodec_decode_function(codec))
    streamer.set_speaker_tokens(speaker_global)
    chunks: list[np.ndarray] = []
    for fragment in fragments:
        if fragment:
            chunk = streamer.push(fragment, is_final=False)
            if len(chunk):
                chunks.append(chunk)
    if fragments:
        tail = streamer.push((), is_final=True)
        if len(tail):
            chunks.append(tail)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _run_e_s2s(
    *,
    trajectory: E2ETrajectory,
    candidate,
    tokenizer,
    objective,
    frontend,
    codec: BiCodecTokenizer,
    args: argparse.Namespace,
) -> dict[str, object]:
    embeddings = _speech_embeddings(
        objective, frontend, candidate, trajectory
    )
    session = PersistentInterleavedSession(
        candidate, tokenizer, embeddings, trajectory
    )
    fragments: list[tuple[int, ...]] = []
    pre_eos_text = False
    pre_eos_semantic = False
    malformed = 0
    event_rows: list[dict[str, object]] = []
    for event in trajectory.events:
        if session.closed:
            malformed += 1
            break
        row = session.run_event(
            event,
            max_fragments=args.max_s2s_fragments,
            max_text_tokens=args.max_s2s_text_tokens,
            max_semantic_tokens=args.max_s2s_semantic_tokens,
        )
        fragments.append(row.semantic_tokens)
        malformed += row.malformed_segments
        if not event.source_final:
            pre_eos_text = pre_eos_text or bool(row.mt_deltas)
            pre_eos_semantic = pre_eos_semantic or bool(row.semantic_tokens)
        event_rows.append(
            {
                "event_index": row.event_index,
                "source_end_ms": row.source_end_ms,
                "source_final": row.source_final,
                "source_glm_start": row.source_glm_start,
                "source_glm_end": row.source_glm_end,
                "chosen_continuations": list(row.chosen_continuations),
                "asr_deltas": list(row.asr_deltas),
                "mt_deltas": list(row.mt_deltas),
                "semantic_tokens": len(row.semantic_tokens),
                "malformed_segments": row.malformed_segments,
            }
        )
    malformed += int(not session.closed)
    audio_error: str | None = None
    try:
        audio = _decode_semantic_fragments(
            codec, trajectory.speaker_global, fragments
        )
    except Exception as exc:  # fail closed while retaining the other sample metrics
        audio = np.zeros(0, dtype=np.float32)
        audio_error = f"{type(exc).__name__}: {exc}"
        malformed += 1
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(audio).all())
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) and finite else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) and finite else 0.0
    non_silent = bool(len(audio) and finite and rms > 1e-5 and peak > 1e-4)
    audio_path = args.audio_dir / f"{trajectory.sample_id}.wav"
    if audio_path.exists():
        raise FileExistsError(audio_path)
    if len(audio) and finite:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, audio, 16_000, subtype="PCM_16")
    return {
        "source_hypothesis": session.source_text,
        "target_hypothesis": session.target_text,
        "semantic_tokens": len(session.semantic),
        "semantic_reference_tokens": trajectory.target_semantic_length,
        "semantic_coverage": min(
            1.0, len(session.semantic) / max(1, trajectory.target_semantic_length)
        ),
        "invalid_semantic_tokens": sum(
            not 0 <= int(value) < 8192 for value in session.semantic
        ),
        "target_text_before_source_eos": pre_eos_text,
        "target_semantic_before_source_eos": pre_eos_semantic,
        "source_rollback_events": 0,
        "target_rollback_events": 0,
        "malformed_segments": malformed,
        "natural_eos": session.closed,
        "events": event_rows,
        "audio": {
            "path": str(audio_path.resolve()) if audio_path.exists() else None,
            "samples": len(audio),
            "duration_seconds": len(audio) / 16_000,
            "finite": finite,
            "rms": rms,
            "peak": peak,
            "non_silent": non_silent,
            "error": audio_error,
        },
    }


def main() -> None:
    args = parse_args()
    if args.report.exists() or args.audio_dir.exists():
        raise FileExistsError("refusing to overwrite free-running worker output")
    if len(args.candidate_hf_sha256) != 64:
        raise ValueError("candidate HF fingerprint is not a SHA256 digest")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = list(selection["records"])
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid free-running worker partition")
    local = [
        value
        for index, value in enumerate(selected)
        if index % args.num_workers == args.worker_index
    ]
    if not local:
        raise ValueError("free-running worker partition is empty")
    offsets = load_index(args.gold)
    if offsets is None:
        raise ValueError("gold trajectory offset index is missing")
    device, tokenizer, candidate, phase3, objective, frontend = _load_models(args)
    codec: BiCodecTokenizer | None = None
    samples: list[dict[str, object]] = []
    started = time.perf_counter()
    for position, selected_row in enumerate(local):
        trajectory = _load_trajectory(
            args.gold, offsets, int(selected_row["record_index"])
        )
        if trajectory.sample_id != str(selected_row["sample_id"]):
            raise RuntimeError("fixed selection sample ID differs from gold trajectory")
        asr_rollout = rollout_trajectory(
            trajectory,
            qwen=candidate,
            tokenizer=tokenizer,
            objective=objective,
            frontend=frontend,
            v1_hf_sha256=args.candidate_hf_sha256,
            max_event_tokens=args.max_asr_event_tokens,
            max_final_tokens=args.max_asr_final_tokens,
        )
        gold_source = [event.gold_source_prefix for event in trajectory.events]
        free_source = [event.v1_source_prefix for event in asr_rollout.events]
        candidate_gold = incremental_mt_rollout(
            candidate,
            tokenizer,
            gold_source,
            trajectory.tgt_lang,
            max_tokens=args.max_mt_tokens,
        )
        candidate_free = incremental_mt_rollout(
            candidate,
            tokenizer,
            free_source,
            trajectory.tgt_lang,
            max_tokens=args.max_mt_tokens,
        )
        phase3_gold = incremental_mt_rollout(
            phase3,
            tokenizer,
            gold_source,
            trajectory.tgt_lang,
            max_tokens=args.max_mt_tokens,
        )
        row: dict[str, object] = {
            "sample_id": trajectory.sample_id,
            "record_index": int(selected_row["record_index"]),
            "src_lang": trajectory.src_lang,
            "tgt_lang": trajectory.tgt_lang,
            "source_duration_ms": trajectory.source_duration_ms,
            "transcription_reference": trajectory.normalized_transcription,
            "translation_reference": trajectory.normalized_translation,
            "e_asr": {
                "hypothesis": asr_rollout.full_text,
                "metric": asr_rollout.metric,
                "errors": asr_rollout.errors,
                "reference_units": asr_rollout.reference_units,
                "error_rate": asr_rollout.error_rate,
                "source_rollbacks": 0,
                "empty_events": asr_rollout.empty_events,
                "early_eos_events": asr_rollout.early_eos_events,
                "malformed_write_events": asr_rollout.malformed_write_events,
                "final_reached_eos": asr_rollout.final_reached_eos,
                "event_hypotheses": [
                    event.v1_source_prefix for event in asr_rollout.events
                ],
            },
            "e_mt_gold": _mt_value(
                candidate_gold,
                trajectory.normalized_translation,
                trajectory.tgt_lang,
            ),
            "e_mt_free": _mt_value(
                candidate_free,
                trajectory.normalized_translation,
                trajectory.tgt_lang,
            ),
            "phase3_mt_gold": _mt_value(
                phase3_gold,
                trajectory.normalized_translation,
                trajectory.tgt_lang,
            ),
            "e_s2s_free": None,
        }
        if bool(selected_row.get("run_e_s2s")):
            if codec is None:
                codec = BiCodecTokenizer(args.bicodec_model, device=device)
                codec.model.eval().requires_grad_(False)
            row["e_s2s_free"] = _run_e_s2s(
                trajectory=trajectory,
                candidate=candidate,
                tokenizer=tokenizer,
                objective=objective,
                frontend=frontend,
                codec=codec,
                args=args,
            )
        samples.append(row)
        print(
            json.dumps(
                {
                    "worker": args.worker_index,
                    "completed": position + 1,
                    "worker_samples": len(local),
                    "sample_id": trajectory.sample_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "schema_version": WORKER_SCHEMA,
        "status": "complete",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "selection": str(args.selection.resolve()),
        "gold": str(args.gold.resolve()),
        "candidate_hf": str(args.candidate_hf.resolve()),
        "phase3_hf": str(args.phase3_hf.resolve()),
        "v1_checkpoint": str(args.v1_checkpoint.resolve()),
        "samples": samples,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_new_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

