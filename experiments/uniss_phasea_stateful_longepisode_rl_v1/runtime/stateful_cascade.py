"""Stateful Phase-A ASR -> incremental MT -> acknowledged semantic TTS runtime.

The acoustic frontend is never reset at an artificial long-form window.  A
bounded speech-embedding ring limits LLM prompt growth, while committed ASR and
MT text, the TTS queue, the speaker condition, and the playback clock remain
append-only for the entire source session.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from training import constants_uniss as c
from training import sample_builders as builders

from .state import StreamingSessionState


SAMPLE_RATE = 16_000
PHYSICAL_BLOCK_MS = 160
PHYSICAL_BLOCK_SAMPLES = SAMPLE_RATE * PHYSICAL_BLOCK_MS // 1000


def text_content(tokens: Sequence[int]) -> list[int]:
    values = [int(value) for value in tokens]
    stop = values.index(c.TOKEN_END_CONTENT) if c.TOKEN_END_CONTENT in values else len(values)
    return [value for value in values[:stop] if value <= c.QWEN_BASE_VOCAB_END]


def semantic_content(tokens: Sequence[int]) -> list[int]:
    return [
        c.BICODEC_SEMANTIC_SPAN.value_for(int(value))
        for value in tokens
        if c.BICODEC_SEMANTIC_OFFSET <= int(value) <= c.BICODEC_SEMANTIC_SPAN.last_id
    ]


def normalized_text(tokenizer, values: Sequence[int], language: str) -> str:
    text = " ".join(tokenizer.decode(list(values), skip_special_tokens=True).split())
    return text.replace(" ", "") if language == "cmn" else text


def _ends_phrase(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", ",", ";", ":", "。", "？", "！", "，", "；", "："))


def phrase_ready(text: str, language: str, *, final: bool) -> bool:
    value = " ".join(text.split())
    if not value:
        return False
    if final or _ends_phrase(value):
        return True
    if language == "cmn":
        return len(value.replace(" ", "")) >= 10
    return len(value.split()) >= 7


def split_ready_prefixes(
    token_ids: Sequence[int],
    tokenizer,
    language: str,
    *,
    final: bool,
) -> tuple[list[tuple[list[int], str]], list[int]]:
    """Split committed target IDs into short speakable append-only phrases."""

    remaining = [int(value) for value in token_ids]
    ready: list[tuple[list[int], str]] = []
    minimum = 6 if language == "cmn" else 3
    maximum = 16 if language == "cmn" else 10
    while remaining:
        chosen = 0
        for end in range(1, len(remaining) + 1):
            text = normalized_text(tokenizer, remaining[:end], language)
            units = len(text.replace(" ", "")) if language == "cmn" else len(text.split())
            if units >= minimum and (_ends_phrase(text) or units >= maximum):
                chosen = end
                break
        if chosen == 0:
            whole = normalized_text(tokenizer, remaining, language)
            if final or phrase_ready(whole, language, final=False):
                chosen = len(remaining)
            else:
                break
        ids = remaining[:chosen]
        text = normalized_text(tokenizer, ids, language)
        if text:
            ready.append((ids, text))
        remaining = remaining[chosen:]
    return ready, remaining


def accept_mt_candidate(
    generated: Sequence[int],
    candidate_text_ids: Sequence[int],
    *,
    source_final: bool,
    minimum_nonfinal_tokens: int = 2,
) -> tuple[bool, str]:
    """Reject obvious non-final early END without pretending a window is EOS."""

    values = [int(value) for value in generated]
    ended = c.TOKEN_END_CONTENT in values or c.TOKEN_EOS in values
    if source_final:
        return True, "true_source_final"
    if ended and len(candidate_text_ids) < int(minimum_nonfinal_tokens):
        return False, "rejected_early_end"
    return True, "accepted"


def waveform_health(waveform: np.ndarray) -> dict[str, float | bool | int]:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(values).all())
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if len(values) else 0.0
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    non_silent = float(np.mean(np.abs(values) >= 1.0e-4)) if len(values) else 0.0
    healthy = bool(len(values) and finite and rms >= 1.0e-5 and non_silent >= 0.01 and peak < 1.2)
    return {
        "samples": len(values),
        "duration_seconds": len(values) / SAMPLE_RATE,
        "finite": finite,
        "rms": rms,
        "peak": peak,
        "non_silent_fraction": non_silent,
        "healthy": healthy,
    }


def maximum_internal_silence_ms(waveform: np.ndarray) -> float:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    block = SAMPLE_RATE // 10
    active = [
        bool(np.sqrt(np.mean(np.square(values[start : start + block], dtype=np.float64))) >= 1.0e-4)
        for start in range(0, len(values), block)
        if len(values[start : start + block])
    ]
    indices = [index for index, value in enumerate(active) if value]
    if not indices:
        return float(len(active) * 100)
    best = run = 0
    for value in active[indices[0] : indices[-1] + 1]:
        if value:
            run = 0
        else:
            run += 1
            best = max(best, run)
    return float(best * 100)


def timeline_audio(events: Sequence[tuple[int, np.ndarray]]) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    schedule: list[dict[str, float | int]] = []
    cursor = 0
    for source_ms, audio in events:
        available = int(round(source_ms * SAMPLE_RATE / 1000.0))
        start = max(cursor, available)
        stop = start + len(audio)
        schedule.append(
            {
                "source_available_ms": int(source_ms),
                "playback_start_ms": start * 1000.0 / SAMPLE_RATE,
                "playback_stop_ms": stop * 1000.0 / SAMPLE_RATE,
                "audio_samples": len(audio),
            }
        )
        cursor = stop
    output = np.zeros(cursor, dtype=np.float32)
    for row, (_, audio) in zip(schedule, events):
        start = int(round(float(row["playback_start_ms"]) * SAMPLE_RATE / 1000.0))
        output[start : start + len(audio)] = audio
    return output, schedule


def write_stereo(source: np.ndarray, translation: np.ndarray, path: Path) -> None:
    length = max(len(source), len(translation))
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[: len(source), 0] = source
    stereo[: len(translation), 1] = translation
    sf.write(path, stereo, SAMPLE_RATE, subtype="PCM_16")


def generate_semantic_with_continuation(
    *,
    generate_fn: Callable[..., list[int]],
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    seed: int,
    maximum_per_pass: int,
    maximum_passes: int,
) -> tuple[list[int], bool, int]:
    """Continue semantic generation when one pass reaches its token cap."""

    semantic: list[int] = []
    ended = False
    passes = 0
    for pass_index in range(int(maximum_passes)):
        continuation_prompt = [*prompt_ids, *c.encode_bicodec_semantic(semantic)]
        generated = generate_fn(
            model,
            tokenizer,
            prompt_ids=continuation_prompt,
            speech_embeddings=None,
            stop_ids={c.TOKEN_END_SEMANTIC, c.TOKEN_EOS},
            maximum=maximum_per_pass,
            seed=seed + pass_index,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
        )
        passes += 1
        semantic.extend(semantic_content(generated))
        ended = c.TOKEN_END_SEMANTIC in generated or c.TOKEN_EOS in generated
        if ended or len(generated) < maximum_per_pass:
            break
    return semantic, ended, max(0, passes - 1)


@torch.inference_mode()
def evaluate_stateful_session(
    row,
    *,
    decision_chunk_ms: int,
    acoustic_rollover_ms: int,
    model,
    tokenizer,
    objective,
    bicodec,
    generate_fn: Callable[..., list[int]],
    output: Path,
    seed: int,
) -> dict[str, object]:
    sample_id = str(row["id"])
    source, rate = sf.read(row["source_audio"], dtype="float32", always_2d=True)
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"source rate is not 16 kHz: {row['source_audio']}")
    source = np.asarray(source.mean(axis=1), dtype=np.float32)
    if not len(source) or not np.isfinite(source).all():
        raise ValueError(f"invalid source audio: {row['source_audio']}")

    sample_root = output / sample_id
    segment_root = sample_root / "segments"
    sample_root.mkdir(parents=True)
    segment_root.mkdir()
    source_path = sample_root / "source.wav"
    sf.write(source_path, source, SAMPLE_RATE, subtype="PCM_16")

    frontend = make_cached_frontend(objective, next(model.parameters()).device)
    state = StreamingSessionState()
    target_staging_ids: list[int] = []
    audio_events: list[tuple[int, np.ndarray]] = []
    event_rows: list[dict[str, object]] = []
    semantic_all: list[int] = []
    rejected_early_end = 0
    semantic_continuations = 0
    tts_failures = 0
    event_index = 0
    next_decision_ms = int(decision_chunk_ms)
    started = time.perf_counter()

    def global_asr_ids() -> list[int]:
        return [*state.asr_committed_ids, *state.asr_segment_committer.committed]

    for block_start in range(0, len(source), PHYSICAL_BLOCK_SAMPLES):
        block_stop = min(len(source), block_start + PHYSICAL_BLOCK_SAMPLES)
        true_final = block_stop == len(source)
        frontend_output = frontend.push(
            source[block_start:block_stop],
            state.frontend_state,
            is_final=true_final,
        )
        state.frontend_state = frontend_output.state
        hidden = frontend_output.pre_vq_hidden[0].to(
            device=next(objective.parameters()).device,
            dtype=objective.bridge_norm.weight.dtype,
        )
        codes = objective._nearest_codes(hidden)
        residual = objective.bridge_projection(objective.bridge_norm(hidden))
        base = model.get_input_embeddings()(codes.long() + c.GLM_SEMANTIC_OFFSET)
        state.speech_embedding_ring.append(base + residual.to(base.dtype))
        source_end_ms = int(frontend_output.source_end_ms)
        if not (true_final or source_end_ms >= next_decision_ms):
            continue
        while next_decision_ms <= source_end_ms:
            next_decision_ms += decision_chunk_ms

        speech = torch.cat(state.speech_embedding_ring, dim=0)
        asr_prompt = builders.build_asr_sample(
            source_glm=[0] * len(speech),
            bicodec_global=row["_stage_a_fixed_speaker_global"],
            src_lang=row["src_lang"],
            transcription="placeholder",
            text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
            source_id=sample_id,
        )
        asr_generated = generate_fn(
            model,
            tokenizer,
            prompt_ids=asr_prompt.prompt_ids,
            speech_embeddings=speech,
            stop_ids={c.TOKEN_END_CONTENT, c.TOKEN_EOS},
            maximum=128,
            seed=seed + event_index,
        )
        asr_candidate = text_content(asr_generated)
        asr_new = state.asr_segment_committer.update(asr_candidate, final=true_final)
        if true_final:
            state.asr_committed_ids.extend(state.asr_segment_committer.committed)
            state.asr_segment_committer.committed.clear()
            state.mark_true_final()

        ring_elapsed = source_end_ms - state.speech_ring_start_ms
        rollover = not true_final and ring_elapsed >= int(acoustic_rollover_ms)
        if rollover:
            forced = state.finalize_asr_segment()
            asr_new.extend(forced)
            state.speech_ring_start_ms = source_end_ms

        complete_asr = global_asr_ids()
        asr_text = normalized_text(tokenizer, complete_asr, str(row["src_lang"]))
        mt_candidate: list[int] = []
        mt_new: list[int] = []
        mt_acceptance = "not_run"
        if asr_text and (asr_new or true_final):
            mt_prompt = builders.build_mt_sample(
                src_lang=row["src_lang"],
                tgt_lang=row["tgt_lang"],
                source_text=asr_text,
                target_text="placeholder",
                text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
                source_id=sample_id,
            )
            prefix_prompt = [*mt_prompt.prompt_ids, *state.mt_committer.committed]
            mt_generated = generate_fn(
                model,
                tokenizer,
                prompt_ids=prefix_prompt,
                speech_embeddings=None,
                stop_ids={c.TOKEN_END_CONTENT, c.TOKEN_EOS},
                maximum=96,
                seed=seed + 10_000 + event_index,
            )
            mt_candidate = text_content(mt_generated)
            accepted, mt_acceptance = accept_mt_candidate(
                mt_generated,
                mt_candidate,
                source_final=true_final,
            )
            if accepted:
                mt_new = state.mt_committer.update(mt_candidate, final=true_final)
                target_staging_ids.extend(mt_new)
            else:
                rejected_early_end += 1

        ready, target_staging_ids = split_ready_prefixes(
            target_staging_ids,
            tokenizer,
            str(row["tgt_lang"]),
            final=true_final,
        )
        for ids, phrase in ready:
            state.tts_queue.append(ids, phrase, source_end_ms)

        emitted_items: list[dict[str, object]] = []
        for item in list(state.tts_queue.pending):
            if item.attempts >= 3 and not true_final:
                continue
            state.tts_queue.begin(item.item_id)
            tts_prompt = builders.build_tts_sample(
                bicodec_global=row["bicodec_global"],
                src_lang=row["tgt_lang"],
                transcription=item.text,
                source_bicodec=[0],
                text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
                source_id=f"{sample_id}:tts:{item.item_id}",
            )
            semantic, ended, continuations = generate_semantic_with_continuation(
                generate_fn=generate_fn,
                model=model,
                tokenizer=tokenizer,
                prompt_ids=tts_prompt.prompt_ids,
                seed=seed + 20_000 + item.item_id * 100,
                maximum_per_pass=320,
                maximum_passes=4,
            )
            semantic_continuations += continuations
            waveform = np.zeros(0, dtype=np.float32)
            if semantic:
                decode_tokens = torch.tensor(
                    [*row["bicodec_global"], *semantic],
                    dtype=torch.long,
                    device=next(model.parameters()).device,
                )
                waveform = np.asarray(
                    bicodec.decode_tokens_to_audio(decode_tokens), dtype=np.float32
                ).reshape(-1)
            health = waveform_health(waveform)
            state.tts_queue.acknowledge(
                item.item_id,
                semantic_tokens=len(semantic),
                continuation_count=continuations,
                audio_samples=len(waveform),
                healthy=bool(health["healthy"]),
            )
            segment_path = None
            if health["healthy"]:
                segment_path = segment_root / f"tts_{item.item_id:04d}_{item.source_available_ms}ms.wav"
                sf.write(segment_path, waveform, SAMPLE_RATE, subtype="PCM_16")
                audio_events.append((item.source_available_ms, waveform))
                semantic_all.extend(semantic)
            else:
                tts_failures += 1
            emitted_items.append(
                {
                    "item_id": item.item_id,
                    "text": item.text,
                    "source_available_ms": item.source_available_ms,
                    "semantic_tokens": len(semantic),
                    "semantic_ended": ended,
                    "semantic_continuations": continuations,
                    "audio_health": health,
                    "segment_audio_path": str(segment_path.resolve()) if segment_path else None,
                    "acknowledged": bool(health["healthy"]),
                }
            )

        event_rows.append(
            {
                "event_index": event_index,
                "source_end_ms": source_end_ms,
                "true_source_final": true_final,
                "memory_rollover": rollover,
                "frontend_encoder_resets": int(state.frontend_state.encoder_resets),
                "visible_ring_speech_tokens": len(speech),
                "asr_candidate": normalized_text(tokenizer, asr_candidate, str(row["src_lang"])),
                "asr_new_commit": normalized_text(tokenizer, asr_new, str(row["src_lang"])),
                "asr_committed": asr_text,
                "mt_candidate_delta": normalized_text(tokenizer, mt_candidate, str(row["tgt_lang"])),
                "mt_new_commit": normalized_text(tokenizer, mt_new, str(row["tgt_lang"])),
                "mt_acceptance": mt_acceptance,
                "mt_committed": normalized_text(tokenizer, state.mt_committer.committed, str(row["tgt_lang"])),
                "unqueued_target_tokens": len(target_staging_ids),
                "tts_pending_items": len(state.tts_queue.pending),
                "tts_emissions": emitted_items,
            }
        )
        event_index += 1

    continuous = (
        np.concatenate([audio for _, audio in audio_events])
        if audio_events
        else np.zeros(0, dtype=np.float32)
    )
    timeline, schedule = timeline_audio(audio_events)
    continuous_path = sample_root / "translation_continuous.wav"
    timeline_path = sample_root / "translation_global_timeline.wav"
    stereo_path = sample_root / "stereo_left_source_right_translation.wav"
    sf.write(continuous_path, continuous, SAMPLE_RATE, subtype="PCM_16")
    sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
    write_stereo(source, timeline, stereo_path)

    source_duration_ms = int(round(len(source) * 1000 / SAMPLE_RATE))
    first_source_ms = audio_events[0][0] if audio_events else None
    write_source_times = [int(value) for value, _ in audio_events]
    inter_write = [right - left for left, right in zip(write_source_times, write_source_times[1:])]
    processing_seconds = time.perf_counter() - started
    target_text = normalized_text(tokenizer, state.mt_committer.committed, str(row["tgt_lang"]))
    asr_text = normalized_text(tokenizer, global_asr_ids(), str(row["src_lang"]))
    return {
        "sample_id": sample_id,
        "src_lang": row["src_lang"],
        "tgt_lang": row["tgt_lang"],
        "source_audio": str(Path(row["source_audio"]).resolve()),
        "source_duration_ms": source_duration_ms,
        "decision_chunk_ms": int(decision_chunk_ms),
        "physical_acoustic_block_ms": PHYSICAL_BLOCK_MS,
        "acoustic_rollover_ms": int(acoustic_rollover_ms),
        "runtime_mode": "stateful_causal_frontend_bounded_embedding_ring_append_only_text_and_audio",
        "generated_streaming_transcription": asr_text,
        "generated_streaming_translation": target_text,
        "events": event_rows,
        "memory_rollovers": state.memory_rollovers,
        "frontend_encoder_resets": int(state.frontend_state.encoder_resets),
        "artificial_boundary_finalizations": state.artificial_boundary_finalizations,
        "rejected_early_end": rejected_early_end,
        "audio_writes": len(audio_events),
        "first_audio_source_ms": first_source_ms,
        "prefinal_audio_emitted": bool(first_source_ms is not None and first_source_ms < source_duration_ms),
        "inter_write_gap_ms": {
            "mean": float(np.mean(inter_write)) if inter_write else None,
            "p50": float(np.percentile(inter_write, 50)) if inter_write else None,
            "p95": float(np.percentile(inter_write, 95)) if inter_write else None,
            "maximum": max(inter_write) if inter_write else None,
        },
        "semantic_tokens": len(semantic_all),
        "semantic_continuations": semantic_continuations,
        "tts_failures": tts_failures,
        "tts_pending_unspoken_items": len(state.tts_queue.pending),
        "tts_pending_unspoken_text": [item.text for item in state.tts_queue.pending],
        "continuous_audio_path": str(continuous_path.resolve()),
        "timeline_audio_path": str(timeline_path.resolve()),
        "stereo_audio_path": str(stereo_path.resolve()),
        "continuous_audio_health": waveform_health(continuous),
        "timeline_audio_health": waveform_health(timeline),
        "maximum_internal_timeline_silence_ms": maximum_internal_silence_ms(timeline),
        "translation_audio_to_source_duration_ratio": len(continuous) / max(1, len(source)),
        "processing_seconds": processing_seconds,
        "rtf": processing_seconds / max(1e-6, source_duration_ms / 1000.0),
        "playback_schedule": schedule,
        "stateful_runtime_passed": bool(
            first_source_ms is not None
            and first_source_ms < source_duration_ms
            and waveform_health(continuous)["healthy"]
            and asr_text
            and target_text
        ),
        "claim_boundary": (
            "The causal acoustic frontend, committed ASR/MT text, TTS queue and playback clock "
            "persist for the full source. The LLM acoustic prompt uses a bounded ring and is "
            "recomputed at each decision; this runtime does not yet claim LLM KV-cache reuse."
        ),
    }


__all__ = [
    "accept_mt_candidate",
    "evaluate_stateful_session",
    "generate_semantic_with_continuation",
    "phrase_ready",
    "split_ready_prefixes",
    "waveform_health",
]
