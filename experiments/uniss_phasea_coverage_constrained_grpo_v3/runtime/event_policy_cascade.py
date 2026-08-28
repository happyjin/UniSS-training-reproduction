"""Stateful Phase-A cascade with an explicit sampled WAIT/WRITE policy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.state import (
    StreamingSessionState,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    PHYSICAL_BLOCK_SAMPLES,
    SAMPLE_RATE,
    accept_mt_candidate,
    generate_semantic_with_continuation,
    maximum_internal_silence_ms,
    normalized_text,
    semantic_content,
    split_ready_prefixes,
    text_content,
    timeline_audio,
    waveform_health,
    write_stereo,
)
from training import constants_uniss as c
from training import sample_builders as builders


def observe_asr_candidate(state: StreamingSessionState, candidate: Sequence[int]) -> None:
    """Update stability history on WAIT without committing model output."""

    current = [int(value) for value in candidate]
    committed = state.asr_segment_committer.committed
    if current[: len(committed)] != committed:
        state.asr_segment_committer.revision_conflicts += 1
    state.asr_segment_committer.previous = current


def micro_ready_prefixes(
    token_ids: Sequence[int],
    tokenizer,
    language: str,
    *,
    final: bool,
    write_requested: bool,
) -> tuple[list[tuple[list[int], str]], list[int]]:
    ready, remaining = split_ready_prefixes(
        token_ids, tokenizer, language, final=final
    )
    if ready or not write_requested or not remaining:
        return ready, remaining
    text = normalized_text(tokenizer, remaining, language)
    units = len(text.replace(" ", "")) if language == "cmn" else len(text.split())
    minimum = 2
    if units < minimum:
        return ready, remaining
    return [(list(remaining), text)], []


@torch.inference_mode()
def evaluate_event_policy_session(
    row,
    *,
    decision_chunk_ms: int,
    acoustic_rollover_ms: int,
    model,
    tokenizer,
    objective,
    bicodec,
    generate_fn: Callable[..., list[int]],
    action_fn: Callable[..., str],
    event_context_fn: Callable[[int], None],
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
    rejected_early_end = semantic_continuations = tts_failures = 0
    event_index = 0
    next_decision_ms = int(decision_chunk_ms)
    last_executed_write_ms = 0
    started = time.perf_counter()

    def global_asr_ids() -> list[int]:
        return [*state.asr_committed_ids, *state.asr_segment_committer.committed]

    for block_start in range(0, len(source), PHYSICAL_BLOCK_SAMPLES):
        block_stop = min(len(source), block_start + PHYSICAL_BLOCK_SAMPLES)
        true_final = block_stop == len(source)
        frontend_output = frontend.push(
            source[block_start:block_stop], state.frontend_state, is_final=true_final
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
        event_context_fn(event_index)
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
        sampled_action = "FLUSH" if true_final else action_fn(
            model,
            tokenizer,
            speech_embeddings=speech,
            target_lang=str(row["tgt_lang"]),
            speaker_global=row["bicodec_global"],
            seed=seed + 50_000 + event_index,
        )
        deadline_forced_write = bool(
            not true_final
            and sampled_action == "WAIT"
            and source_end_ms - last_executed_write_ms >= 4_000
        )
        executed_action = (
            "WRITE"
            if true_final or sampled_action == "WRITE" or deadline_forced_write
            else "WAIT"
        )
        if executed_action == "WRITE":
            last_executed_write_ms = source_end_ms
        asr_new: list[int] = []
        if executed_action == "WRITE":
            asr_new = state.asr_segment_committer.update(asr_candidate, final=true_final)
        else:
            observe_asr_candidate(state, asr_candidate)
        if true_final:
            state.asr_revision_conflicts_total += state.asr_segment_committer.revision_conflicts
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
        mt_acceptance = "policy_wait"
        if executed_action == "WRITE" and asr_text:
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
                mt_generated, mt_candidate, source_final=true_final
            )
            if accepted:
                mt_new = state.mt_committer.update(mt_candidate, final=true_final)
                target_staging_ids.extend(mt_new)
            else:
                rejected_early_end += 1

        ready, target_staging_ids = micro_ready_prefixes(
            target_staging_ids,
            tokenizer,
            str(row["tgt_lang"]),
            final=true_final,
            write_requested=executed_action == "WRITE",
        )
        for ids, phrase in ready:
            state.tts_queue.append(ids, phrase, source_end_ms)

        emitted_items: list[dict[str, object]] = []
        if executed_action == "WRITE":
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
        flush_complete = bool(
            true_final and not target_staging_ids and not state.tts_queue.pending
        )
        event_rows.append(
            {
                "event_index": event_index,
                "source_end_ms": source_end_ms,
                "true_source_final": true_final,
                "policy_action": sampled_action,
                "executed_action": executed_action,
                "deadline_forced_write": deadline_forced_write,
                "flush_complete": flush_complete,
                "memory_rollover": rollover,
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

    continuous = np.concatenate([audio for _, audio in audio_events]) if audio_events else np.zeros(0, dtype=np.float32)
    timeline, schedule = timeline_audio(audio_events)
    continuous_path = sample_root / "translation_continuous.wav"
    timeline_path = sample_root / "translation_global_timeline.wav"
    stereo_path = sample_root / "stereo_left_source_right_translation.wav"
    sf.write(continuous_path, continuous, SAMPLE_RATE, subtype="PCM_16")
    sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
    write_stereo(source, timeline, stereo_path)
    source_duration_ms = int(round(len(source) * 1000 / SAMPLE_RATE))
    first_source_ms = audio_events[0][0] if audio_events else None
    write_times = [int(value) for value, _ in audio_events]
    gaps = [right - left for left, right in zip(write_times, write_times[1:])]
    processing_seconds = time.perf_counter() - started
    result = {
        "sample_id": sample_id,
        "src_lang": row["src_lang"],
        "tgt_lang": row["tgt_lang"],
        "source_audio": str(Path(row["source_audio"]).resolve()),
        "source_duration_ms": source_duration_ms,
        "decision_chunk_ms": int(decision_chunk_ms),
        "acoustic_rollover_ms": int(acoustic_rollover_ms),
        "runtime_mode": "phase_a_stateful_cascade_explicit_wait_write_event_policy_v2",
        "generated_streaming_transcription": normalized_text(tokenizer, global_asr_ids(), str(row["src_lang"])),
        "generated_streaming_translation": normalized_text(tokenizer, state.mt_committer.committed, str(row["tgt_lang"])),
        "events": event_rows,
        "memory_rollovers": state.memory_rollovers,
        "asr_revision_conflicts": state.asr_revision_conflicts_total,
        "mt_revision_conflicts": state.mt_committer.revision_conflicts,
        "frontend_encoder_resets": int(state.frontend_state.encoder_resets),
        "artificial_boundary_finalizations": state.artificial_boundary_finalizations,
        "rejected_early_end": rejected_early_end,
        "audio_writes": len(audio_events),
        "first_audio_source_ms": first_source_ms,
        "prefinal_audio_emitted": bool(first_source_ms is not None and first_source_ms < source_duration_ms),
        "inter_write_gap_ms": {
            "mean": float(np.mean(gaps)) if gaps else None,
            "p50": float(np.percentile(gaps, 50)) if gaps else None,
            "p95": float(np.percentile(gaps, 95)) if gaps else None,
            "maximum": max(gaps) if gaps else None,
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
            and global_asr_ids()
            and state.mt_committer.committed
        ),
        "claim_boundary": (
            "The causal frontend and committed text persist, but the LLM acoustic prompt is "
            "still recomputed and synchronous TTS still affects wall-clock RTF."
        ),
    }
    (sample_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "evaluate_event_policy_session",
    "micro_ready_prefixes",
    "observe_asr_candidate",
]
