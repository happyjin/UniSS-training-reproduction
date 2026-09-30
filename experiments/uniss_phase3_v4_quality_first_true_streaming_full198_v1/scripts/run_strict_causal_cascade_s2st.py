#!/usr/bin/env python3
"""Stage A strict-causal cascade S2ST demo.

Short clips keep the original 15-shard commit/flush contract.
Longer audio uses prefix-forced incremental ASR/MT so length is not capped
by a single 128-token regenerate, plus slower TTS flush and utterance reset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from training import constants_uniss as c
from training import sample_builders as builders
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer


SAMPLE_RATE = 16_000
PHYSICAL_BLOCK_MS = 160
PHYSICAL_BLOCK_SAMPLES = SAMPLE_RATE * PHYSICAL_BLOCK_MS // 1000
ALLOWED_CHUNKS = (160, 320, 640, 1280)
ALLOWED_LANGS = {"cmn", "eng"}
LONG_AUDIO_SECONDS = 15.0
STEP_NEW_TOKENS = 128
MT_IDLE_FINAL_TOKENS = 24
MT_GROWTH_CAP = 48
MAX_SPEECH_TOKENS = 4000
SILENCE_RMS = 0.008
SILENCE_BLOCKS = 4
LOOP_NGRAM = 8
PILOT15_DEMO_ROOT = Path(
    "/opt/dlami/nvme/neuhao/UniSS/eval_outputs/"
    "uniss_phase3_v4_quality_first_true_streaming_pilot15_v1/"
    "demo_audios_all_20260820T111500Z"
)


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    value = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        value += 1
    return value


class ConservativeCommitter:
    def __init__(self, holdback: int, *, recover: bool = False) -> None:
        self.holdback = int(holdback)
        self.recover = bool(recover)
        self.committed: list[int] = []
        self.previous: list[int] | None = None
        self.revision_conflicts = 0

    def reset(self) -> None:
        self.committed = []
        self.previous = None

    def update(self, candidate: Sequence[int], *, final: bool) -> list[int]:
        current = [int(value) for value in candidate]
        if current[: len(self.committed)] != self.committed:
            self.revision_conflicts += 1
            if not self.recover or not self.committed:
                self.previous = current
                return []
            keep = longest_common_prefix(self.committed, current)
            self.committed = self.committed[:keep]
            self.previous = current
            if current[:keep] != self.committed:
                return []
        if final:
            stable = len(current)
        elif self.previous is None:
            stable = len(self.committed)
        else:
            stable = max(
                len(self.committed),
                longest_common_prefix(self.previous, current) - self.holdback,
            )
        stable = min(stable, len(current))
        new = current[len(self.committed) : stable]
        self.committed.extend(new)
        self.previous = current
        return new


def token_tail_loops(values: Sequence[int], ngram: int = LOOP_NGRAM) -> bool:
    if len(values) < 2 * ngram:
        return False
    tail = tuple(int(value) for value in values[-ngram:])
    body = [int(value) for value in values[:-ngram]]
    for index in range(len(body) - ngram + 1):
        if tuple(body[index : index + ngram]) == tail:
            return True
    return False


def cut_token_loop(values: Sequence[int]) -> list[int]:
    ids = [int(value) for value in values]
    for ngram in (16, 12, LOOP_NGRAM):
        seen: dict[tuple[int, ...], int] = {}
        for index in range(0, len(ids) - ngram + 1):
            key = tuple(ids[index : index + ngram])
            if key in seen:
                return ids[:index]
            seen[key] = index
    return ids


def mt_new_token_budget(asr_new: Sequence[int], src_lang: str, *, final: bool) -> int:
    if not asr_new:
        return MT_IDLE_FINAL_TOKENS if final else 0
    grown = len(asr_new)
    if src_lang == "cmn":
        return min(MT_GROWTH_CAP, max(8, grown * 2 + 8))
    return min(MT_GROWTH_CAP, max(8, grown + 8))


def maximum_identical_run(values: Sequence[int]) -> int:
    best = current = 0
    previous = None
    for raw in values:
        value = int(raw)
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)
    return best


def audio_audit(path: Path) -> dict[str, object]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    finite = bool(np.isfinite(audio).all())
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    non_silent = float(np.mean(np.abs(mono) >= 1.0e-4)) if len(mono) else 0.0
    clipped = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 0.0
    return {
        "path": str(path.resolve()),
        "sample_rate": int(rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_seconds": float(len(mono) / rate) if rate else 0.0,
        "finite": finite,
        "rms": rms,
        "peak": peak,
        "non_silent_fraction": non_silent,
        "clipped_fraction": clipped,
        "healthy": bool(
            rate == SAMPLE_RATE
            and len(mono)
            and finite
            and rms >= 1.0e-5
            and non_silent >= 0.01
            and clipped < 0.01
        ),
    }


def apply_repetition_penalty(logits: torch.Tensor, history: Sequence[int], penalty: float) -> None:
    if penalty == 1.0 or not history:
        return
    indices = torch.tensor(sorted(set(int(value) for value in history)), device=logits.device)
    selected = logits.index_select(0, indices)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    logits.index_copy_(0, indices, selected)


def sample_token(logits: torch.Tensor, *, temperature: float, top_p: float, generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    values, indices = torch.sort(logits / temperature, descending=True)
    probabilities = torch.softmax(values, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    values[remove] = -torch.inf
    probabilities = torch.softmax(values, dim=-1)
    selected = int(torch.multinomial(probabilities, 1, generator=generator))
    return int(indices[selected])


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    *,
    prompt_ids: Sequence[int],
    speech_embeddings: torch.Tensor | None,
    stop_ids: set[int],
    maximum: int,
    seed: int,
    temperature: float = 0.0,
    top_p: float = 0.8,
    repetition_penalty: float = 1.0,
    forced_prefix: Sequence[int] = (),
    stop_on_loop: bool = False,
) -> list[int]:
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    embeddings = model.get_input_embeddings()(ids)
    if speech_embeddings is not None:
        positions = [
            index
            for index, token in enumerate(prompt_ids)
            if c.GLM_SEMANTIC_OFFSET <= int(token) <= c.GLM_SEMANTIC_SPAN.last_id
        ]
        if len(positions) != len(speech_embeddings):
            raise ValueError(
                f"speech/prompt geometry differs: {len(speech_embeddings)} vs {len(positions)}"
            )
        embeddings.index_copy_(
            0,
            torch.tensor(positions, dtype=torch.long, device=device),
            speech_embeddings.to(embeddings.dtype),
        )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(inputs_embeds=embeddings.unsqueeze(0), use_cache=True)
    cache = output.past_key_values
    logits = output.logits[0, -1].float()
    generated: list[int] = []
    if forced_prefix:
        prefix = [int(value) for value in forced_prefix]
        generated.extend(prefix)
        prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=prefix_ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[0, -1].float()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for _ in range(int(maximum)):
        logical = logits[: len(tokenizer)].clone()
        apply_repetition_penalty(logical, generated, repetition_penalty)
        token = sample_token(
            logical,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(token)
        if token in stop_ids:
            break
        if stop_on_loop and token_tail_loops(generated):
            generated = generated[:-LOOP_NGRAM]
            break
        step = torch.tensor([[token]], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=step, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[0, -1].float()
    return generated


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


def needs_tts_flush(text: str, language: str, *, final: bool, long_audio: bool) -> bool:
    normalized = " ".join(text.split())
    if not normalized:
        return False
    if final or normalized[-1:] in {".", "?", "!", "。", "？", "！"}:
        return True
    if long_audio:
        if normalized[-1:] in {",", "，"}:
            return True
        if language == "cmn":
            return len(normalized.replace(" ", "")) >= 14
        return len(normalized.split()) >= 10
    if normalized[-1:] in {",", "，"}:
        return True
    if language == "cmn":
        return len(normalized.replace(" ", "")) >= 6
    return len(normalized.split()) >= 3


def timeline_audio(events: Sequence[tuple[int, np.ndarray]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    scheduled: list[dict[str, object]] = []
    cursor = 0
    for source_ms, audio in events:
        available = int(round(source_ms * SAMPLE_RATE / 1000.0))
        start = max(cursor, available)
        stop = start + len(audio)
        scheduled.append(
            {
                "source_available_ms": int(source_ms),
                "playback_start_ms": float(start * 1000 / SAMPLE_RATE),
                "playback_stop_ms": float(stop * 1000 / SAMPLE_RATE),
                "audio_samples": len(audio),
            }
        )
        cursor = stop
    result = np.zeros(cursor, dtype=np.float32)
    for row, (_, audio) in zip(scheduled, events):
        start = int(round(float(row["playback_start_ms"]) * SAMPLE_RATE / 1000.0))
        result[start : start + len(audio)] = audio
    return result, scheduled


def write_stereo(source: np.ndarray, translation_timeline: np.ndarray, path: Path) -> None:
    length = max(len(source), len(translation_timeline))
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[: len(source), 0] = source
    stereo[: len(translation_timeline), 1] = translation_timeline
    sf.write(path, stereo, SAMPLE_RATE, subtype="PCM_16")


def normalized_text(tokenizer, values: Sequence[int]) -> str:
    return " ".join(tokenizer.decode(list(values), skip_special_tokens=True).split())


def load_mono_16k(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = values.mean(axis=1).astype(np.float32, copy=False)
    if not len(mono) or not bool(np.isfinite(mono).all()):
        raise ValueError(f"invalid PCM: {path}")
    if rate == SAMPLE_RATE:
        return mono
    import torchaudio

    wav = torch.from_numpy(mono).unsqueeze(0)
    return torchaudio.functional.resample(wav, int(rate), SAMPLE_RATE).squeeze(0).numpy()


def extract_speaker_tokens(bicodec: BiCodecTokenizer, wav_path: Path) -> list[int]:
    global_tokens, _semantic = bicodec.tokenize(str(wav_path))
    values = [int(value) for value in global_tokens.detach().cpu().reshape(-1).tolist()]
    if len(values) < 32:
        raise ValueError(f"BiCodec speaker tokens shorter than 32: {wav_path}")
    return values[:32]


def speech_token_count(parts: Sequence[torch.Tensor]) -> int:
    return int(sum(int(part.shape[0]) for part in parts))


def evaluate_sample(row, *, decision_chunk_ms, model, tokenizer, objective, bicodec, output, seed, long_audio):
    sample_id = str(row["id"])
    source = np.asarray(row["_waveform"], dtype=np.float32)
    sample_root = output / sample_id
    segment_root = sample_root / "segments"
    sample_root.mkdir(parents=True)
    segment_root.mkdir()
    source_path = sample_root / "source.wav"
    sf.write(source_path, source, SAMPLE_RATE, subtype="PCM_16")

    frontend = make_cached_frontend(objective, next(model.parameters()).device)
    frontend_state = None
    speech_parts: list[torch.Tensor] = []
    asr_committer = ConservativeCommitter(holdback=1, recover=long_audio)
    mt_committer = ConservativeCommitter(holdback=2, recover=long_audio)
    asr_text = ""
    target_text = ""
    asr_all: list[str] = []
    mt_all: list[str] = []
    pending_target_ids: list[int] = []
    audio_events: list[tuple[int, np.ndarray]] = []
    semantic_all: list[int] = []
    event_rows: list[dict[str, object]] = []
    next_decision_ms = decision_chunk_ms
    event_index = 0
    utterance_index = 0
    silence_blocks = 0
    started = time.perf_counter()
    asr_max = STEP_NEW_TOKENS if long_audio else 128

    def decode_pending(text: str, source_end_ms: int, *, final: bool) -> tuple[list[int], Path | None, dict | None]:
        nonlocal pending_target_ids
        if not needs_tts_flush(text, row["tgt_lang"], final=final, long_audio=long_audio):
            return [], None, None
        tts_prompt = builders.build_tts_sample(
            bicodec_global=row["bicodec_global"],
            src_lang=row["tgt_lang"],
            transcription=text,
            source_bicodec=[0],
            text_encoder=lambda value: tokenizer.encode(value, add_special_tokens=False),
            source_id=sample_id,
        )
        tts_generated = generate(
            model,
            tokenizer,
            prompt_ids=tts_prompt.prompt_ids,
            speech_embeddings=None,
            stop_ids={c.TOKEN_END_SEMANTIC, c.TOKEN_EOS},
            maximum=320,
            seed=seed + 20_000 + event_index,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
        )
        semantic = semantic_content(tts_generated)
        segment_path = None
        segment_audit = None
        if semantic:
            decode_tokens = torch.tensor(
                [*row["bicodec_global"], *semantic],
                dtype=torch.long,
                device=next(model.parameters()).device,
            )
            waveform = np.asarray(
                bicodec.decode_tokens_to_audio(decode_tokens),
                dtype=np.float32,
            ).reshape(-1)
            segment_path = segment_root / f"event_{event_index:03d}_{source_end_ms}ms.wav"
            sf.write(segment_path, waveform, SAMPLE_RATE, subtype="PCM_16")
            segment_audit = audio_audit(segment_path)
            audio_events.append((source_end_ms, waveform))
            semantic_all.extend(semantic)
        pending_target_ids.clear()
        return semantic, segment_path, segment_audit

    def reset_utterance(*, keep_recent_speech: bool) -> None:
        nonlocal speech_parts, asr_text, target_text, pending_target_ids
        nonlocal utterance_index, silence_blocks
        if asr_text:
            asr_all.append(asr_text)
        if target_text:
            mt_all.append(target_text)
        asr_committer.reset()
        mt_committer.reset()
        asr_text = ""
        target_text = ""
        pending_target_ids = []
        utterance_index += 1
        silence_blocks = 0
        if keep_recent_speech and speech_parts:
            keep = max(1, len(speech_parts) // 3)
            speech_parts = speech_parts[-keep:]
        else:
            speech_parts = []

    for block_start in range(0, len(source), PHYSICAL_BLOCK_SAMPLES):
        block_stop = min(len(source), block_start + PHYSICAL_BLOCK_SAMPLES)
        final_block = block_stop == len(source)
        block = source[block_start:block_stop]
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if len(block) else 0.0
        if long_audio and rms < SILENCE_RMS:
            silence_blocks += 1
        else:
            silence_blocks = 0
        frontend_output = frontend.push(block, frontend_state, is_final=final_block)
        frontend_state = frontend_output.state
        hidden = frontend_output.pre_vq_hidden[0].to(
            device=next(objective.parameters()).device,
            dtype=objective.bridge_norm.weight.dtype,
        )
        codes = objective._nearest_codes(hidden)
        residual = objective.bridge_projection(objective.bridge_norm(hidden))
        base = model.get_input_embeddings()(codes.long() + c.GLM_SEMANTIC_OFFSET)
        speech_parts.append(base + residual.to(base.dtype))
        source_end_ms = int(frontend_output.source_end_ms)
        overflow = long_audio and speech_token_count(speech_parts) > MAX_SPEECH_TOKENS
        pause = (
            long_audio
            and not final_block
            and silence_blocks >= SILENCE_BLOCKS
            and bool(asr_committer.committed or mt_committer.committed)
        )
        decision = final_block or source_end_ms >= next_decision_ms or pause or overflow
        if not decision:
            continue
        while next_decision_ms <= source_end_ms:
            next_decision_ms += decision_chunk_ms
        speech = torch.cat(speech_parts, dim=0)
        asr_prompt = builders.build_asr_sample(
            source_glm=[0] * len(speech),
            bicodec_global=row["_stage_a_fixed_speaker_global"],
            src_lang=row["src_lang"],
            transcription="placeholder",
            text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
            source_id=sample_id,
        )
        asr_generated = generate(
            model,
            tokenizer,
            prompt_ids=asr_prompt.prompt_ids,
            speech_embeddings=speech,
            stop_ids={c.TOKEN_END_CONTENT, c.TOKEN_EOS},
            maximum=asr_max,
            seed=seed + event_index,
            forced_prefix=asr_committer.committed if long_audio else (),
        )
        asr_candidate = text_content(asr_generated)
        asr_new = asr_committer.update(asr_candidate, final=final_block or pause)
        if asr_new:
            asr_text = normalized_text(tokenizer, asr_committer.committed)
            if row["src_lang"] == "cmn":
                asr_text = asr_text.replace(" ", "")

        mt_candidate: list[int] = []
        mt_new: list[int] = []
        mt_budget = (
            mt_new_token_budget(asr_new, str(row["src_lang"]), final=final_block or pause)
            if long_audio
            else (160 if asr_text else 0)
        )
        if asr_text and mt_budget > 0:
            mt_prompt = builders.build_mt_sample(
                src_lang=row["src_lang"],
                tgt_lang=row["tgt_lang"],
                source_text=asr_text,
                target_text="placeholder",
                text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
                source_id=sample_id,
            )
            mt_generated = generate(
                model,
                tokenizer,
                prompt_ids=mt_prompt.prompt_ids,
                speech_embeddings=None,
                stop_ids={c.TOKEN_END_CONTENT, c.TOKEN_EOS},
                maximum=mt_budget,
                seed=seed + 10_000 + event_index,
                forced_prefix=mt_committer.committed if long_audio else (),
                repetition_penalty=1.2 if long_audio else 1.0,
                stop_on_loop=long_audio,
            )
            mt_candidate = cut_token_loop(text_content(mt_generated)) if long_audio else text_content(mt_generated)
            mt_new = mt_committer.update(mt_candidate, final=final_block or pause)
            if mt_new:
                pending_target_ids.extend(mt_new)
                target_text = normalized_text(tokenizer, mt_committer.committed)
                if row["tgt_lang"] == "cmn":
                    target_text = target_text.replace(" ", "")

        tts_text = normalized_text(tokenizer, pending_target_ids)
        if row["tgt_lang"] == "cmn":
            tts_text = tts_text.replace(" ", "")
        semantic, segment_path, segment_audit = decode_pending(
            tts_text, source_end_ms, final=final_block or pause
        )
        event_rows.append(
            {
                "event_index": event_index,
                "utterance_index": utterance_index,
                "source_end_ms": source_end_ms,
                "source_final": final_block,
                "long_audio": long_audio,
                "pause_reset": pause,
                "speech_overflow_reset": overflow,
                "mt_token_budget": mt_budget,
                "visible_source_glm_tokens": len(speech),
                "asr_candidate": normalized_text(tokenizer, asr_candidate),
                "asr_new_commit": normalized_text(tokenizer, asr_new),
                "asr_committed": asr_text,
                "mt_candidate": normalized_text(tokenizer, mt_candidate),
                "mt_new_commit": normalized_text(tokenizer, mt_new),
                "mt_committed": target_text,
                "tts_text": tts_text if semantic else "",
                "semantic_tokens": len(semantic),
                "semantic_max_identical_run": maximum_identical_run(semantic),
                "segment_audio_path": str(segment_path.resolve()) if segment_path else None,
                "segment_audio_audit": segment_audit,
            }
        )
        event_index += 1
        if pause:
            reset_utterance(keep_recent_speech=False)
        elif overflow:
            reset_utterance(keep_recent_speech=True)

    continuous = (
        np.concatenate([audio for _, audio in audio_events])
        if audio_events
        else np.zeros(0, dtype=np.float32)
    )
    timeline, schedule = timeline_audio(audio_events)
    continuous_path = sample_root / "translation_continuous.wav"
    timeline_path = sample_root / "translation_timeline.wav"
    stereo_path = sample_root / "stereo_left_source_right_translation.wav"
    sf.write(continuous_path, continuous, SAMPLE_RATE, subtype="PCM_16")
    sf.write(timeline_path, timeline, SAMPLE_RATE, subtype="PCM_16")
    write_stereo(source, timeline, stereo_path)
    reference = str(row.get("transcription") or "")
    if reference:
        metric, errors, units = stage_a_eval.error_counts(
            reference, asr_text, str(row["src_lang"])
        )
    else:
        metric, errors, units = ("cer" if row["src_lang"] == "cmn" else "wer"), 0, 0
    if asr_text:
        asr_all.append(asr_text)
    if target_text:
        mt_all.append(target_text)
    asr_text = " ".join(part for part in asr_all if part).strip()
    target_text = " ".join(part for part in mt_all if part).strip()
    if row["src_lang"] == "cmn":
        asr_text = asr_text.replace(" ", "")
    if row["tgt_lang"] == "cmn":
        target_text = target_text.replace(" ", "")
    source_duration_ms = int(round(len(source) * 1000 / SAMPLE_RATE))
    first_audio_source_ms = audio_events[0][0] if audio_events else None
    audio_result = audio_audit(continuous_path)
    return {
        "sample_id": sample_id,
        "src_lang": row["src_lang"],
        "tgt_lang": row["tgt_lang"],
        "decision_chunk_ms": decision_chunk_ms,
        "long_audio": long_audio,
        "physical_acoustic_block_ms": PHYSICAL_BLOCK_MS,
        "source_duration_ms": source_duration_ms,
        "source_audio": str(Path(row["source_audio"]).resolve()),
        "reference_transcription": reference,
        "generated_streaming_transcription": asr_text,
        "asr_metric": metric,
        "asr_errors": errors,
        "asr_reference_units": units,
        "asr_error_rate": errors / max(1, units) if units else None,
        "reference_translation": row.get("translation") or "",
        "generated_streaming_translation": target_text,
        "asr_revision_conflicts": asr_committer.revision_conflicts,
        "mt_revision_conflicts": mt_committer.revision_conflicts,
        "events": event_rows,
        "audio_writes": len(audio_events),
        "semantic_tokens": len(semantic_all),
        "first_audio_source_ms": first_audio_source_ms,
        "prefinal_audio_emitted": bool(
            first_audio_source_ms is not None and first_audio_source_ms < source_duration_ms
        ),
        "processing_seconds": time.perf_counter() - started,
        "continuous_audio_path": str(continuous_path.resolve()),
        "timeline_audio_path": str(timeline_path.resolve()),
        "stereo_audio_path": str(stereo_path.resolve()),
        "audio_audit": audio_result,
        "playback_schedule": schedule,
        "pilot15_sample_dir": row.get("pilot15_sample_dir"),
        "model_scope_warning": (
            "Stage A trained streaming/causal ASR, not incremental MT/TTS. "
            "Long-audio mode prefix-forces committed text and resets on pause/overflow; "
            "it is still a runtime cascade, not a jointly trained E2E student."
        ),
    }


def load_manifest_rows(path: Path, sample_ids: Sequence[str] | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    requested = list(dict.fromkeys(sample_ids)) if sample_ids else None
    selected: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if requested is None:
                rows.append(row)
                continue
            if sample_id in requested:
                selected[sample_id] = row
    if requested is None:
        return rows
    missing = [value for value in requested if value not in selected]
    if missing:
        raise ValueError(f"requested samples are absent from manifest: {missing}")
    return [selected[value] for value in requested]


def write_compare_markdown(output: Path, results: Sequence[dict[str, object]]) -> Path:
    lines = [
        "# full198 vs 15-shard cascade S2ST listen sheet",
        "",
        "| sample | chunk | full198 | 15-shard | ASR | MT | first audio ms | writes |",
        "|---|---:|---|---|---|---|---:|---:|",
    ]
    for row in results:
        chunk = int(row["decision_chunk_ms"])
        old_dir = row.get("pilot15_sample_dir")
        old = ""
        if old_dir:
            old_path = (
                PILOT15_DEMO_ROOT
                / f"chunk_{chunk}ms_v1"
                / str(old_dir)
                / "stereo_left_source_right_translation.wav"
            )
            old = str(old_path) if old_path.is_file() else f"(missing) {old_path}"
        lines.append(
            "| {sample} | {chunk} | `{new}` | `{old}` | {asr} | {mt} | {first} | {writes} |".format(
                sample=row["sample_id"],
                chunk=chunk,
                new=row["stereo_audio_path"],
                old=old,
                asr=str(row["generated_streaming_transcription"]).replace("|", "/"),
                mt=str(row["generated_streaming_translation"]).replace("|", "/"),
                first=row["first_audio_source_ms"] if row["first_audio_source_ms"] is not None else "none",
                writes=row["audio_writes"],
            )
        )
    path = output / "COMPARE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--whispervq", type=Path, required=True)
    parser.add_argument("--bicodec", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=[160, 320, 640, 1280])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--src-lang", choices=sorted(ALLOWED_LANGS))
    parser.add_argument("--tgt-lang", choices=sorted(ALLOWED_LANGS))
    parser.add_argument("--sample-name", default="")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--long-audio", action="store_true")
    parser.add_argument("--force-short", action="store_true")
    parser.add_argument("--seed-base", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value not in ALLOWED_CHUNKS for value in args.chunk_ms):
        raise ValueError(f"chunk sizes must be in {ALLOWED_CHUNKS}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if bool(args.audio) == bool(args.manifest):
        raise ValueError("provide exactly one of --manifest or --audio")
    if args.audio and (not args.src_lang or not args.tgt_lang or args.src_lang == args.tgt_lang):
        raise ValueError("custom audio needs --src-lang and --tgt-lang, and they must differ")
    if args.long_audio and args.force_short:
        raise ValueError("choose at most one of --long-audio and --force-short")
    args.output.mkdir(parents=True)

    snapshot = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    fixed_speaker = [int(value) for value in snapshot["fixed_system_speaker"]["global_tokens"]]
    if len(fixed_speaker) != 32:
        raise ValueError("Stage A fixed speaker must contain 32 global tokens")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    objective = (
        stage_a_eval.load_objective(args.checkpoint, args.whispervq, device)
        .eval()
        .requires_grad_(False)
    )
    bicodec = BiCodecTokenizer(model_dir=args.bicodec, device=device)
    bicodec.model.eval()

    if args.audio:
        audio_path = args.audio.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        waveform = load_mono_16k(audio_path)
        prepared = args.output / "_prepared"
        prepared.mkdir()
        prepared_wav = prepared / "input_16k_mono.wav"
        sf.write(prepared_wav, waveform, SAMPLE_RATE, subtype="PCM_16")
        sample_id = args.sample_name or audio_path.stem
        rows = [
            {
                "id": sample_id,
                "src_lang": args.src_lang,
                "tgt_lang": args.tgt_lang,
                "transcription": "",
                "translation": "",
                "source_audio": str(prepared_wav),
                "bicodec_global": extract_speaker_tokens(bicodec, prepared_wav),
                "_waveform": waveform,
                "_stage_a_fixed_speaker_global": fixed_speaker,
            }
        ]
    else:
        rows = load_manifest_rows(args.manifest, args.sample_id)
        for row in rows:
            wav_path = Path(str(row["source_audio"]))
            if not wav_path.is_file():
                raise FileNotFoundError(f"manifest audio missing: {wav_path}")
            row["_waveform"] = load_mono_16k(wav_path)
            row["_stage_a_fixed_speaker_global"] = fixed_speaker
            row["bicodec_global"] = [int(value) for value in row["bicodec_global"]]

    results: list[dict[str, object]] = []
    for chunk_ms in args.chunk_ms:
        chunk_root = args.output / f"chunk_{chunk_ms}ms"
        chunk_root.mkdir()
        for index, row in enumerate(rows):
            duration = len(row["_waveform"]) / SAMPLE_RATE
            if args.force_short:
                long_audio = False
            elif args.long_audio:
                long_audio = True
            else:
                long_audio = duration > LONG_AUDIO_SECONDS
            print(
                f"chunk={chunk_ms} sample={row['id']} long_audio={long_audio} duration={duration:.1f}s",
                flush=True,
            )
            result = evaluate_sample(
                row,
                decision_chunk_ms=int(chunk_ms),
                model=model,
                tokenizer=tokenizer,
                objective=objective,
                bicodec=bicodec,
                output=chunk_root,
                seed=args.seed_base + int(chunk_ms) * 100 + index * 1_000_000,
                long_audio=long_audio,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "sample_id",
                            "decision_chunk_ms",
                            "long_audio",
                            "generated_streaming_transcription",
                            "generated_streaming_translation",
                            "audio_writes",
                            "first_audio_source_ms",
                            "asr_revision_conflicts",
                            "mt_revision_conflicts",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "hf_model": str(args.hf_model.resolve()),
        "chunk_ms": [int(value) for value in args.chunk_ms],
        "physical_acoustic_block_ms": PHYSICAL_BLOCK_MS,
        "results": results,
    }
    (args.output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.compare:
        compare = write_compare_markdown(args.output, results)
        print(f"COMPARE={compare.resolve()}", flush=True)
    print(f"OUTPUT={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
