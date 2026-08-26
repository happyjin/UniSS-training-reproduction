#!/usr/bin/env python3
"""Reference/offline ASR, MT and TTS routes for long-episode attribution."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sacrebleu
import soundfile as sf
import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from experiments.uniss_phasea_stateful_longepisode_rl_v1.runtime.stateful_cascade import (
    PHYSICAL_BLOCK_SAMPLES,
    SAMPLE_RATE,
    generate_semantic_with_continuation,
    normalized_text,
    semantic_content,
    split_ready_prefixes,
    waveform_health,
    write_stereo,
)
from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.strict_cascade import (
    _load_models,
)
from training import constants_uniss as c
from training import sample_builders as builders


def load_generate(path: Path):
    spec = importlib.util.spec_from_file_location("uniss_attribution_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate


def text_content(tokens: list[int]) -> list[int]:
    stop = tokens.index(c.TOKEN_END_CONTENT) if c.TOKEN_END_CONTENT in tokens else len(tokens)
    return [int(value) for value in tokens[:stop] if int(value) <= c.QWEN_BASE_VOCAB_END]


def full_speech_embeddings(source: np.ndarray, objective, model) -> torch.Tensor:
    frontend = make_cached_frontend(objective, next(model.parameters()).device)
    state = None
    values: list[torch.Tensor] = []
    for start in range(0, len(source), PHYSICAL_BLOCK_SAMPLES):
        stop = min(len(source), start + PHYSICAL_BLOCK_SAMPLES)
        output = frontend.push(source[start:stop], state, is_final=stop == len(source))
        state = output.state
        hidden = output.pre_vq_hidden[0].to(
            device=next(objective.parameters()).device,
            dtype=objective.bridge_norm.weight.dtype,
        )
        codes = objective._nearest_codes(hidden)
        residual = objective.bridge_projection(objective.bridge_norm(hidden))
        base = model.get_input_embeddings()(codes.long() + c.GLM_SEMANTIC_OFFSET)
        values.append(base + residual.to(base.dtype))
    if not values:
        raise ValueError("source produced no acoustic embeddings")
    return torch.cat(values, dim=0)


def asr_similarity(reference: str, hypothesis: str, language: str) -> tuple[float, int, int]:
    _, errors, units = stage_a_eval.error_counts(reference, hypothesis, language)
    return max(0.0, 1.0 - errors / max(1, units)), int(errors), int(units)


@torch.inference_mode()
def evaluate_row(row, *, model, tokenizer, objective, codec, generate_fn, output: Path, seed: int):
    episode_id = str(row["episode_id"])
    root = output / episode_id
    root.mkdir(parents=True)
    source, rate = sf.read(row["source_audio"], dtype="float32", always_2d=True)
    if int(rate) != SAMPLE_RATE:
        raise ValueError(f"source is not 16 kHz: {row['source_audio']}")
    source = np.asarray(source.mean(axis=1), dtype=np.float32)
    speech = full_speech_embeddings(source, objective, model)

    asr_prompt = builders.build_asr_sample(
        source_glm=[0] * len(speech),
        bicodec_global=row["speaker_global"],
        src_lang=row["src_lang"],
        transcription="placeholder",
        text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
        source_id=f"{episode_id}:offline_asr",
    )
    asr_generated = generate_fn(
        model,
        tokenizer,
        prompt_ids=asr_prompt.prompt_ids,
        speech_embeddings=speech,
        stop_ids={c.TOKEN_END_CONTENT, c.TOKEN_EOS},
        maximum=512,
        seed=seed,
    )
    offline_asr = normalized_text(
        tokenizer, text_content(asr_generated), str(row["src_lang"])
    )
    asr_score, asr_errors, asr_units = asr_similarity(
        str(row["teacher_transcription"]), offline_asr, str(row["src_lang"])
    )

    mt_prompt = builders.build_mt_sample(
        src_lang=row["src_lang"],
        tgt_lang=row["tgt_lang"],
        source_text=str(row["teacher_transcription"]),
        target_text="placeholder",
        text_encoder=lambda text: tokenizer.encode(text, add_special_tokens=False),
        source_id=f"{episode_id}:gold_source_mt",
    )
    mt_generated = generate_fn(
        model,
        tokenizer,
        prompt_ids=mt_prompt.prompt_ids,
        speech_embeddings=None,
        stop_ids={c.TOKEN_END_CONTENT, c.TOKEN_EOS},
        maximum=512,
        seed=seed + 10_000,
    )
    gold_source_mt = normalized_text(
        tokenizer, text_content(mt_generated), str(row["tgt_lang"])
    )
    mt_chrf = float(
        sacrebleu.corpus_chrf(
            [gold_source_mt], [[str(row["teacher_translation"])]]
        ).score
    )

    target_ids = tokenizer.encode(
        str(row["teacher_translation"]), add_special_tokens=False
    )
    phrases, remaining = split_ready_prefixes(
        target_ids, tokenizer, str(row["tgt_lang"]), final=True
    )
    if remaining:
        raise RuntimeError("gold target TTS splitter left an unspoken suffix")
    waveforms: list[np.ndarray] = []
    phrase_rows: list[dict[str, object]] = []
    for index, (ids, text) in enumerate(phrases):
        del ids
        tts_prompt = builders.build_tts_sample(
            bicodec_global=row["speaker_global"],
            src_lang=row["tgt_lang"],
            transcription=text,
            source_bicodec=[0],
            text_encoder=lambda value: tokenizer.encode(value, add_special_tokens=False),
            source_id=f"{episode_id}:gold_tts:{index}",
        )
        semantic, ended, continuations = generate_semantic_with_continuation(
            generate_fn=generate_fn,
            model=model,
            tokenizer=tokenizer,
            prompt_ids=tts_prompt.prompt_ids,
            seed=seed + 20_000 + index * 100,
            maximum_per_pass=320,
            maximum_passes=4,
        )
        waveform = np.zeros(0, dtype=np.float32)
        if semantic:
            tokens = torch.tensor(
                [*row["speaker_global"], *semantic],
                dtype=torch.long,
                device=next(model.parameters()).device,
            )
            waveform = np.asarray(codec.decode_tokens_to_audio(tokens), dtype=np.float32).reshape(-1)
        health = waveform_health(waveform)
        if bool(health["healthy"]):
            waveforms.append(waveform)
        phrase_rows.append(
            {
                "index": index,
                "text": text,
                "semantic_tokens": len(semantic),
                "semantic_ended": ended,
                "semantic_continuations": continuations,
                "audio_health": health,
            }
        )
    oracle_audio = np.concatenate(waveforms) if waveforms else np.zeros(0, dtype=np.float32)
    source_path = root / "source.wav"
    oracle_path = root / "gold_target_phasea_tts.wav"
    stereo_path = root / "stereo_left_source_right_gold_target_tts.wav"
    sf.write(source_path, source, SAMPLE_RATE, subtype="PCM_16")
    sf.write(oracle_path, oracle_audio, SAMPLE_RATE, subtype="PCM_16")
    write_stereo(source, oracle_audio, stereo_path)
    return {
        "episode_id": episode_id,
        "direction": row["direction"],
        "source_audio": str(source_path.resolve()),
        "teacher_transcription": row["teacher_transcription"],
        "offline_full_context_asr": offline_asr,
        "offline_asr_similarity": asr_score,
        "offline_asr_errors": asr_errors,
        "offline_asr_reference_units": asr_units,
        "teacher_translation": row["teacher_translation"],
        "gold_source_offline_mt": gold_source_mt,
        "gold_source_mt_chrf": mt_chrf,
        "gold_target_tts_audio": str(oracle_path.resolve()),
        "gold_target_tts_stereo": str(stereo_path.resolve()),
        "gold_target_tts_health": waveform_health(oracle_audio),
        "gold_target_tts_phrases": phrase_rows,
        "gold_target_tts_phrase_success_fraction": len(waveforms) / max(1, len(phrases)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--maximum-episodes", type=int)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--bicodec-model", type=Path, required=True)
    parser.add_argument("--strict-runtime", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.episodes.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for index, row in enumerate(rows) if index % args.num_workers == args.worker_index]
    if args.maximum_episodes is not None:
        rows = rows[: args.maximum_episodes]
    if not rows:
        raise ValueError("empty attribution worker selection")
    load_args = SimpleNamespace(
        device=args.device,
        adapter_checkpoint=None,
        base_hf=args.base_hf,
        v1_checkpoint=args.v1_checkpoint,
        whispervq_model=args.whispervq_model,
        bicodec_model=args.bicodec_model,
    )
    model, tokenizer, controller, manifest, objective, codec = _load_models(load_args)
    generate_fn = load_generate(args.strict_runtime)
    args.output.mkdir(parents=True)
    results = []
    try:
        for index, row in enumerate(rows):
            value = evaluate_row(
                row,
                model=model,
                tokenizer=tokenizer,
                objective=objective,
                codec=codec,
                generate_fn=generate_fn,
                output=args.output,
                seed=20260826 + args.worker_index * 1_000_000 + index * 100_000,
            )
            results.append(value)
            print(
                json.dumps(
                    {
                        "episode_id": value["episode_id"],
                        "offline_asr_similarity": value["offline_asr_similarity"],
                        "gold_source_mt_chrf": value["gold_source_mt_chrf"],
                        "gold_target_tts_phrase_success_fraction": value[
                            "gold_target_tts_phrase_success_fraction"
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        controller.close()
    payload = {
        "schema_version": "uniss_phasea_reference_attribution_worker_v1",
        "status": "complete",
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "model_manifest": manifest,
        "results": results,
    }
    (args.output / "ATTRIBUTION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
