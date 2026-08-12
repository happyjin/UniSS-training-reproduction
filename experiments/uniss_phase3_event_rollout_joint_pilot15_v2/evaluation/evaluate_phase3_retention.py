#!/usr/bin/env python3
"""Evaluate offline Phase3 behavior before and after the streaming adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from evaluation.uniss_outputs import parse_with_tokenizer
from training import constants_uniss as c
from training.sample_builders import build_performance_sample
from training.simul_uniss.jsonl_index import load_index
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from web_demo.true_subsecond_pilot15_streaming_v1.model_loader import (
    inject_exact_runtime_lora,
)


SCHEMA = "uniss_event_rollout_fixed15_phase3_retention_v1"
SYSTEMS = ("phase3_v4", "streaming_adapter")


def _safe_name(index: int, sample_id: str, system: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:96]
    return f"{index:05d}_{system}_{value}.wav"


def _row(handle, offset: int) -> dict[str, object]:
    handle.seek(int(offset))
    return json.loads(handle.readline())


def _audio_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames / info.samplerate)


def _truncate_at_eos(values: Sequence[int]) -> list[int]:
    output = []
    for value in values:
        output.append(int(value))
        if int(value) == c.TOKEN_EOS:
            break
    return output


def _load_model(
    export: Path,
    *,
    device: torch.device,
    with_adapter: bool,
) -> tuple[object, object, dict[str, object]]:
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    base = Path(str(manifest["base_model"])).resolve()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    config.rms_norm_eps = float(manifest.get("layernorm_epsilon", 1.0e-5))
    kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "config": config,
    }
    if device.type == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if with_adapter:
        inject_exact_runtime_lora(
            model,
            rank=int(manifest["rank"]),
            alpha=float(manifest["alpha"]),
        )
        adapter = load_file(export / "adapter_model.safetensors")
        _, unexpected = model.load_state_dict(adapter, strict=False)
        missing_adapter = [
            name
            for name, _ in model.named_parameters()
            if (".lora_A." in name or ".lora_B." in name) and name not in adapter
        ]
        unexpected_adapter = [name for name in unexpected if name in adapter]
        if missing_adapter or unexpected_adapter:
            raise ValueError(
                "offline retention adapter mismatch: "
                f"missing={missing_adapter[:8]}, unexpected={unexpected_adapter[:8]}"
            )
    model.to(device=device, dtype=kwargs["torch_dtype"]).eval().requires_grad_(False)
    return model, tokenizer, manifest


def _decode_audio(
    codec: BiCodecTokenizer,
    global_values: Sequence[int],
    semantic_values: Sequence[int],
    output: Path,
    device: torch.device,
) -> tuple[str | None, str | None, dict[str, object]]:
    if not semantic_values:
        return None, "no_semantic_tokens", {
            "audio_finite": False,
            "audio_rms": 0.0,
            "audio_non_silent_fraction": 0.0,
        }
    try:
        global_tensor = torch.tensor(global_values, dtype=torch.long, device=device)
        semantic_tensor = torch.tensor(semantic_values, dtype=torch.long, device=device)
        audio = codec.detokenize(global_tensor.unsqueeze(0), semantic_tensor.unsqueeze(0))
        value = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(output, value, 16000, subtype="PCM_16")
    except Exception as exc:
        return None, f"decode_error:{type(exc).__name__}:{exc}", {
            "audio_finite": False,
            "audio_rms": 0.0,
            "audio_non_silent_fraction": 0.0,
        }
    finite = bool(np.isfinite(value).all())
    rms = float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))) if len(value) else 0.0
    non_silent = float(np.mean(np.abs(value) >= 1.0e-4)) if len(value) else 0.0
    return str(output.resolve()), None, {
        "audio_finite": finite,
        "audio_rms": rms,
        "audio_non_silent_fraction": non_silent,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(args.formal_manifest).resolve()
    output = Path(args.output).resolve()
    export = Path(args.export).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retention evaluation: {output}")
    offsets = load_index(manifest_path)
    if offsets is None:
        raise ValueError("retention manifest lacks its uint64 index")
    output.mkdir(parents=True)
    wav_root = output / "wav"
    wav_root.mkdir()
    rows = []
    with manifest_path.open("rb") as handle:
        for index in range(min(args.samples, len(offsets))):
            rows.append(_row(handle, offsets[index]))
    device = torch.device(args.device)
    codec = BiCodecTokenizer(
        model_dir=Path(args.speech_tokenizer).resolve() / "bicodec",
        device=device,
    )
    results: list[dict[str, object]] = []
    for system, with_adapter in ((SYSTEMS[0], False), (SYSTEMS[1], True)):
        model, tokenizer, export_manifest = _load_model(
            export, device=device, with_adapter=with_adapter
        )
        suppressed = list(range(c.VOCAB_SIZE, int(model.config.vocab_size)))
        encoder = lambda text: tokenizer.encode(text, add_special_tokens=False)
        for index, row in enumerate(rows):
            sample = build_performance_sample(
                source_glm=row["source_glm"],
                bicodec_global=row["bicodec_global"],
                tgt_lang=str(row["tgt_lang"]),
                translation=str(row["translation"]),
                target_bicodec=row["target_bicodec"],
                text_encoder=encoder,
                source_id=str(row["id"]),
            )
            prompt = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    prompt,
                    max_new_tokens=args.maximum_new_tokens,
                    do_sample=False,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=c.TOKEN_PAD,
                    eos_token_id=c.TOKEN_EOS,
                    suppress_tokens=suppressed,
                )
            generation_seconds = time.perf_counter() - started
            tail = _truncate_at_eos(generated[0, prompt.shape[1] :].tolist())
            parsed = parse_with_tokenizer(tail, mode="performance", tokenizer=tokenizer)
            semantic = [int(value) for value in parsed["semantic_values"]]
            wav = wav_root / _safe_name(index, str(row["id"]), system)
            audio_path, error, audio_audit = _decode_audio(
                codec,
                row["bicodec_global"],
                semantic,
                wav,
                device,
            )
            result = {
                "schema_version": SCHEMA,
                "id": str(row["id"]),
                "mode": system,
                "src_lang": str(row["src_lang"]),
                "tgt_lang": str(row["tgt_lang"]),
                "transcription_ref": str(row.get("transcription", "")),
                "translation_ref": str(row["translation"]),
                "generated_translation": parsed["generated_translation"],
                "generated_token_ids": tail,
                "semantic_token_count": len(semantic),
                "has_semantic_start": bool(parsed["has_semantic_start"]),
                "has_semantic_end": bool(parsed["has_semantic_end"]),
                "has_eos": bool(parsed["has_eos"]),
                "generation_seconds": generation_seconds,
                "audio_path": audio_path,
                "audio_duration_seconds": (
                    None if audio_path is None else _audio_duration(Path(audio_path))
                ),
                "source_audio_path": str(Path(str(row["source_audio"])).resolve()),
                "source_audio_duration_seconds": float(row["source_duration_ms"]) / 1000.0,
                "reference_audio_path": str(Path(str(row["target_audio"])).resolve()),
                "error": error,
                "streaming_adapter_loaded": with_adapter,
                "checkpoint": (
                    export_manifest["source_checkpoint"] if with_adapter else str(export_manifest["base_model"])
                ),
                **audio_audit,
            }
            results.append(result)
            with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "schema_version": SCHEMA,
        "formal_manifest": str(manifest_path),
        "runtime_export": str(export),
        "systems": list(SYSTEMS),
        "samples_per_system": len(rows),
        "result_rows": len(results),
        "protocol": {
            "task": "phase3_performance_offline_s2st",
            "decoding": "deterministic_greedy",
            "maximum_new_tokens": args.maximum_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "paired_sample_ids": True,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--speech-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2**31 - 1)
    parser.add_argument("--maximum-new-tokens", type=int, default=1500)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2))
