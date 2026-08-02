"""Measure frozen Phase3 sensitivity to alternative source GLM token streams."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from evaluation.text_metrics import compute_grouped_bleu
from evaluation.uniss_outputs import parse_with_tokenizer
from training import constants_uniss as c
from training.generate_unist_eval_audio import load_hf_text_encoder
from training.sample_builders import (
    TrainingSample,
    build_direct_s2st_sample,
    build_performance_sample,
)
from training.simul_uniss.subsecond_v2.audit_teacher_prefix_ceiling import (
    _sample_records,
    build_immediate_causal_stream,
)
from training.simul_uniss.subsecond_v2.validate_stage_b_latent import load_model
from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
    StreamingWhisperVQTeacher,
)
from uniss.speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer


SCHEMA = "simul_uniss_phase3_token_stream_sensitivity_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prefix_streams(
    teacher: Glm4Tokenizer,
    waveform: torch.Tensor,
    token_end_ms: Sequence[int],
    *,
    chunk_ms: int,
    lookaheads: Sequence[int],
) -> dict[str, list[int]]:
    duration_ms = int(round(waveform.shape[-1] / 16))
    commit_ends = list(range(chunk_ms, duration_ms + chunk_ms, chunk_ms))
    commit_ends = list(dict.fromkeys(min(value, duration_ms) for value in commit_ends))
    requests: list[tuple[int, int]] = []
    audio: list[tuple[torch.Tensor, int]] = []
    for lookahead in lookaheads:
        for committed_ms in commit_ends:
            visible_ms = min(duration_ms, committed_ms + lookahead)
            samples = max(400, min(waveform.shape[-1], visible_ms * 16))
            requests.append((lookahead, committed_ms))
            audio.append((waveform[..., :samples], 16_000))
    outputs = teacher.bacth_tokenize(audio)
    grouped: dict[int, list[list[int]]] = {value: [] for value in lookaheads}
    for (lookahead, _), tokens in zip(requests, outputs):
        grouped[lookahead].append([int(value) for value in tokens])
    return {
        f"prefix_causal_{lookahead}ms": build_immediate_causal_stream(
            token_end_ms, commit_ends, grouped[lookahead]
        )
        for lookahead in lookaheads
    }


def _build_sample(
    record: Mapping[str, object],
    source_glm: Sequence[int],
    task: str,
    text_encoder,
) -> TrainingSample:
    common = {
        "source_glm": source_glm,
        "bicodec_global": record["bicodec_global"],
        "tgt_lang": str(record["tgt_lang"]),
        "target_bicodec": record["target_bicodec"],
        "source_id": str(record.get("id", "")) or None,
    }
    if task == "performance":
        return build_performance_sample(
            **common,  # type: ignore[arg-type]
            translation=str(record["translation"]),
            text_encoder=text_encoder,
        )
    if task == "direct_s2st":
        return build_direct_s2st_sample(**common)  # type: ignore[arg-type]
    raise ValueError(task)


@torch.inference_mode()
def _teacher_forced_score(model, sample: TrainingSample, device: torch.device) -> dict[str, object]:
    input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids).logits[0, :-1].float()
    next_ids = input_ids[0, 1:]
    losses = F.cross_entropy(logits, next_ids, reduction="none")
    start = sample.prompt_length - 1
    target_losses = losses[start : start + sample.target_length]
    target_logits = logits[start : start + sample.target_length]
    result: dict[str, object] = {
        "target_nll_sum": float(target_losses.sum()),
        "target_tokens": sample.target_length,
        "target_exact": int((target_logits.argmax(dim=-1) == input_ids[0, sample.prompt_length :]).sum()),
    }
    for name, (left, right) in sample.segment_spans.items():
        segment = target_losses[left:right]
        result[f"segment:{name}:nll_sum"] = float(segment.sum())
        result[f"segment:{name}:tokens"] = right - left
    return result


def _aggregate_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total_nll = sum(float(row["target_nll_sum"]) for row in rows)
    total_tokens = sum(int(row["target_tokens"]) for row in rows)
    exact = sum(int(row["target_exact"]) for row in rows)
    segment_sums: dict[str, float] = defaultdict(float)
    segment_tokens: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if key.startswith("segment:") and key.endswith(":nll_sum"):
                name = key[len("segment:") : -len(":nll_sum")]
                segment_sums[name] += float(value)
            elif key.startswith("segment:") and key.endswith(":tokens"):
                name = key[len("segment:") : -len(":tokens")]
                segment_tokens[name] += int(value)
    return {
        "samples": len(rows),
        "target_tokens": total_tokens,
        "mean_target_nll": total_nll / max(1, total_tokens),
        "target_perplexity": math.exp(min(20.0, total_nll / max(1, total_tokens))),
        "target_token_accuracy": exact / max(1, total_tokens),
        "segments": {
            name: {
                "tokens": segment_tokens[name],
                "mean_nll": segment_sums[name] / max(1, segment_tokens[name]),
            }
            for name in sorted(segment_sums)
        },
    }


@torch.inference_mode()
def _generate_performance_translation(
    model,
    tokenizer,
    sample: TrainingSample,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prompt = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
    suppressed = list(range(c.VOCAB_SIZE, int(model.config.vocab_size)))
    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=c.TOKEN_PAD,
        eos_token_id=c.TOKEN_EOS,
        suppress_tokens=suppressed or None,
    )
    tail = generated[0, prompt.shape[1] :].tolist()
    parsed = parse_with_tokenizer(tail, mode="performance", tokenizer=tokenizer)
    return str(parsed.get("generated_translation") or "")


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    manifest = Path(args.manifest).resolve()
    records = _sample_records(manifest, args.samples, args.audio_workers)
    lookaheads = sorted(set(args.lookahead_ms))
    teacher = Glm4Tokenizer(args.whispervq_model, device=args.device)
    for index, record in enumerate(records, start=1):
        waveform = record["_waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform is not a tensor")
        waveform = waveform[..., : args.max_audio_seconds * 16_000]
        duration_ms = int(round(waveform.shape[-1] / 16))
        ends = [int(value) for value in record[args.reference_end_field]]  # type: ignore[index]
        count = bisect.bisect_right(ends, duration_ms)
        record["_streams"] = {
            "released": [int(value) for value in record["released_source_glm"]][:count],  # type: ignore[index]
            "reconstructed_full": [int(value) for value in record[args.reference_field]][:count],  # type: ignore[index]
            **_prefix_streams(
                teacher,
                waveform,
                ends[:count],
                chunk_ms=args.chunk_ms,
                lookaheads=lookaheads,
            ),
        }
        print(json.dumps({"teacher_processed": index, "samples": len(records)}), flush=True)
    del teacher
    torch.cuda.empty_cache()

    streaming_teacher = StreamingWhisperVQTeacher(
        args.whispervq_model,
        device=args.device,
        chunk_ms=args.streaming_clone_chunk_ms,
        right_context_ms=args.streaming_clone_right_context_ms,
    )
    clone_outputs = streaming_teacher.encode(
        [record["_waveform"] for record in records]  # type: ignore[list-item]
    )
    for record, output in zip(records, clone_outputs):
        record["_streams"][
            f"streaming_clone_{args.streaming_clone_chunk_ms}x"
            f"{args.streaming_clone_right_context_ms}ms"
        ] = output.tokens.tolist()  # type: ignore[index]
    del streaming_teacher
    torch.cuda.empty_cache()

    device = torch.device(args.device)
    student, _ = load_model(args.student_checkpoint, device, None, None)
    for record in records:
        waveform = record["_waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform is not a tensor")
        waveform = waveform[..., : args.max_audio_seconds * 16_000].to(device)
        output = student.infer_waveform(waveform)
        length = int(output["token_lengths"][0])
        tokens = student.quantize(output["glm_latent"][:, :length]).reshape(-1).tolist()
        record["_streams"]["student_v1"] = [int(value) for value in tokens]  # type: ignore[index]
    del student
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.phase3_model, local_files_only=True)
    text_encoder = load_hf_text_encoder(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.phase3_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    scores: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    generation_rows: list[dict[str, object]] = []
    for record_index, record in enumerate(records, start=1):
        streams = record["_streams"]
        if not isinstance(streams, Mapping):
            raise TypeError("streams are missing")
        for stream_name, source_glm in streams.items():
            if not isinstance(source_glm, Sequence):
                raise TypeError(stream_name)
            for task in ("performance", "direct_s2st"):
                sample = _build_sample(record, source_glm, task, text_encoder)
                scores[str(stream_name)][task].append(
                    _teacher_forced_score(model, sample, device)
                )
                if task == "performance":
                    hypothesis = _generate_performance_translation(
                        model,
                        tokenizer,
                        sample,
                        device,
                        args.max_new_tokens,
                    )
                    generation_rows.append(
                        {
                            "id": record.get("id"),
                            "mode": str(stream_name),
                            "src_lang": record.get("src_lang"),
                            "tgt_lang": record.get("tgt_lang"),
                            "translation_ref": record.get("translation"),
                            "generated_translation": hypothesis,
                        }
                    )
        print(json.dumps({"phase3_processed": record_index, "samples": len(records)}), flush=True)
    aggregates = {
        stream: {task: _aggregate_scores(rows) for task, rows in tasks.items()}
        for stream, tasks in scores.items()
    }
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "manifest": str(manifest),
        "phase3_model": str(Path(args.phase3_model).resolve()),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "samples": len(records),
        "lookahead_ms": lookaheads,
        "streaming_clone_chunk_ms": args.streaming_clone_chunk_ms,
        "streaming_clone_right_context_ms": args.streaming_clone_right_context_ms,
        "teacher_forced": aggregates,
        "text_bleu": compute_grouped_bleu(
            generation_rows,
            hypothesis_field="generated_translation",
            reference_field="translation_ref",
            score_empty_hypotheses=True,
        ),
        "generations": generation_rows,
    }
    _atomic_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--phase3-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--audio-workers", type=int, default=8)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, nargs="+", default=[80, 160, 320, 640])
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--streaming-clone-chunk-ms", type=int, default=160)
    parser.add_argument("--streaming-clone-right-context-ms", type=int, default=80)
    parser.add_argument("--reference-field", default="teacher_source_glm")
    parser.add_argument("--reference-end-field", default="teacher_source_glm_end_ms")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
