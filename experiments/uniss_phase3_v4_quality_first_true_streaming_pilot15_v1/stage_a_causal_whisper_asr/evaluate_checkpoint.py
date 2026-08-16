#!/usr/bin/env python3
"""Read-only Stage A diagnosis on fixed real-PCM validation samples."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import soundfile as sf
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.frontend import (
    TrainableSharedCausalWhisperVQ,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.objective import (
    StageAObjective,
    terminal_codec_extension_deficit_samples,
)
from training import constants_uniss as c


SAMPLE_RATE = 16_000
TASKS = ("streaming_asr", "causal_full_asr")


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Stage A diagnosis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
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


def generated_runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate((*flags, False)):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            result.append((start, index))
            start = None
    return result


def collapse_ctc(ids: Sequence[int], blank_id: int = 256) -> list[int]:
    result: list[int] = []
    previous: int | None = None
    for raw in ids:
        value = int(raw)
        if value != previous and value != blank_id:
            result.append(value)
        previous = value
    return result


def content_ids(token_ids: Sequence[int]) -> list[int]:
    result: list[int] = []
    cursor = 0
    values = [int(value) for value in token_ids]
    while cursor < len(values):
        try:
            start = values.index(c.TOKEN_START_CONTENT, cursor) + 1
        except ValueError:
            break
        try:
            end = values.index(c.TOKEN_END_CONTENT, start)
        except ValueError:
            break
        result.extend(values[start:end])
        cursor = end + 1
    return result


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, left in enumerate(reference, 1):
        current = [row]
        for column, right in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left != right),
                )
            )
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str, language: str) -> tuple[str, int, int]:
    if language == "cmn":
        left = list("".join(reference.split()))
        right = list("".join(hypothesis.split()))
        metric = "cer"
    else:
        left = reference.lower().split()
        right = hypothesis.lower().split()
        metric = "wer"
    return metric, edit_distance(left, right), len(left)


def join_content_chunks(chunks: Sequence[str], language: str) -> str:
    normalized = [" ".join(value.split()) for value in chunks if value.strip()]
    return "".join(normalized) if language == "cmn" else " ".join(normalized)


def load_waveform(path: str) -> torch.Tensor:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz validation PCM, got {rate}: {path}")
    waveform = torch.from_numpy(values.mean(axis=1).copy())
    if not waveform.numel() or not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"invalid validation PCM: {path}")
    return waveform


def iter_selected(
    packs: Path,
    max_samples_per_task: int,
    *,
    worker_index: int = 0,
    num_workers: int = 1,
) -> Iterable[dict[str, object]]:
    if not 0 <= worker_index < num_workers:
        raise ValueError("invalid Stage A diagnosis worker partition")
    seen: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    with packs.open(encoding="utf-8") as handle:
        for pack_index, line in enumerate(handle):
            pack = json.loads(line)
            boundaries = pack["sample_boundaries"]
            for acoustic in pack.get("acoustics", []):
                task = str(acoustic["task"])
                if task not in TASKS:
                    continue
                occurrence = seen[task]
                seen[task] += 1
                if occurrence % num_workers != worker_index:
                    continue
                if max_samples_per_task and selected[task] >= max_samples_per_task:
                    continue
                boundary_index = int(acoustic["batch_boundary_index"])
                start, end = (int(value) for value in boundaries[boundary_index])
                conceptual = [int(value) for value in pack["tokens"][start:end]]
                conceptual.append(int(pack["labels"][end - 1]))
                flags = [False, *[bool(value) for value in pack["loss_mask"][start:end]]]
                glm_positions = [int(value) - start for value in acoustic["glm_positions"]]
                if len(conceptual) != len(flags) or len(glm_positions) != len(acoustic["source_glm"]):
                    raise ValueError("malformed Stage A validation sample geometry")
                selected[task] += 1
                yield {
                    "pack_index": pack_index,
                    "sample_id": str(acoustic["sample_id"]),
                    "task": task,
                    "language": str(acoustic["src_lang"]),
                    "reference": str(acoustic["canonical_transcript"]),
                    "source_audio": str(acoustic["source_audio"]),
                    "conceptual": conceptual,
                    "generated_flags": flags,
                    "glm_positions": glm_positions,
                    "source_glm": [int(value) for value in acoustic["source_glm"]],
                }
            if max_samples_per_task and all(
                selected[task] >= max_samples_per_task for task in TASKS
            ):
                return
    missing = (
        {
            task: max_samples_per_task - selected[task]
            for task in TASKS
            if selected[task] < max_samples_per_task
        }
        if max_samples_per_task
        else {}
    )
    if missing:
        raise RuntimeError(f"validation packs do not contain requested samples: {missing}")


def load_objective(checkpoint: Path, model_path: Path, device: torch.device) -> StageAObjective:
    objective = StageAObjective(
        TrainableSharedCausalWhisperVQ(model_path, gradient_checkpointing=False),
        qwen_hidden_size=896,
    ).to(device=device, dtype=torch.bfloat16).eval()
    state = {
        f"stage_a_objective.{name}": value
        for name, value in objective.state_dict().items()
    }
    dcp.load(state, checkpoint_id=str(checkpoint))
    return objective


@torch.inference_mode()
def acoustic_outputs(
    objective: StageAObjective,
    waveform: torch.Tensor,
    source_glm: Sequence[int],
    *,
    chunk_ms: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, object]]:
    device = next(objective.parameters()).device
    waveform = waveform.unsqueeze(0).to(device)
    lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = objective.frontend(waveform, lengths, chunk_ms=chunk_ms)
    hidden = output.pooled_hidden[0, : int(output.pooled_lengths[0])]
    if len(hidden) + 1 == len(source_glm):
        deficit = terminal_codec_extension_deficit_samples(
            int(lengths[0]), len(hidden), len(source_glm)
        )
        if deficit is None:
            raise ValueError("unaudited terminal causal-token extension during diagnosis")
        hidden = torch.cat((hidden, hidden[-1:]), dim=0)
    if len(hidden) != len(source_glm):
        raise ValueError(f"causal GLM length mismatch: {len(hidden)} vs {len(source_glm)}")
    codes = objective._nearest_codes(hidden)
    residual = objective.bridge_projection(objective.bridge_norm(hidden))
    ctc_logits = objective.ctc_head(output.frame_hidden)[0, : int(output.frame_lengths[0])]
    raw_ctc = ctc_logits.float().argmax(dim=-1).tolist()
    collapsed = collapse_ctc(raw_ctc, objective.ctc_blank_id)
    ctc_text = bytes(value for value in collapsed if 0 <= value < 256).decode(
        "utf-8", errors="replace"
    )
    diagnostic = {
        "input_frames": len(raw_ctc),
        "raw_nonblank_frames": sum(value != objective.ctc_blank_id for value in raw_ctc),
        "collapsed_nonblank_tokens": len(collapsed),
        "blank_ratio": sum(value == objective.ctc_blank_id for value in raw_ctc)
        / max(1, len(raw_ctc)),
        "text": ctc_text,
    }
    return (codes, residual), diagnostic


def prompt_embeddings(
    qwen,
    token_ids: Sequence[int],
    glm_indices: Sequence[int | None],
    speech_embeddings: torch.Tensor,
) -> torch.Tensor:
    device = speech_embeddings.device
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    embeddings = qwen.get_input_embeddings()(ids)
    positions = [index for index, value in enumerate(glm_indices) if value is not None]
    if positions:
        speech_indices = torch.tensor(
            [int(glm_indices[index]) for index in positions], dtype=torch.long, device=device
        )
        embeddings.index_copy_(
            0,
            torch.tensor(positions, dtype=torch.long, device=device),
            speech_embeddings.index_select(0, speech_indices).to(embeddings.dtype),
        )
    return embeddings.unsqueeze(0)


@torch.inference_mode()
def teacher_forced_accuracy(
    qwen,
    conceptual: Sequence[int],
    generated_flags: Sequence[bool],
    glm_map: dict[int, int],
    speech_embeddings: torch.Tensor,
    tokenizer_size: int,
) -> dict[str, object]:
    input_ids = [int(value) for value in conceptual[:-1]]
    targets = torch.tensor(conceptual[1:], dtype=torch.long, device=speech_embeddings.device)
    flags = torch.tensor(generated_flags[1:], dtype=torch.bool, device=speech_embeddings.device)
    glm_indices = [glm_map.get(index) for index in range(len(input_ids))]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = qwen(
            inputs_embeds=prompt_embeddings(qwen, input_ids, glm_indices, speech_embeddings),
            use_cache=False,
        ).logits[0].float()
    logits[:, tokenizer_size:] = -torch.inf
    predicted = logits.argmax(dim=-1)
    correct = (predicted == targets) & flags
    return {
        "target_tokens": int(flags.sum()),
        "correct_tokens": int(correct.sum()),
        "token_accuracy": float(correct.sum() / flags.sum().clamp_min(1)),
    }


@torch.inference_mode()
def generate_segment(
    qwen,
    prefix: Sequence[int],
    glm_indices: Sequence[int | None],
    speech_embeddings: torch.Tensor,
    *,
    stop_id: int,
    tokenizer_size: int,
    max_tokens: int,
) -> list[int]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = qwen(
            inputs_embeds=prompt_embeddings(qwen, prefix, glm_indices, speech_embeddings),
            use_cache=True,
        )
    cache = output.past_key_values
    logits = output.logits[:, -1].float()
    generated: list[int] = []
    for _ in range(max_tokens):
        logits[:, tokenizer_size:] = -torch.inf
        token = int(logits.argmax(dim=-1)[0])
        generated.append(token)
        if token == stop_id or token == c.TOKEN_EOS:
            break
        ids = torch.tensor([[token]], dtype=torch.long, device=logits.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = qwen(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1].float()
    return generated


def free_running_asr(
    qwen,
    tokenizer,
    conceptual: Sequence[int],
    generated_flags: Sequence[bool],
    glm_map: dict[int, int],
    speech_embeddings: torch.Tensor,
    *,
    language: str,
    max_event_tokens: int,
) -> dict[str, object]:
    built: list[int] = []
    built_glm: list[int | None] = []
    generated_all: list[int] = []
    generated_content: list[int] = []
    content_chunks: list[str] = []
    event_rows: list[dict[str, object]] = []
    cursor = 0
    for start, end in generated_runs(generated_flags):
        for position in range(cursor, start):
            built.append(int(conceptual[position]))
            built_glm.append(glm_map.get(position))
        expected = [int(value) for value in conceptual[start:end]]
        stop = expected[-1]
        maximum = min(max_event_tokens, max(8, len(expected) + 16))
        predicted = generate_segment(
            qwen,
            built,
            built_glm,
            speech_embeddings,
            stop_id=stop,
            tokenizer_size=len(tokenizer),
            max_tokens=maximum,
        )
        built.extend(predicted)
        built_glm.extend([None] * len(predicted))
        generated_all.extend(predicted)
        predicted_content = content_ids(predicted)
        if not predicted_content and c.TOKEN_START_CONTENT in conceptual[:start] and c.TOKEN_END_CONTENT in expected:
            content_end = (
                predicted.index(c.TOKEN_END_CONTENT)
                if c.TOKEN_END_CONTENT in predicted
                else len(predicted)
            )
            predicted_content = [
                value
                for value in predicted[:content_end]
                if value not in (c.TOKEN_END_CONTENT, c.TOKEN_EOS)
            ]
        generated_content.extend(predicted_content)
        decoded_content = tokenizer.decode(predicted_content, skip_special_tokens=True)
        if decoded_content.strip():
            content_chunks.append(decoded_content)
        event_rows.append(
            {
                "expected_tokens": len(expected),
                "generated_tokens": len(predicted),
                "expected_stop": stop,
                "reached_stop": bool(predicted and predicted[-1] == stop),
                "predicted_tokens": predicted,
                "content_tokens": predicted_content,
                "content_text": " ".join(decoded_content.split()),
                "write_structure": predicted[:3]
                == [c.TOKEN_WRITE_GENERATE, conceptual[start + 1], c.TOKEN_START_CONTENT]
                if len(expected) >= 3 and expected[0] == c.TOKEN_WRITE_GENERATE
                else None,
            }
        )
        cursor = end
    for position in range(cursor, len(conceptual)):
        built.append(int(conceptual[position]))
        built_glm.append(glm_map.get(position))
    structured = [row for row in event_rows if row["write_structure"] is not None]
    return {
        "text": join_content_chunks(content_chunks, language),
        "generated_tokens": generated_all,
        "generated_content_tokens": generated_content,
        "content_chunks": content_chunks,
        "events": event_rows,
        "content_events": sum(bool(row["content_tokens"]) for row in event_rows),
        "all_events_reached_stop": all(bool(row["reached_stop"]) for row in event_rows),
        "write_structure_rate": sum(row["write_structure"] is True for row in structured)
        / max(1, len(structured)),
    }


def markdown_report(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    lines = [
        "# Stage A checkpoint free-running diagnosis",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Evaluations: {summary['samples']}",
        f"- CTC blank collapse: **{summary['ctc_blank_collapse']}**",
        f"- AR final-only/empty collapse: **{summary['ar_empty_collapse']}**",
        f"- AR teacher-forced token accuracy: **{summary['ar_teacher_token_accuracy']:.4f}**",
        f"- Weighted CTC blank ratio: **{summary['ctc_blank_ratio']:.4f}**",
        f"- Weighted streaming WER/CER: **{summary['ar_error_rate_by_task'].get('streaming_asr', 0.0):.4f}**",
        f"- Weighted causal-full WER/CER: **{summary['ar_error_rate_by_task'].get('causal_full_asr', 0.0):.4f}**",
        "",
        "| chunk | task | sample | CTC blank | CTC nonblank | AR text | metric | error rate |",
        "|---:|---|---|---:|---:|---|---|---:|",
    ]
    for row in payload["samples"]:
        lines.append(
            "| {chunk_ms} | {task} | {sample_id} | {blank_ratio:.4f} | "
            "{collapsed_nonblank_tokens} | {text} | {metric} | {error_rate:.4f} |".format(
                chunk_ms=row["chunk_ms"],
                task=row["task"],
                sample_id=row["sample_id"],
                blank_ratio=row["ctc"]["blank_ratio"],
                collapsed_nonblank_tokens=row["ctc"]["collapsed_nonblank_tokens"],
                text=str(row["ar_free_running"]["text"]).replace("|", "\\|"),
                metric=row["ar_free_running"]["metric"],
                error_rate=row["ar_free_running"]["error_rate"],
            )
        )
    lines.extend(
        [
            "",
            "结论：CTC 与 AR 分支必须分开判定。CTC 全 blank 只说明辅助 CTC head 塌缩；"
            "只有 free-running AR 也为空、final-only 或高错误率时，才能判定 Stage A 主 ASR 路径失败。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    teacher_correct = sum(int(row["ar_teacher_forced"]["correct_tokens"]) for row in rows)
    teacher_tokens = sum(int(row["ar_teacher_forced"]["target_tokens"]) for row in rows)
    input_frames = sum(int(row["ctc"]["input_frames"]) for row in rows)
    nonblank_frames = sum(int(row["ctc"]["raw_nonblank_frames"]) for row in rows)
    error_by_task: dict[str, float] = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        errors = sum(int(row["ar_free_running"]["errors"]) for row in selected)
        units = sum(int(row["ar_free_running"]["reference_units"]) for row in selected)
        error_by_task[task] = errors / max(1, units)
    return {
        "samples": len(rows),
        "unique_samples": len({(row["task"], row["sample_id"]) for row in rows}),
        "evaluations_by_task": dict(Counter(str(row["task"]) for row in rows)),
        "evaluations_by_chunk_ms": dict(Counter(str(row["chunk_ms"]) for row in rows)),
        "ctc_blank_collapse": all(
            int(row["ctc"]["collapsed_nonblank_tokens"]) == 0 for row in rows
        ),
        "ctc_blank_ratio": (input_frames - nonblank_frames) / max(1, input_frames),
        "ar_empty_collapse": all(not str(row["ar_free_running"]["text"]) for row in rows),
        "ar_teacher_token_accuracy": teacher_correct / max(1, teacher_tokens),
        "ar_all_events_reached_stop_rate": sum(
            bool(row["ar_free_running"]["all_events_reached_stop"]) for row in rows
        )
        / max(1, len(rows)),
        "ar_error_rate_by_task": error_by_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--valid-packs", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=[960, 1280])
    parser.add_argument("--max-samples-per-task", type=int, default=2)
    parser.add_argument("--max-event-tokens", type=int, default=96)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite Stage A checkpoint diagnosis")
    if args.max_samples_per_task < 0 or args.max_event_tokens <= 0:
        raise ValueError("Stage A diagnosis limits are invalid")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("invalid Stage A diagnosis worker partition")
    if any(value <= 0 or value % 160 for value in args.chunk_ms):
        raise ValueError("chunk sizes must be positive multiples of 160 ms")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    qwen.requires_grad_(False)
    objective = load_objective(args.checkpoint, args.whispervq_model, device)
    selected = list(
        iter_selected(
            args.valid_packs,
            args.max_samples_per_task,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )
    )
    rows: list[dict[str, object]] = []
    for sample in selected:
        waveform = load_waveform(str(sample["source_audio"]))
        conceptual = sample["conceptual"]
        flags = sample["generated_flags"]
        glm_map = {
            int(position): index for index, position in enumerate(sample["glm_positions"])
        }
        for chunk_ms in args.chunk_ms:
            (codes, residual), ctc = acoustic_outputs(
                objective,
                waveform,
                sample["source_glm"],
                chunk_ms=int(chunk_ms),
            )
            base_ids = codes.long() + c.GLM_SEMANTIC_OFFSET
            speech_embeddings = qwen.get_input_embeddings()(base_ids) + residual.to(
                qwen.get_input_embeddings().weight.dtype
            )
            teacher = teacher_forced_accuracy(
                qwen,
                conceptual,
                flags,
                glm_map,
                speech_embeddings,
                len(tokenizer),
            )
            free = free_running_asr(
                qwen,
                tokenizer,
                conceptual,
                flags,
                glm_map,
                speech_embeddings,
                language=str(sample["language"]),
                max_event_tokens=args.max_event_tokens,
            )
            metric, errors, units = error_counts(
                str(sample["reference"]), str(free["text"]), str(sample["language"])
            )
            free.update(
                {
                    "metric": metric,
                    "errors": errors,
                    "reference_units": units,
                    "error_rate": errors / max(1, units),
                }
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "task": sample["task"],
                    "language": sample["language"],
                    "reference": sample["reference"],
                    "source_audio": sample["source_audio"],
                    "chunk_ms": int(chunk_ms),
                    "ctc": ctc,
                    "ar_teacher_forced": teacher,
                    "ar_free_running": free,
                }
            )
    summary = summarize_rows(rows)
    payload = {
        "schema_version": "uniss_quality_first_stage_a_checkpoint_diagnosis_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "hf_model": str(args.hf_model.resolve()),
        "valid_packs": str(args.valid_packs.resolve()),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "summary": summary,
        "samples": rows,
    }
    atomic_json(args.output_json.resolve(), payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
