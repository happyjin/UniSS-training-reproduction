#!/usr/bin/env python3
"""Build compact causal-WhisperVQ and Phase3-teacher trajectory caches.

Each process owns one GPU and a disjoint shard subset. Source audio is decoded
from UniST BiCodec tokens in memory, consumed immediately, and never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.build_trajectory_schedule import (
    plans_for_row,
)
from experiments.uniss_phase3_true_subsecond_deadline_full198_v1.data.schema import (
    Action,
    TrajectoryRecord,
)
from training import constants_uniss as c


CACHE_PART_SCHEMA = "uniss_true_subsecond_trajectory_cache_part_v2"
REQUIRED_COLUMNS = (
    "id",
    "transcription",
    "translation",
    "source_glm",
    "source_bicodec",
    "target_bicodec",
    "bicodec_global",
    "src_lang",
    "tgt_lang",
)


def stable_prefix_length(
    reference: Sequence[int],
    predictions: Sequence[Sequence[int]],
    confidences: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> tuple[int, tuple[bool, ...]]:
    if len(predictions) != 4 or len(confidences) != 4:
        raise ValueError("current/future1/future2/full teacher outputs are required")
    safe = []
    for index, token in enumerate(reference):
        accepted = all(
            index < len(prediction)
            and index < len(confidence)
            and int(prediction[index]) == int(token)
            and float(confidence[index]) >= threshold
            for prediction, confidence in zip(predictions, confidences)
        )
        safe.append(accepted)
    length = 0
    for accepted in safe:
        if not accepted:
            break
        length += 1
    return length, tuple(safe)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _pad_sequences(values: Sequence[Sequence[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(value) for value in values)
    ids = torch.full((len(values), maximum), c.TOKEN_PAD, dtype=torch.long, device=device)
    mask = torch.zeros((len(values), maximum), dtype=torch.long, device=device)
    for row, sequence in enumerate(values):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, : len(sequence)] = 1
    return ids, mask


def trim_decoded_waveforms(
    waveform: torch.Tensor, semantic_lengths: Sequence[int], *, samples_per_token: int = 320
) -> list[torch.Tensor]:
    if waveform.shape[0] != len(semantic_lengths):
        raise ValueError("decoded waveform batch size does not match semantic lengths")
    if samples_per_token <= 0:
        raise ValueError("samples_per_token must be positive")
    result = []
    for row, token_length in enumerate(semantic_lengths):
        expected_samples = int(token_length) * samples_per_token
        flattened = waveform[row].reshape(-1)
        if expected_samples <= 0 or expected_samples > flattened.numel():
            raise ValueError("decoded waveform is shorter than its semantic-token clock")
        result.append(flattened[:expected_samples].contiguous())
    return result


class BatchedBiCodecDecoder:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        from uniss.speech_tokenizer.bicodec.models.bicodec import BiCodec

        self.device = device
        self.model = BiCodec.load_from_checkpoint(checkpoint).to(device).eval()

    @torch.inference_mode()
    def decode(
        self, globals_: Sequence[Sequence[int]], semantics: Sequence[Sequence[int]]
    ) -> list[torch.Tensor]:
        lengths = [len(value) for value in semantics]
        maximum = max(lengths)
        semantic = torch.zeros(len(semantics), maximum, dtype=torch.long, device=self.device)
        for row, values in enumerate(semantics):
            semantic[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=self.device)
        global_tokens = torch.tensor(globals_, dtype=torch.long, device=self.device).unsqueeze(1)
        waveform = self.model.detokenize(semantic, global_tokens).detach().float().cpu()
        return trim_decoded_waveforms(waveform, lengths)


class Phase3Teacher:
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        topk: int = 32,
        temperature: float = 1.5,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.topk = topk
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()

    def encode_text(self, text: str) -> list[int]:
        values = self.tokenizer.encode(text, add_special_tokens=False)
        if not values:
            raise ValueError("translation encoded to an empty token sequence")
        return [int(value) for value in values]

    @staticmethod
    def prompt(row: dict[str, Any], source_glm: Sequence[int]) -> list[int]:
        return [
            c.TOKEN_TASK_STREAMING_S2TT,
            c.TOKEN_STREAMING_MODE,
            c.TOKEN_DYNAMIC_MODE,
            c.language_token_id(str(row["tgt_lang"])),
            *c.wrap_global_tokens([int(value) for value in row["bicodec_global"]]),
            c.TOKEN_START_GLM,
            *c.encode_glm_semantic(source_glm),
            c.TOKEN_END_GLM,
            c.TOKEN_WRITE_GENERATE,
            c.language_token_id(str(row["tgt_lang"])),
            c.TOKEN_START_CONTENT,
        ]

    @torch.inference_mode()
    def summarize(
        self, requests: Sequence[tuple[list[int], list[int]]]
    ) -> list[dict[str, np.ndarray]]:
        sequences = [[*prompt, *target] for prompt, target in requests]
        ids, attention = _pad_sequences(sequences, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.model.model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        selected = []
        lengths = []
        for row, (prompt, target) in enumerate(requests):
            selected.append(hidden[row, len(prompt) - 1 : len(prompt) - 1 + len(target)])
            lengths.append(len(target))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.lm_head(torch.cat(selected, dim=0)).float()
        split = torch.split(logits, lengths, dim=0)
        output = []
        for value in split:
            raw_probability = F.softmax(value, dim=-1)
            confidence, top1 = raw_probability.max(dim=-1)
            scaled = value / self.temperature
            top_values, top_indices = torch.topk(scaled, min(self.topk, scaled.shape[-1]), dim=-1)
            output.append(
                {
                    "indices": top_indices.cpu().to(torch.int32).numpy(),
                    "probabilities": F.softmax(top_values, dim=-1).cpu().to(torch.float16).numpy(),
                    "top1": top1.cpu().to(torch.int32).numpy(),
                    "confidence": confidence.cpu().to(torch.float16).numpy(),
                }
            )
        return output


def _prefix(tokens: Sequence[int], end_ms: int) -> list[int]:
    # WhisperVQ emits one linguistic token every 80 ms.
    count = max(1, min(len(tokens), (end_ms + 79) // 80))
    return [int(value) for value in tokens[:count]]


def _block_size(sample_id: str, kind: str) -> int:
    value = int.from_bytes(hashlib.blake2b(f"{sample_id}:{kind}".encode(), digest_size=2).digest(), "big")
    return (8, 12, 16)[value % 3]


def teacher_bundle_reference(cache_file: Path, request_index: int) -> str:
    if request_index < 0:
        raise ValueError("teacher request index must be non-negative")
    return f"{cache_file}::teacher:{request_index}"


def causal_bundle_reference(cache_file: Path, row_index: int) -> str:
    if row_index < 0:
        raise ValueError("causal-token row index must be non-negative")
    return f"{cache_file}::causal:{row_index}"


def build_records_for_row(
    *,
    shard: int,
    row_index: int,
    row: dict[str, Any],
    causal_tokens: Sequence[int],
    translation_ids: Sequence[int],
    summaries: Sequence[dict[str, np.ndarray]],
    cache_file: Path,
    cache_row_index: int,
    request_offset: int,
    confidence_threshold: float,
) -> tuple[TrajectoryRecord, TrajectoryRecord]:
    plans = plans_for_row(shard, row_index, row)
    records = []
    previous = 0
    target_semantic = [int(value) for value in row["target_bicodec"]]
    for plan_index, plan in enumerate(plans):
        group = summaries[plan_index * 4 : plan_index * 4 + 4]
        stable, safe = stable_prefix_length(
            translation_ids,
            [value["top1"] for value in group],
            [value["confidence"] for value in group],
            threshold=confidence_threshold,
        )
        stable = max(previous, stable)
        supported = stable - previous
        natural = Action.WRITE if supported > 0 else Action.READ
        deadline = Action.WRITE if supported > 0 or plan.chunk_end_ms >= 800 else Action.READ
        forced = deadline is Action.WRITE and supported == 0
        block = _block_size(plan.sample_id, plan.trajectory_kind)
        semantic_start = min(
            max(0, round(previous / max(1, len(translation_ids)) * len(target_semantic))),
            max(0, len(target_semantic) - block),
        )
        semantic_end = min(len(target_semantic), semantic_start + block)
        if semantic_end - semantic_start < block:
            semantic_start = max(0, semantic_end - block)
        history_end = semantic_start
        history_start = max(0, history_end - 200)
        base = request_offset + plan_index * 4
        record = TrajectoryRecord(
            sample_id=plan.sample_id,
            shard=shard,
            row_index=row_index,
            src_lang=plan.src_lang,
            tgt_lang=plan.tgt_lang,
            source_duration_ms=plan.source_duration_ms,
            chunk_end_ms=plan.chunk_end_ms,
            future_1_end_ms=plan.future_1_end_ms,
            future_2_end_ms=plan.future_2_end_ms,
            causal_source_glm=tuple(_prefix(causal_tokens, plan.chunk_end_ms)),
            future_1_source_glm=tuple(_prefix(causal_tokens, plan.future_1_end_ms)),
            future_2_source_glm=tuple(_prefix(causal_tokens, plan.future_2_end_ms)),
            frontend_token_cache=causal_bundle_reference(cache_file, cache_row_index),
            translation_ids=tuple(int(value) for value in translation_ids),
            teacher_prefix_topk_path=teacher_bundle_reference(cache_file, base),
            teacher_future_1_topk_path=teacher_bundle_reference(cache_file, base + 1),
            teacher_future_2_topk_path=teacher_bundle_reference(cache_file, base + 2),
            teacher_full_topk_path=teacher_bundle_reference(cache_file, base + 3),
            previous_committed_length=previous,
            stable_target_length=stable,
            new_supported_count=supported,
            support_bucket=min(supported, 4),
            safe_commit_mask=safe,
            natural_action_target=natural,
            deadline_action_target=deadline,
            deadline_forced_target=forced,
            target_text_delta_ids=tuple(int(value) for value in translation_ids[previous:stable]),
            semantic_history_start=history_start,
            semantic_history_end=history_end,
            semantic_target_start=semantic_start,
            semantic_target_end=semantic_end,
            speaker_global=tuple(int(value) for value in row["bicodec_global"]),
            quality_flags=("deadline_anticipation_soft_target",) if forced else (),
        ).with_checksum()
        records.append(record)
        previous = stable
    return records[0], records[1]


def _save_bundle(path: Path, summaries: Sequence[dict[str, np.ndarray]], causal_tokens: Sequence[Sequence[int]]) -> None:
    values: dict[str, np.ndarray] = {}
    values["bundle_schema"] = np.asarray([CACHE_PART_SCHEMA])
    for index, summary in enumerate(summaries):
        for name, array in summary.items():
            values[f"request_{index}_{name}"] = array
    offsets = [0]
    flattened = []
    for sequence in causal_tokens:
        flattened.extend(int(value) for value in sequence)
        offsets.append(len(flattened))
    values["causal_tokens"] = np.asarray(flattened, dtype=np.int16)
    values["causal_token_offsets"] = np.asarray(offsets, dtype=np.int64)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


def process_shard(args: argparse.Namespace, shard: int, decoder, whisper, teacher) -> dict[str, Any]:
    output_dir = Path(args.output_root) / f"part-{shard:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "PART_COMPLETE.json"
    output = output_dir / "trajectory_cache.jsonl"
    if marker.is_file() and output.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("schema_version") == CACHE_PART_SCHEMA:
            return value
    source = Path(args.raw_unist_dir) / f"train-{shard:05d}.parquet"
    index_root = Path(args.index_root)
    accepted = np.sort(
        np.concatenate(
            (
                np.load(index_root / f"train-{shard:05d}.eng.npy", mmap_mode="r"),
                np.load(index_root / f"train-{shard:05d}.cmn.npy", mmap_mode="r"),
            )
        )
    )
    table = pq.read_table(source, columns=list(REQUIRED_COLUMNS))
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    started = time.time()
    try:
        with temporary.open("wb") as handle:
            for batch_number, start in enumerate(range(0, len(accepted), args.batch_size)):
                row_indices = [int(value) for value in accepted[start : start + args.batch_size]]
                rows = table.take(pa.array(row_indices, type=pa.int64())).to_pylist()
                waveforms = decoder.decode(
                    [[int(value) for value in row["bicodec_global"]] for row in rows],
                    [[int(value) for value in row["source_bicodec"]] for row in rows],
                )
                whisper_outputs = whisper.encode([(waveform, 16_000) for waveform in waveforms])
                translation_ids = [teacher.encode_text(str(row["translation"])) for row in rows]
                requests = []
                for cache_row_index, (row_index, row, output_row, text_ids) in enumerate(
                    zip(row_indices, rows, whisper_outputs, translation_ids)
                ):
                    causal = [int(value) for value in output_row.tokens]
                    for plan in plans_for_row(shard, row_index, row):
                        for end_ms in (
                            plan.chunk_end_ms,
                            plan.future_1_end_ms,
                            plan.future_2_end_ms,
                            plan.source_duration_ms,
                        ):
                            requests.append((teacher.prompt(row, _prefix(causal, end_ms)), text_ids))
                summaries = teacher.summarize(requests)
                cache_file = output_dir / f"bundle-{batch_number:06d}.npz"
                _save_bundle(
                    cache_file,
                    summaries,
                    [[int(value) for value in output_row.tokens] for output_row in whisper_outputs],
                )
                cursor = 0
                for cache_row_index, (row_index, row, output_row, text_ids) in enumerate(
                    zip(row_indices, rows, whisper_outputs, translation_ids)
                ):
                    records = build_records_for_row(
                        shard=shard,
                        row_index=row_index,
                        row=row,
                        causal_tokens=[int(value) for value in output_row.tokens],
                        translation_ids=text_ids,
                        summaries=summaries[cursor : cursor + 8],
                        cache_file=cache_file,
                        cache_row_index=cache_row_index,
                        request_offset=cursor,
                        confidence_threshold=args.confidence_threshold,
                    )
                    cursor += 8
                    for record in records:
                        encoded = (json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        handle.write(encoded)
                        counts["trajectories"] += 1
                        counts[f"action:{record.natural_action_target.value}"] += 1
                        counts["deadline_forced"] += int(record.deadline_forced_target)
                if args.progress_interval and (start + len(rows)) % args.progress_interval < len(rows):
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "rank": args.rank,
                                "shard": shard,
                                "rows": start + len(rows),
                                "rows_per_second": (start + len(rows)) / elapsed,
                            }
                        ),
                        flush=True,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    value = {
        "schema_version": CACHE_PART_SCHEMA,
        "rank": args.rank,
        "shard": shard,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "accepted_rows": len(accepted),
        "trajectory_count": counts["trajectories"],
        "natural_write": counts["action:WRITE"],
        "natural_read": counts["action:READ"],
        "deadline_forced": counts["deadline_forced"],
        "elapsed_seconds": time.time() - started,
    }
    if value["trajectory_count"] != 2 * value["accepted_rows"]:
        raise AssertionError("cache did not materialize two trajectories per accepted row")
    _atomic_json(marker, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-unist-dir", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase3-model", required=True, type=Path)
    parser.add_argument("--whispervq-model", required=True, type=Path)
    parser.add_argument("--bicodec-checkpoint", required=True, type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--progress-interval", type=int, default=1000)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch.cuda.set_device(args.rank)
    device = torch.device(f"cuda:{args.rank}")
    from training.simul_uniss.subsecond_v2.streaming_whispervq_teacher import (
        StreamingWhisperVQTeacher,
    )

    decoder = BatchedBiCodecDecoder(args.bicodec_checkpoint, device)
    whisper = StreamingWhisperVQTeacher(
        args.whispervq_model,
        device=str(device),
        chunk_ms=160,
        right_context_ms=80,
    )
    teacher = Phase3Teacher(
        args.phase3_model,
        device,
        topk=args.topk,
        temperature=args.temperature,
    )
    shards = list(range(args.rank, 198, args.world_size))
    if args.limit_shards is not None:
        shards = shards[: args.limit_shards]
    results = []
    for shard in shards:
        if args.limit_records is not None:
            # Smoke mode writes to a dedicated output root and trims its index
            # arrays before invoking this program; formal mode never sets it.
            raise ValueError("limit-records must be implemented by a smoke index, not formal cache")
        results.append(process_shard(args, shard, decoder, whisper, teacher))
    print(json.dumps({"rank": args.rank, "parts": results}, sort_keys=True))


if __name__ == "__main__":
    main()
