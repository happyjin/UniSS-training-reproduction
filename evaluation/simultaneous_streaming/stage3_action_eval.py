"""Evaluate Simul-UniSS Stage3 WAIT/WRITE logits on one GPU rank.

This evaluator intentionally uses unpacked samples with an attention mask.  It
does not concatenate independent examples under a normal causal mask, so one
sample can never attend to another sample's answer.  Only hidden states at
action prediction positions are projected through the full LM head, avoiding a
full ``batch x sequence x vocabulary`` logits tensor.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from training import constants_uniss as c


ACTION_IDS = (c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE)
ACTION_NAMES = {
    c.TOKEN_WAIT_READ: "wait",
    c.TOKEN_WRITE_GENERATE: "write",
}


@dataclass(frozen=True)
class EvaluationRecord:
    index: int
    sample_id: str
    input_ids: list[int]
    action_positions: list[int]
    action_labels: list[int]
    events: list[dict[str, object]]
    src_lang: str
    tgt_lang: str
    dataset_name: str

    @property
    def length(self) -> int:
        return len(self.input_ids)


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _action_positions(sample: dict[str, object]) -> tuple[list[int], list[int]]:
    input_ids = sample.get("input_ids")
    token_weights = sample.get("token_weights")
    if not isinstance(input_ids, list) or not all(isinstance(value, int) for value in input_ids):
        raise TypeError("input_ids must be a list of ints")
    if not isinstance(token_weights, list) or len(token_weights) != len(input_ids):
        raise TypeError("token_weights must be a list matching input_ids")
    weighted_positions = [
        index for index, weight in enumerate(token_weights) if float(weight) > 0.0
    ]
    positions = [index for index in weighted_positions if input_ids[index] in ACTION_IDS]
    if positions != weighted_positions:
        unexpected = [(index, input_ids[index]) for index in weighted_positions if index not in positions]
        raise ValueError(f"non-action tokens have positive weight: {unexpected[:5]}")
    if not positions or positions[0] == 0:
        raise ValueError("sample must contain action tokens after at least one context token")
    return positions, [input_ids[index] for index in positions]


def load_records(
    samples_path: Path,
    schedules_path: Path,
    *,
    rank: int,
    world_size: int,
    limit_records: int | None = None,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    sentinel = object()
    pairs = itertools.zip_longest(iter_jsonl(samples_path), iter_jsonl(schedules_path), fillvalue=sentinel)
    for index, pair in enumerate(pairs):
        sample, schedule = pair
        if sample is sentinel or schedule is sentinel:
            raise ValueError("sample and schedule JSONL files have different record counts")
        if limit_records is not None and index >= limit_records:
            break
        if index % world_size != rank:
            continue
        assert isinstance(sample, dict) and isinstance(schedule, dict)
        sample_id = str(sample.get("id", ""))
        if sample_id != str(schedule.get("id", "")):
            raise ValueError(
                f"sample/schedule id mismatch at record {index}: "
                f"{sample_id!r} != {schedule.get('id')!r}"
            )
        positions, labels = _action_positions(sample)
        events = schedule.get("events")
        if not isinstance(events, list) or len(events) != len(positions):
            raise ValueError(
                f"{sample_id}: schedule events ({len(events) if isinstance(events, list) else 'invalid'}) "
                f"do not match actions ({len(positions)})"
            )
        event_labels = [
            c.TOKEN_WAIT_READ if str(event.get("action")) == "wait" else c.TOKEN_WRITE_GENERATE
            for event in events
        ]
        if labels != event_labels:
            raise ValueError(f"{sample_id}: sample action labels do not match schedule events")
        input_ids = sample["input_ids"]
        assert isinstance(input_ids, list)
        records.append(
            EvaluationRecord(
                index=index,
                sample_id=sample_id,
                input_ids=input_ids,
                action_positions=positions,
                action_labels=labels,
                events=events,
                src_lang=str(schedule.get("src_lang", "unknown")),
                tgt_lang=str(schedule.get("tgt_lang", "unknown")),
                dataset_name=str(schedule.get("dataset_name", "unknown")),
            )
        )
    return records


def build_batches(
    records: Sequence[EvaluationRecord],
    *,
    max_batch_tokens: int,
    max_batch_size: int,
) -> Iterator[list[EvaluationRecord]]:
    if max_batch_tokens <= 0 or max_batch_size <= 0:
        raise ValueError("batch limits must be positive")
    current: list[EvaluationRecord] = []
    current_max_length = 0
    for record in sorted(records, key=lambda item: (item.length, item.index)):
        next_max = max(current_max_length, record.length)
        next_size = len(current) + 1
        if current and (next_size > max_batch_size or next_max * next_size > max_batch_tokens):
            yield current
            current = []
            current_max_length = 0
            next_max = record.length
        if record.length > max_batch_tokens:
            raise ValueError(
                f"sample {record.sample_id} length {record.length} exceeds max_batch_tokens "
                f"{max_batch_tokens}; truncation is forbidden"
            )
        current.append(record)
        current_max_length = next_max
    if current:
        yield current


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _safe_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _open_partial(path: Path):
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    return temporary, temporary.open("w", encoding="utf-8")


def _resolve_distributed_args(args: argparse.Namespace) -> tuple[int, int, int]:
    local_rank = args.local_rank
    rank = args.rank
    world_size = args.world_size
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank is None:
        rank = int(os.environ.get("RANK", str(local_rank)))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    return local_rank, rank, world_size


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    local_rank, rank, world_size = _resolve_distributed_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage3 full evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / f"COMPLETE.rank{rank:03d}"
    summary_path = output_dir / f"summary.rank{rank:03d}.json"
    events_path = output_dir / f"events.rank{rank:03d}.jsonl"
    samples_output_path = output_dir / f"samples.rank{rank:03d}.jsonl"
    for path in (summary_path, events_path, samples_output_path):
        if path.exists() and not (args.recover_completed and marker.is_file()):
            raise FileExistsError(f"refusing to overwrite existing rank output: {path}")
    if args.recover_completed and marker.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    load_started = time.perf_counter()
    records = load_records(
        Path(args.samples),
        Path(args.schedules),
        rank=rank,
        world_size=world_size,
        limit_records=args.limit_records,
    )
    if not records:
        raise ValueError(f"rank {rank} was assigned no records")
    batches = list(
        build_batches(
            records,
            max_batch_tokens=args.max_batch_tokens,
            max_batch_size=args.max_batch_size,
        )
    )
    data_load_seconds = time.perf_counter() - load_started

    model_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_dtype(args.dtype),
        attn_implementation=args.attention_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    model.config.use_cache = False
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError("expected a Qwen-style causal LM with .model and .lm_head")
    if model.config.vocab_size <= max(ACTION_IDS):
        raise ValueError(
            f"model vocab {model.config.vocab_size} does not include action token {max(ACTION_IDS)}"
        )
    model_load_seconds = time.perf_counter() - model_started

    if args.warmup_batches > 0:
        warmup_batch = batches[min(len(batches) - 1, args.warmup_batches - 1)]
        warmup_batch = warmup_batch[: min(len(warmup_batch), args.warmup_batch_size)]
        max_length = max(record.length for record in warmup_batch)
        warmup_ids = torch.full(
            (len(warmup_batch), max_length), c.TOKEN_PAD, dtype=torch.long, device=device
        )
        warmup_mask = torch.zeros_like(warmup_ids)
        for row, record in enumerate(warmup_batch):
            length = record.length
            warmup_ids[row, :length] = torch.tensor(record.input_ids, device=device)
            warmup_mask[row, :length] = 1
        with torch.inference_mode():
            for _ in range(args.warmup_batches):
                _ = model.model(
                    input_ids=warmup_ids,
                    attention_mask=warmup_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    forward_seconds = 0.0
    head_seconds = 0.0
    padded_tokens = 0
    actual_tokens = 0
    action_events = 0
    ce_values: list[float] = []
    batch_sizes: list[int] = []
    batch_padded_lengths: list[int] = []

    event_partial, event_handle = _open_partial(events_path)
    sample_partial, sample_handle = _open_partial(samples_output_path)
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches):
                batch_size = len(batch)
                max_length = max(record.length for record in batch)
                input_ids = torch.full(
                    (batch_size, max_length), c.TOKEN_PAD, dtype=torch.long, device=device
                )
                attention_mask = torch.zeros_like(input_ids)
                selected_rows: list[int] = []
                selected_positions: list[int] = []
                selected_labels: list[int] = []
                selected_metadata: list[tuple[EvaluationRecord, int]] = []
                for row, record in enumerate(batch):
                    length = record.length
                    input_ids[row, :length] = torch.tensor(record.input_ids, device=device)
                    attention_mask[row, :length] = 1
                    for event_index, (position, label) in enumerate(
                        zip(record.action_positions, record.action_labels)
                    ):
                        selected_rows.append(row)
                        selected_positions.append(position - 1)
                        selected_labels.append(label)
                        selected_metadata.append((record, event_index))

                torch.cuda.synchronize(device)
                started = time.perf_counter()
                hidden = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                selected_hidden = hidden[
                    torch.tensor(selected_rows, device=device),
                    torch.tensor(selected_positions, device=device),
                ]
                torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - started

                predictions: list[dict[str, object]] = []
                started = time.perf_counter()
                for start in range(0, len(selected_labels), args.logit_event_batch):
                    end = min(len(selected_labels), start + args.logit_event_batch)
                    logits = model.lm_head(selected_hidden[start:end]).float()
                    labels = torch.tensor(selected_labels[start:end], device=device)
                    target_logits = logits.gather(1, labels[:, None]).squeeze(1)
                    token_ce = torch.logsumexp(logits, dim=-1) - target_logits
                    global_predictions = logits.argmax(dim=-1)
                    binary_logits = logits[:, list(ACTION_IDS)]
                    binary_probabilities = F.softmax(binary_logits, dim=-1)
                    binary_predictions = binary_logits.argmax(dim=-1)
                    for offset in range(end - start):
                        global_id = int(global_predictions[offset])
                        binary_index = int(binary_predictions[offset])
                        ce = float(token_ce[offset])
                        predictions.append(
                            {
                                "global_prediction_id": global_id,
                                "binary_prediction_id": ACTION_IDS[binary_index],
                                "binary_wait_probability": float(binary_probabilities[offset, 0]),
                                "binary_write_probability": float(binary_probabilities[offset, 1]),
                                "target_ce": ce,
                                "target_probability": float(math.exp(-min(ce, 80.0))),
                            }
                        )
                        ce_values.append(ce)
                    del logits, labels, target_logits, token_ce, global_predictions
                    del binary_logits, binary_probabilities, binary_predictions
                torch.cuda.synchronize(device)
                head_seconds += time.perf_counter() - started

                per_sample_predictions: dict[str, list[dict[str, object]]] = {
                    record.sample_id: [] for record in batch
                }
                for metadata, prediction, label in zip(
                    selected_metadata, predictions, selected_labels
                ):
                    record, event_index = metadata
                    event = record.events[event_index]
                    reference_action = ACTION_NAMES[label]
                    binary_prediction_id = int(prediction["binary_prediction_id"])
                    global_prediction_id = int(prediction["global_prediction_id"])
                    event_result = {
                        "sample_id": record.sample_id,
                        "record_index": record.index,
                        "event_index": event_index,
                        "chunk_index": int(event.get("chunk_index", event_index)),
                        "source_end_ms": float(event.get("source_end_ms", 0.0)),
                        "source_is_final": bool(event.get("source_is_final", False)),
                        "src_lang": record.src_lang,
                        "tgt_lang": record.tgt_lang,
                        "dataset_name": record.dataset_name,
                        "reference_action": reference_action,
                        "reference_action_id": label,
                        "binary_prediction": ACTION_NAMES[binary_prediction_id],
                        "binary_prediction_id": binary_prediction_id,
                        "global_prediction_id": global_prediction_id,
                        "global_prediction_action": ACTION_NAMES.get(global_prediction_id, "other"),
                        **prediction,
                    }
                    event_handle.write(
                        json.dumps(event_result, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    per_sample_predictions[record.sample_id].append(event_result)

                for record in batch:
                    sample_events = per_sample_predictions[record.sample_id]
                    reference_first = next(
                        (
                            float(event["source_end_ms"])
                            for event in sample_events
                            if event["reference_action"] == "write"
                        ),
                        None,
                    )
                    predicted_first = next(
                        (
                            float(event["source_end_ms"])
                            for event in sample_events
                            if event["binary_prediction"] == "write"
                        ),
                        None,
                    )
                    final_event = sample_events[-1]
                    sample_result = {
                        "sample_id": record.sample_id,
                        "record_index": record.index,
                        "src_lang": record.src_lang,
                        "tgt_lang": record.tgt_lang,
                        "dataset_name": record.dataset_name,
                        "input_tokens": record.length,
                        "events": len(sample_events),
                        "reference_first_write_ms": reference_first,
                        "predicted_first_write_ms": predicted_first,
                        "first_write_delta_ms": (
                            None
                            if reference_first is None or predicted_first is None
                            else predicted_first - reference_first
                        ),
                        "predicted_write_count": sum(
                            event["binary_prediction"] == "write" for event in sample_events
                        ),
                        "reference_write_count": sum(
                            event["reference_action"] == "write" for event in sample_events
                        ),
                        "final_binary_prediction": final_event["binary_prediction"],
                        "final_flush_success": final_event["binary_prediction"] == "write",
                        "mean_target_ce": sum(float(event["target_ce"]) for event in sample_events)
                        / len(sample_events),
                    }
                    sample_handle.write(
                        json.dumps(sample_result, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )

                actual_tokens += sum(record.length for record in batch)
                padded_tokens += batch_size * max_length
                action_events += len(selected_labels)
                batch_sizes.append(batch_size)
                batch_padded_lengths.append(max_length)
                if args.progress_interval and (batch_index + 1) % args.progress_interval == 0:
                    elapsed = max(time.perf_counter() - run_started, 1e-9)
                    print(
                        json.dumps(
                            {
                                "rank": rank,
                                "batch": batch_index + 1,
                                "batches": len(batches),
                                "samples": sum(batch_sizes),
                                "tokens_per_second": actual_tokens / elapsed,
                                "events_per_second": action_events / elapsed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                del input_ids, attention_mask, hidden, selected_hidden
        torch.cuda.synchronize(device)
    except BaseException:
        event_handle.close()
        sample_handle.close()
        event_partial.unlink(missing_ok=True)
        sample_partial.unlink(missing_ok=True)
        raise
    else:
        event_handle.close()
        sample_handle.close()
        event_partial.replace(events_path)
        sample_partial.replace(samples_output_path)

    inference_seconds = time.perf_counter() - run_started
    mean_ce = sum(ce_values) / len(ce_values)
    summary: dict[str, object] = {
        "schema_version": "simul_uniss_stage3_action_rank_summary_v1",
        "split": args.split,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "model": str(Path(args.model).resolve()),
        "samples_path": str(Path(args.samples).resolve()),
        "schedules_path": str(Path(args.schedules).resolve()),
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "logit_event_batch": args.logit_event_batch,
        "samples": len(records),
        "action_events": action_events,
        "actual_tokens": actual_tokens,
        "padded_tokens": padded_tokens,
        "padding_efficiency": actual_tokens / padded_tokens,
        "batches": len(batches),
        "batch_size_mean": sum(batch_sizes) / len(batch_sizes),
        "batch_size_p95": _percentile([float(value) for value in batch_sizes], 0.95),
        "padded_length_p95": _percentile(
            [float(value) for value in batch_padded_lengths], 0.95
        ),
        "mean_target_ce": mean_ce,
        "target_perplexity": math.exp(min(mean_ce, 50.0)),
        "data_load_seconds": data_load_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "forward_seconds": forward_seconds,
        "lm_head_seconds": head_seconds,
        "samples_per_second": len(records) / inference_seconds,
        "tokens_per_second": actual_tokens / inference_seconds,
        "padded_tokens_per_second": padded_tokens / inference_seconds,
        "events_per_second": action_events / inference_seconds,
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    _safe_write_json(summary_path, summary)
    marker.write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--max-batch-tokens", type=int, default=131072)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--logit-event-batch", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--warmup-batch-size", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--recover-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
