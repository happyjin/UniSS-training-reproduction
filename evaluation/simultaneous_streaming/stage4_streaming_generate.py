"""Free-running Stage4 WAIT/WRITE, text, and semantic generation with vLLM.

Each request receives one additional source chunk.  The model first predicts a
single action token.  A WRITE action triggers autoregressive generation through
the end-semantic delimiter; a WAIT action immediately advances to the next
source chunk.  Independent samples are never packed into a shared causal
sequence, and every output directory is rank-specific and resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.vllm_generate import SuppressPaddedVocabulary
from training import constants_uniss as c
from training.generate_unist_eval_audio import write_jsonl_row


ACTION_IDS = (c.TOKEN_WAIT_READ, c.TOKEN_WRITE_GENERATE)


def batched(values: Iterable[object], size: int) -> Iterator[list[object]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def stage4_header(schedule: Mapping[str, object]) -> list[int]:
    return [
        c.TOKEN_TASK_STREAMING_S2ST,
        c.TOKEN_STREAMING_MODE,
        c.TOKEN_DYNAMIC_MODE,
        c.language_token_id(str(schedule["tgt_lang"])),
        c.speed_token_id(1.0),
        *c.wrap_global_tokens(schedule["speaker_tokens"]),  # type: ignore[arg-type]
    ]


def source_chunk_tokens(event: Mapping[str, object]) -> list[int]:
    return [
        c.TOKEN_START_GLM,
        *c.encode_glm_semantic(event["source_glm"]),  # type: ignore[arg-type]
        c.TOKEN_END_GLM,
    ]


def _between(ids: Sequence[int], start_token: int, end_token: int) -> tuple[list[int], bool, bool]:
    try:
        start = ids.index(start_token)
    except ValueError:
        return [], False, False
    try:
        end = ids.index(end_token, start + 1)
    except ValueError:
        return list(ids[start + 1 :]), True, False
    return list(ids[start + 1 : end]), True, True


def parse_write_tokens(token_ids: Sequence[int], tokenizer) -> dict[str, object]:
    ids = [int(value) for value in token_ids]
    text_ids, has_content_start, has_content_end = _between(
        ids, c.TOKEN_START_CONTENT, c.TOKEN_END_CONTENT
    )
    semantic_ids, has_semantic_start, has_semantic_end = _between(
        ids, c.TOKEN_START_SEMANTIC, c.TOKEN_END_SEMANTIC
    )
    semantic_values = [
        c.BICODEC_SEMANTIC_SPAN.value_for(token_id)
        for token_id in semantic_ids
        if c.BICODEC_SEMANTIC_OFFSET <= token_id <= c.BICODEC_SEMANTIC_SPAN.last_id
    ]
    invalid_semantic_tokens = sum(
        not (c.BICODEC_SEMANTIC_OFFSET <= token_id <= c.BICODEC_SEMANTIC_SPAN.last_id)
        for token_id in semantic_ids
    )
    return {
        "text_ids": text_ids,
        "text": tokenizer.decode(text_ids, skip_special_tokens=False).strip(),
        "semantic_values": semantic_values,
        "has_content_start": has_content_start,
        "has_content_end": has_content_end,
        "has_semantic_start": has_semantic_start,
        "has_semantic_end": has_semantic_end,
        "invalid_semantic_tokens": invalid_semantic_tokens,
    }


def normalized_write_tail(parsed: Mapping[str, object], target_language: str) -> list[int]:
    return [
        c.language_token_id(target_language),
        c.TOKEN_START_CONTENT,
        *[int(value) for value in parsed["text_ids"]],  # type: ignore[index]
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_SEMANTIC,
        *c.encode_bicodec_semantic(parsed["semantic_values"]),  # type: ignore[arg-type]
        c.TOKEN_END_SEMANTIC,
    ]


def output_token_ids(candidate) -> list[int]:
    values = [int(value) for value in candidate.token_ids]
    stop_reason = candidate.stop_reason
    if isinstance(stop_reason, int) and (not values or values[-1] != stop_reason):
        values.append(stop_reason)
    return values


def request_timing(request_output) -> dict[str, float | None]:
    metrics = request_output.metrics
    if metrics is None:
        return {
            "request_seconds": None,
            "queue_seconds": None,
            "ttft_seconds": None,
            "decode_seconds": None,
        }
    request_seconds = float(metrics.finished_time - metrics.arrival_time)
    ttft_seconds = float(metrics.first_token_time - metrics.arrival_time)
    return {
        "request_seconds": request_seconds,
        "queue_seconds": float(metrics.time_in_queue),
        "ttft_seconds": ttft_seconds,
        "decode_seconds": max(0.0, request_seconds - ttft_seconds),
    }


@dataclass
class GenerationState:
    index: int
    schedule: dict[str, object]
    prompt_ids: list[int]
    event_trace: list[dict[str, object]] = field(default_factory=list)
    generated_text_ids: list[int] = field(default_factory=list)
    semantic_chunks: list[list[int]] = field(default_factory=list)
    forced_actions: int = 0
    structural_recoveries: int = 0
    max_prompt_tokens: int = 0
    training_context_exceeded: bool = False


def account_prompt_length(state: GenerationState, training_context_limit: int) -> None:
    state.max_prompt_tokens = max(state.max_prompt_tokens, len(state.prompt_ids))
    if len(state.prompt_ids) > training_context_limit:
        state.training_context_exceeded = True


def load_states(
    schedules_path: Path,
    *,
    rank: int,
    world_size: int,
    limit_records: int | None,
    completed: set[int],
) -> list[GenerationState]:
    states: list[GenerationState] = []
    for index, schedule in enumerate(iter_jsonl(schedules_path)):
        if limit_records is not None and limit_records > 0 and index >= limit_records:
            break
        if index % world_size != rank or index in completed:
            continue
        state = GenerationState(
                index=index,
                schedule=dict(schedule),
                prompt_ids=stage4_header(schedule),
            )
        state.max_prompt_tokens = len(state.prompt_ids)
        states.append(state)
    return states


def prepare_output(
    output_dir: Path,
    *,
    rank: int,
    config: Mapping[str, object],
    resume: bool,
) -> tuple[Path, Path, set[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"run_config.rank{rank:03d}.json"
    results_path = output_dir / f"generation.rank{rank:03d}.jsonl"
    marker = output_dir / f"GENERATION_COMPLETE.rank{rank:03d}"
    if marker.is_file() and resume:
        completed = {int(row["index"]) for row in iter_jsonl(results_path)}
        return results_path, marker, completed
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"refusing to reuse rank output: {config_path}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != dict(config):
            raise ValueError("resume configuration does not match existing Stage4 run")
    else:
        if results_path.exists() and results_path.stat().st_size:
            raise ValueError(f"rank result exists without configuration: {results_path}")
        write_json(config_path, config)
    results_path.touch(exist_ok=True)
    completed = {int(row["index"]) for row in iter_jsonl(results_path)}
    return results_path, marker, completed


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    from vllm import LLM, SamplingParams, __version__ as vllm_version

    rank = args.rank
    world_size = args.world_size
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    output_dir = Path(args.output_dir)
    config = {
        "schema_version": "simul_uniss_stage4_generation_config_v1",
        "streaming_mode": args.streaming_mode,
        "model": str(Path(args.model).resolve()),
        "schedules": str(Path(args.schedules).resolve()),
        "rank": rank,
        "world_size": world_size,
        "limit_records": args.limit_records,
        "batch_records": args.batch_records,
        "max_write_tokens": args.max_write_tokens,
        "max_model_len": args.max_model_len,
        "training_context_limit": args.training_context_limit,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "repetition_penalty": args.repetition_penalty,
        "dtype": args.dtype,
        "seed": args.seed,
        "vllm_version": vllm_version,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1", ""),
        "decode": "greedy",
    }
    results_path, marker, completed = prepare_output(
        output_dir,
        rank=rank,
        config=config,
        resume=args.resume,
    )
    if marker.is_file() and args.resume:
        return json.loads((output_dir / f"generation_summary.rank{rank:03d}.json").read_text())

    states = load_states(
        Path(args.schedules),
        rank=rank,
        world_size=world_size,
        limit_records=args.limit_records,
        completed=completed,
    )
    if not states and not completed:
        raise ValueError(f"rank {rank} was assigned no Stage4 schedules")
    if not states:
        summary = {
            "schema_version": "simul_uniss_stage4_generation_summary_v1",
            "rank": rank,
            "world_size": world_size,
            "completed_before_resume": len(completed),
            "generated": 0,
            "total_completed": len(completed),
            "events": 0,
            "generated_tokens": 0,
            "invalid_actions": 0,
            "structural_recoveries": 0,
            "elapsed_seconds": 0.0,
            "record_batches_completed": 0,
        }
        write_json(output_dir / f"generation_summary.rank{rank:03d}.json", summary)
        marker.write_text("complete\n", encoding="utf-8")
        return summary

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_seq_len_to_capture=args.max_seq_len_to_capture,
        trust_remote_code=False,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        enable_prefix_caching=True,
    )
    action_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=args.seed,
        logits_processors=[SuppressPaddedVocabulary(c.VOCAB_SIZE)],
    )
    write_sampling = SamplingParams(
        temperature=0.0,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_write_tokens,
        seed=args.seed,
        stop_token_ids=[
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_WAIT_READ,
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_EOS,
        ],
        include_stop_str_in_output=True,
        logits_processors=[SuppressPaddedVocabulary(c.VOCAB_SIZE)],
    )

    started = time.time()
    newly_completed = 0
    total_events = 0
    total_generated_tokens = 0
    invalid_actions = 0
    structural_recoveries = 0
    training_context_exceeded = 0
    for state_batch_index, raw_batch in enumerate(batched(states, args.batch_records)):
        state_batch = [value for value in raw_batch if isinstance(value, GenerationState)]
        max_events = max(len(state.schedule["events"]) for state in state_batch)  # type: ignore[arg-type]
        for event_index in range(max_events):
            active = [
                state
                for state in state_batch
                if event_index < len(state.schedule["events"])  # type: ignore[arg-type]
            ]
            if not active:
                continue
            for state in active:
                event = state.schedule["events"][event_index]  # type: ignore[index]
                state.prompt_ids.extend(source_chunk_tokens(event))
                account_prompt_length(state, args.training_context_limit)
                if len(state.prompt_ids) >= args.max_model_len:
                    raise ValueError(
                        f"{state.schedule['id']}: prompt length {len(state.prompt_ids)} "
                        f"reached max_model_len {args.max_model_len}"
                    )

            action_started = time.perf_counter()
            action_outputs = llm.generate(
                [{"prompt_token_ids": state.prompt_ids} for state in active],
                action_sampling,
                use_tqdm=False,
            )
            action_wall_seconds = time.perf_counter() - action_started
            writes: list[tuple[GenerationState, dict[str, object]]] = []
            for state, request_output in zip(active, action_outputs):
                event = state.schedule["events"][event_index]  # type: ignore[index]
                candidate = request_output.outputs[0]
                raw_action = int(candidate.token_ids[0]) if candidate.token_ids else -1
                predicted_action = raw_action
                forced_reason: str | None = None
                if predicted_action not in ACTION_IDS:
                    forced_reason = "invalid_action"
                    invalid_actions += 1
                    predicted_action = (
                        c.TOKEN_WRITE_GENERATE
                        if bool(event["source_is_final"])
                        else c.TOKEN_WAIT_READ
                    )
                elif bool(event["source_is_final"]) and predicted_action == c.TOKEN_WAIT_READ:
                    forced_reason = "final_flush"
                    predicted_action = c.TOKEN_WRITE_GENERATE
                if forced_reason is not None:
                    state.forced_actions += 1
                state.prompt_ids.append(predicted_action)
                account_prompt_length(state, args.training_context_limit)
                trace: dict[str, object] = {
                    "event_index": event_index,
                    "chunk_index": int(event["chunk_index"]),
                    "source_glm_end": int(event["source_glm_end"]),
                    "source_end_ms": float(event["source_end_ms"]),
                    "source_is_final": bool(event["source_is_final"]),
                    "reference_action": str(event["action"]),
                    "raw_action_token_id": raw_action,
                    "raw_action": (
                        "wait"
                        if raw_action == c.TOKEN_WAIT_READ
                        else "write"
                        if raw_action == c.TOKEN_WRITE_GENERATE
                        else "other"
                    ),
                    "action": "write" if predicted_action == c.TOKEN_WRITE_GENERATE else "wait",
                    "forced_reason": forced_reason,
                    "eligible_proxy": int(event["target_ctc_count_proxy"])
                    > len(state.generated_text_ids),
                    "action_batch_wall_seconds": action_wall_seconds,
                    **{f"action_{key}": value for key, value in request_timing(request_output).items()},
                }
                state.event_trace.append(trace)
                total_events += 1
                total_generated_tokens += 1
                if predicted_action == c.TOKEN_WRITE_GENERATE:
                    writes.append((state, trace))

            if writes:
                write_started = time.perf_counter()
                write_outputs = llm.generate(
                    [{"prompt_token_ids": state.prompt_ids} for state, _ in writes],
                    write_sampling,
                    use_tqdm=False,
                )
                write_wall_seconds = time.perf_counter() - write_started
                for (state, trace), request_output in zip(writes, write_outputs):
                    candidate = request_output.outputs[0]
                    raw_tail = output_token_ids(candidate)
                    parsed = parse_write_tokens(raw_tail, tokenizer)
                    structurally_valid = bool(
                        parsed["has_content_start"]
                        and parsed["has_content_end"]
                        and parsed["has_semantic_start"]
                        and parsed["has_semantic_end"]
                        and int(parsed["invalid_semantic_tokens"]) == 0
                    )
                    normalized = normalized_write_tail(parsed, str(state.schedule["tgt_lang"]))
                    if raw_tail != normalized:
                        state.structural_recoveries += 1
                        structural_recoveries += 1
                    state.prompt_ids.extend(normalized)
                    account_prompt_length(state, args.training_context_limit)
                    text_ids = [int(value) for value in parsed["text_ids"]]  # type: ignore[index]
                    semantic_values = [int(value) for value in parsed["semantic_values"]]  # type: ignore[index]
                    state.generated_text_ids.extend(text_ids)
                    state.semantic_chunks.append(semantic_values)
                    trace.update(
                        {
                            "write_raw_token_ids": raw_tail,
                            "write_normalized_token_ids": normalized,
                            "write_finish_reason": candidate.finish_reason,
                            "write_stop_reason": candidate.stop_reason,
                            "write_structurally_valid": structurally_valid,
                            "generated_text": parsed["text"],
                            "generated_text_ids": text_ids,
                            "generated_semantic_values": semantic_values,
                            "generated_semantic_count": len(semantic_values),
                            "write_batch_wall_seconds": write_wall_seconds,
                            **{
                                f"write_{key}": value
                                for key, value in request_timing(request_output).items()
                            },
                        }
                    )
                    total_generated_tokens += len(raw_tail)

        for state in state_batch:
            schedule = state.schedule
            reference_prefix_ids: list[int] = []
            generated_prefix_ids: list[int] = []
            for trace, reference_event in zip(state.event_trace, schedule["events"]):  # type: ignore[arg-type]
                reference_event_text_ids = [
                    int(value) for value in reference_event.get("target_text_ids", [])
                ]
                generated_event_text_ids = [
                    int(value) for value in trace.get("generated_text_ids", [])
                ]
                reference_prefix_ids.extend(reference_event_text_ids)
                generated_prefix_ids.extend(generated_event_text_ids)
                trace.update(
                    {
                        "reference_target_text_ids": reference_event_text_ids,
                        "reference_semantic_count": len(reference_event.get("target_semantic", [])),
                        "reference_prefix_text_ids": list(reference_prefix_ids),
                        "generated_prefix_text_ids": list(generated_prefix_ids),
                        "reference_prefix_text": tokenizer.decode(
                            reference_prefix_ids, skip_special_tokens=False
                        ).strip(),
                        "generated_prefix_text": tokenizer.decode(
                            generated_prefix_ids, skip_special_tokens=False
                        ).strip(),
                    }
                )
            source_bicodec = [
                int(token)
                for event in schedule["events"]  # type: ignore[index]
                for token in event["source_bicodec"]
            ]
            reference_semantic = [
                int(token)
                for event in schedule["events"]  # type: ignore[index]
                if event["action"] == "write"
                for token in event["target_semantic"]
            ]
            semantic_values = [token for chunk in state.semantic_chunks for token in chunk]
            row = {
                "schema_version": "simul_uniss_stage4_generation_result_v1",
                "index": state.index,
                "id": schedule["id"],
                "mode": args.streaming_mode,
                "split": schedule.get("split", "dev"),
                "src_lang": schedule["src_lang"],
                "tgt_lang": schedule["tgt_lang"],
                "dataset_name": schedule.get("dataset_name", "unknown"),
                "transcription_ref": schedule["transcription"],
                "translation_ref": schedule["translation"],
                "generated_translation": tokenizer.decode(
                    state.generated_text_ids, skip_special_tokens=False
                ).strip(),
                "generated_text_ids": state.generated_text_ids,
                "semantic_values": semantic_values,
                "semantic_chunks": state.semantic_chunks,
                "semantic_token_count": len(semantic_values),
                "speaker_tokens": schedule["speaker_tokens"],
                "source_bicodec_values": source_bicodec,
                "reference_semantic_values": reference_semantic,
                "source_glm_length": schedule["source_glm_length"],
                "reference_target_text_length": schedule["target_text_length"],
                "source_duration_ms_proxy": float(schedule["events"][-1]["source_end_ms"]),  # type: ignore[index]
                "chunk_ms": schedule["chunk_ms"],
                "event_trace": state.event_trace,
                "forced_action_count": state.forced_actions,
                "structural_recovery_count": state.structural_recoveries,
                "max_prompt_tokens": state.max_prompt_tokens,
                "training_context_limit": args.training_context_limit,
                "training_context_exceeded": state.training_context_exceeded,
                "checkpoint": str(Path(args.model).resolve()),
                "seed": args.seed,
                "decode": "greedy",
            }
            write_jsonl_row(results_path, row)
            newly_completed += 1
            training_context_exceeded += int(state.training_context_exceeded)

        summary = {
            "schema_version": "simul_uniss_stage4_generation_summary_v1",
            "rank": rank,
            "world_size": world_size,
            "completed_before_resume": len(completed),
            "generated": newly_completed,
            "total_completed": len(completed) + newly_completed,
            "events": total_events,
            "generated_tokens": total_generated_tokens,
            "invalid_actions": invalid_actions,
            "structural_recoveries": structural_recoveries,
            "training_context_exceeded": training_context_exceeded,
            "elapsed_seconds": time.time() - started,
            "record_batches_completed": state_batch_index + 1,
        }
        write_json(output_dir / f"generation_summary.rank{rank:03d}.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    marker.write_text("complete\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--batch-records", type=int, default=512)
    parser.add_argument("--max-write-tokens", type=int, default=700)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--training-context-limit", type=int, default=18000)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=262144)
    parser.add_argument("--max-seq-len-to-capture", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--streaming-mode", default="streaming_stage4")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "auto"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_generation(parse_args(argv)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
