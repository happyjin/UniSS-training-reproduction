"""Generate UniSS outputs from fixed UniST manifests with resumable vLLM batches."""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.uniss_outputs import parse_with_tokenizer
from training import constants_uniss as c
from training.generate_unist_eval_audio import (
    build_eval_sample,
    iter_manifest_records,
    load_hf_text_encoder,
    truncate_at_eos,
    write_jsonl_row,
)


class SuppressPaddedVocabulary:
    """Mask Megatron's padded dummy embedding rows during sampling."""

    def __init__(self, logical_vocab_size: int):
        self.logical_vocab_size = logical_vocab_size

    def __call__(self, *args):
        logits = args[-1]
        logits[self.logical_vocab_size :] = float("-inf")
        return logits


def batched(iterator: Iterable[object], size: int) -> Iterator[list[object]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    source = iter(iterator)
    while batch := list(islice(source, size)):
        yield batch


def result_key(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row["id"]), str(row["mode"])


def iter_jobs(records: Iterable[Mapping[str, object]], modes: Sequence[str], text_encoder):
    for record_index, record in enumerate(records):
        for mode in modes:
            sample = build_eval_sample(record, mode=mode, text_encoder=text_encoder)
            yield {
                "record_index": record_index,
                "record": record,
                "mode": mode,
                "prompt_ids": sample.prompt_ids,
            }


def validate_resume_config(existing: Mapping[str, object], current: Mapping[str, object]) -> None:
    keys = (
        "model",
        "manifest",
        "modes",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
        "seed",
        "vllm_use_v1",
    )
    mismatches = {key: (existing.get(key), current.get(key)) for key in keys if existing.get(key) != current.get(key)}
    if mismatches:
        raise ValueError(f"Resume configuration mismatch: {mismatches}")


def prepare_output_directory(
    output_dir: Path,
    *,
    current_config: Mapping[str, object],
    resume: bool,
) -> set[tuple[str, str]]:
    """Initialize or resume output, including a pre-first-batch interruption.

    The run config is written before vLLM model initialization. If the process
    is interrupted during initialization, the directory and config exist but
    no result file has been created yet. That state is a valid zero-result
    checkpoint and should resume without deleting or overwriting the directory.
    """

    results_path = output_dir / "generation_results.jsonl"
    config_path = output_dir / "run_config.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"Refusing to reuse vLLM output without --resume: {output_dir}")
        if config_path.is_file():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            validate_resume_config(existing_config, current_config)
        elif any(output_dir.iterdir()):
            raise ValueError(f"Cannot resume output without run config: {output_dir}")
        else:
            write_json(config_path, current_config)
        results_path.touch(exist_ok=True)
        return {result_key(row) for row in iter_jsonl(results_path)}

    output_dir.mkdir(parents=True)
    write_json(config_path, current_config)
    results_path.touch()
    return set()


def run_generation(args: argparse.Namespace) -> dict[str, object]:
    from vllm import LLM, SamplingParams, __version__ as vllm_version

    output_dir = Path(args.output_dir)
    results_path = output_dir / "generation_results.jsonl"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "generation_summary.json"
    current_config = {
        "backend": "vllm",
        "vllm_version": vllm_version,
        "vllm_use_v1": os.environ.get("VLLM_USE_V1", ""),
        "model": str(Path(args.model).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "modes": list(args.mode),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "num_scheduler_steps": args.num_scheduler_steps,
        "max_seq_len_to_capture": args.max_seq_len_to_capture,
        "request_batch_size": args.request_batch_size,
        "dtype": args.dtype,
    }

    completed = prepare_output_directory(
        output_dir,
        current_config=current_config,
        resume=args.resume,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    text_encoder = load_hf_text_encoder(tokenizer)
    records = iter_manifest_records(Path(args.manifest), limit_records=args.limit_records)
    pending_jobs = (
        job
        for job in iter_jobs(records, args.mode, text_encoder)
        if (str(job["record"]["id"]), str(job["mode"])) not in completed  # type: ignore[index]
    )

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_scheduler_steps=args.num_scheduler_steps,
        max_seq_len_to_capture=args.max_seq_len_to_capture,
        trust_remote_code=False,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=[c.TOKEN_EOS],
        include_stop_str_in_output=True,
        logits_processors=[SuppressPaddedVocabulary(c.VOCAB_SIZE)],
    )

    counts = {
        "completed_before_resume": len(completed),
        "generated": 0,
        "no_semantic_tokens": 0,
        "missing_translation": 0,
        "dummy_generated_tokens": 0,
    }
    started = time.time()
    for job_batch in batched(pending_jobs, args.request_batch_size):
        prompts = [{"prompt_token_ids": job["prompt_ids"]} for job in job_batch]  # type: ignore[index]
        batch_started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=args.show_progress)
        batch_seconds = time.perf_counter() - batch_started
        if len(outputs) != len(job_batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(job_batch)} prompts")
        for job, request_output in zip(job_batch, outputs):
            candidate = request_output.outputs[0]
            token_ids = truncate_at_eos(candidate.token_ids)
            mode = str(job["mode"])
            record = job["record"]  # type: ignore[assignment]
            parsed = parse_with_tokenizer(token_ids, mode=mode, tokenizer=tokenizer)
            row = {
                "index": job["record_index"],
                "id": record.get("id"),
                "mode": mode,
                "src_lang": record.get("src_lang"),
                "tgt_lang": record.get("tgt_lang"),
                "dataset_name": record.get("dataset_name"),
                "transcription_ref": record.get("transcription"),
                "translation_ref": record.get("translation"),
                "generated_text_raw": tokenizer.decode(token_ids, skip_special_tokens=False),
                "generated_transcription": parsed["generated_transcription"],
                "generated_translation": parsed["generated_translation"],
                "generated_token_ids": token_ids,
                "semantic_values": parsed["semantic_values"],
                "semantic_token_count": len(parsed["semantic_values"]),
                "dummy_token_count": sum(token_id >= c.VOCAB_SIZE for token_id in token_ids),
                "has_semantic_start": parsed["has_semantic_start"],
                "has_semantic_end": parsed["has_semantic_end"],
                "has_eos": parsed["has_eos"],
                "finish_reason": candidate.finish_reason,
                "stop_reason": candidate.stop_reason,
                "batch_generation_seconds": batch_seconds,
                "checkpoint": str(Path(args.model).resolve()),
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
            write_jsonl_row(results_path, row)
            counts["generated"] += 1
            if not parsed["semantic_values"]:
                counts["no_semantic_tokens"] += 1
            if mode in {"quality", "performance"} and not parsed["generated_translation"]:
                counts["missing_translation"] += 1
            counts["dummy_generated_tokens"] += int(row["dummy_token_count"])
        write_json(
            summary_path,
            {
                **counts,
                "elapsed_seconds": time.time() - started,
                "total_results": len(completed) + counts["generated"],
            },
        )
    return {
        **counts,
        "elapsed_seconds": time.time() - started,
        "total_results": len(completed) + counts["generated"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", nargs="+", choices=("quality", "performance"), default=["quality", "performance"])
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--num-scheduler-steps", type=int, default=1)
    parser.add_argument("--max-seq-len-to-capture", type=int, default=8192)
    parser.add_argument("--request-batch-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "auto"), default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_generation(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
