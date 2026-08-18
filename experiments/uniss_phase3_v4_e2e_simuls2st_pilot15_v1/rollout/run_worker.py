#!/usr/bin/env python3
"""One GPU worker for indexed, append-only V1 ASR rollout generation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    validate_trajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    IndexedJSONLWriter,
    atomic_json,
    iter_trajectories,
    partition_bounds,
    selected_total,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.persistent_runtime import (
    rollout_trajectory,
    runtime_sha256,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from training.simul_uniss.jsonl_index import write_index


WORKER_SCHEMA = "uniss_phase3_v4_e2e_simuls2st_v1_rollout_worker_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--v1-hf-sha256", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-event-tokens", type=int, default=96)
    parser.add_argument("--max-final-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_models(args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal V1 rollout requires an available CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    if int(qwen.config.vocab_size) < len(tokenizer):
        raise ValueError("V1 HF model vocabulary is smaller than its tokenizer")
    objective = stage_a_eval.load_objective(
        args.checkpoint,
        args.whispervq_model,
        device,
    ).eval().requires_grad_(False)
    frontend = make_cached_frontend(objective, device)
    return qwen, tokenizer, objective, frontend


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite rollout worker report: {args.report}")
    if len(args.v1_hf_sha256) != 64:
        raise ValueError("V1 HF fingerprint is not a SHA256 digest")
    if args.max_event_tokens <= 0 or args.max_final_tokens <= 0:
        raise ValueError("rollout generation limits must be positive")
    offsets, total = selected_total(args.input, args.limit)
    start, stop = partition_bounds(total, args.worker_index, args.num_workers)
    if start == stop:
        raise ValueError("rollout worker partition is empty")

    qwen, tokenizer, objective, frontend = load_models(args)
    expected_v1_sha: str | None = None
    counts: Counter[str] = Counter()
    weighted_errors: Counter[str] = Counter()
    started = time.perf_counter()
    writer = IndexedJSONLWriter(args.output)
    try:
        for record_index, trajectory in iter_trajectories(args.input, offsets, start, stop):
            validate_trajectory(
                trajectory,
                require_audio_hash=True,
                require_audio_audit=True,
            )
            if expected_v1_sha is None:
                expected_v1_sha = trajectory.v1_checkpoint_sha256
            elif trajectory.v1_checkpoint_sha256 != expected_v1_sha:
                raise ValueError("gold trajectory V1 checkpoint fingerprint changed")
            rollout = rollout_trajectory(
                trajectory,
                qwen=qwen,
                tokenizer=tokenizer,
                objective=objective,
                frontend=frontend,
                v1_hf_sha256=args.v1_hf_sha256,
                max_event_tokens=args.max_event_tokens,
                max_final_tokens=args.max_final_tokens,
            )
            writer.write(rollout.to_json())
            counts["records"] += 1
            counts["events"] += len(rollout.events)
            counts["empty_events"] += rollout.empty_events
            counts["early_eos_events"] += rollout.early_eos_events
            counts["malformed_write_events"] += rollout.malformed_write_events
            counts["final_eos_samples"] += int(rollout.final_reached_eos)
            counts[f"language:{rollout.src_lang}"] += 1
            weighted_errors[f"errors:{rollout.src_lang}"] += rollout.errors
            weighted_errors[f"units:{rollout.src_lang}"] += rollout.reference_units
            if counts["records"] % 100 == 0:
                elapsed = max(1e-9, time.perf_counter() - started)
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "records": counts["records"],
                            "partition_records": stop - start,
                            "records_per_second": counts["records"] / elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        output = writer.close()
    index = write_index(args.output, writer.offsets)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": WORKER_SCHEMA,
        "status": "complete",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "global_selected_records": total,
        "global_start": start,
        "global_stop": stop,
        "input": str(args.input.resolve()),
        "input_size_bytes": args.input.stat().st_size,
        "checkpoint": str(args.checkpoint.resolve()),
        "v1_checkpoint_sha256": expected_v1_sha,
        "hf_model": str(args.hf_model.resolve()),
        "v1_hf_sha256": args.v1_hf_sha256,
        "runtime_sha256": runtime_sha256(),
        "device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)),
        "max_event_tokens": args.max_event_tokens,
        "max_final_tokens": args.max_final_tokens,
        "counts": dict(sorted(counts.items())),
        "weighted_errors": dict(sorted(weighted_errors.items())),
        "elapsed_seconds": elapsed,
        "records_per_second": counts["records"] / max(1e-9, elapsed),
        "output": output,
        "index": index,
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
