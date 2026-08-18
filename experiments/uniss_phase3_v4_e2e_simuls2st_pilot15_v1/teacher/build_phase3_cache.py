#!/usr/bin/env python3
"""Build one GPU part of the future-safe Phase3 MT/semantic top-k cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.audit_rollouts import (
    _audit_pair,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.io import (
    atomic_json,
    partition_bounds,
    selected_total,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.cache import (
    CACHE_SCHEMA,
    combine_sample,
    save_bundle,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.requests import (
    Phase3TeacherRequest,
    build_phase3_requests,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index, write_index


PART_SCHEMA = "uniss_phase3_v4_e2e_phase3_teacher_part_v1"


def runtime_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("requests.py"),
        Path(__file__).with_name("cache.py"),
        Path(c.__file__),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Phase3Teacher:
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        topk: int,
        temperature: float,
    ) -> None:
        if topk <= 0 or temperature <= 0:
            raise ValueError("invalid Phase3 teacher top-k geometry")
        self.device = device
        self.topk = min(int(topk), c.VOCAB_SIZE)
        self.temperature = float(temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval().requires_grad_(False)
        if int(self.model.config.vocab_size) < c.VOCAB_SIZE:
            raise ValueError("Phase3 teacher vocabulary is smaller than UniSS")
        self.pad_id = int(self.tokenizer.pad_token_id or c.TOKEN_PAD)

    @torch.inference_mode()
    def summarize_batch(
        self, requests: Sequence[Phase3TeacherRequest]
    ) -> list[dict[str, np.ndarray]]:
        if not requests:
            return []
        sequences = [
            torch.tensor(
                [*request.prompt_ids, *request.target_ids],
                dtype=torch.long,
                device=self.device,
            )
            for request in requests
        ]
        maximum = max(len(value) for value in sequences)
        ids = torch.full(
            (len(sequences), maximum),
            self.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = sequence
            attention[row, : len(sequence)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.model.model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        selected: list[torch.Tensor] = []
        lengths: list[int] = []
        for row, request in enumerate(requests):
            positions = torch.tensor(
                [
                    len(request.prompt_ids) - 1 + index
                    for index in request.selected_target_indices
                ],
                dtype=torch.long,
                device=self.device,
            )
            selected.append(hidden[row].index_select(0, positions))
            lengths.append(len(positions))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.lm_head(torch.cat(selected, dim=0)).float()
        logical = logits[:, : c.VOCAB_SIZE]
        top1_values, top1 = logical.max(dim=-1)
        confidence = (top1_values - torch.logsumexp(logical, dim=-1)).exp()
        values, indices = torch.topk(
            logical / self.temperature,
            self.topk,
            dim=-1,
            sorted=True,
        )
        probabilities = F.softmax(values, dim=-1)
        output: list[dict[str, np.ndarray]] = []
        cursor = 0
        for length in lengths:
            stop = cursor + length
            output.append(
                {
                    "indices": indices[cursor:stop].to(torch.int32).cpu().numpy(),
                    "probabilities": probabilities[cursor:stop]
                    .to(torch.float16)
                    .cpu()
                    .numpy(),
                    "top1": top1[cursor:stop].to(torch.int32).cpu().numpy(),
                    "confidence": confidence[cursor:stop]
                    .to(torch.float16)
                    .cpu()
                    .numpy(),
                }
            )
            cursor = stop
        return output

    def summarize(
        self,
        requests: Sequence[Phase3TeacherRequest],
        *,
        max_padded_tokens: int,
        max_batch_size: int,
        max_selected_positions: int,
    ) -> list[dict[str, np.ndarray]]:
        if max_padded_tokens <= 0 or max_batch_size <= 0 or max_selected_positions <= 0:
            raise ValueError("Phase3 teacher batching limits must be positive")
        ordered = sorted(
            range(len(requests)),
            key=lambda index: len(requests[index].prompt_ids)
            + len(requests[index].target_ids),
        )
        output: list[dict[str, np.ndarray] | None] = [None] * len(requests)
        cursor = 0
        while cursor < len(ordered):
            batch_indices: list[int] = []
            maximum = 0
            selected_positions = 0
            while cursor < len(ordered) and len(batch_indices) < max_batch_size:
                index = ordered[cursor]
                length = len(requests[index].prompt_ids) + len(requests[index].target_ids)
                candidate_max = max(maximum, length)
                candidate_selected = selected_positions + len(
                    requests[index].selected_target_indices
                )
                if batch_indices and candidate_max * (len(batch_indices) + 1) > max_padded_tokens:
                    break
                if batch_indices and candidate_selected > max_selected_positions:
                    break
                batch_indices.append(index)
                maximum = candidate_max
                selected_positions = candidate_selected
                cursor += 1
            summaries = self.summarize_batch([requests[index] for index in batch_indices])
            for index, summary in zip(batch_indices, summaries):
                output[index] = summary
        if any(value is None for value in output):
            raise AssertionError("Phase3 teacher batching did not cover every request")
        return [value for value in output if value is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phase3-hf-sha256", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--semantic-stride", type=int, default=8)
    parser.add_argument("--sample-group-size", type=int, default=8)
    parser.add_argument("--records-per-bundle", type=int, default=64)
    parser.add_argument("--max-padded-tokens", type=int, default=65536)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-selected-positions", type=int, default=512)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Phase3 teacher part: {output_dir}")
    output_dir.mkdir(parents=True)
    if args.semantic_stride <= 0 or args.sample_group_size <= 0 or args.records_per_bundle <= 0:
        raise ValueError("Phase3 teacher cache geometry must be positive")
    gold_offsets, gold_total = selected_total(args.gold, None)
    rollout_offsets = load_index(args.rollouts)
    if rollout_offsets is None:
        raise ValueError("V1 rollout is missing its offset index")
    selection_start = int(args.start_index)
    total = gold_total - selection_start
    if args.limit is not None:
        total = min(total, max(0, int(args.limit)))
    if total <= 0 or len(rollout_offsets) < selection_start + total:
        raise ValueError("Phase3 teacher selection is empty or outside V1 rollouts")
    local_start, local_stop = partition_bounds(total, args.rank, args.world_size)
    start = selection_start + local_start
    stop = selection_start + local_stop
    if start == stop:
        raise ValueError("Phase3 teacher worker partition is empty")

    teacher = Phase3Teacher(
        args.model.resolve(),
        torch.device(args.device),
        topk=args.topk,
        temperature=args.temperature,
    )
    manifest = output_dir / "teacher_cache.jsonl"
    temporary_manifest = output_dir / f".teacher_cache.tmp.{os.getpid()}.jsonl"
    rows_pending: list[dict[str, object]] = []
    samples_pending: list[tuple[E2ETrajectory, list[Phase3TeacherRequest]]] = []
    counts: Counter[str] = Counter()
    byte_offsets: list[int] = []
    byte_offset = 0
    bundle_index = 0
    started = time.perf_counter()

    def write_rows(handle) -> None:
        nonlocal rows_pending, byte_offset, bundle_index
        if not rows_pending:
            return
        path = output_dir / "bundles" / f"bundle-{bundle_index:06d}.npz"
        for row in save_bundle(path, rows_pending):
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            byte_offsets.append(byte_offset)
            handle.write(encoded)
            byte_offset += len(encoded)
        rows_pending = []
        bundle_index += 1

    def process_samples(handle) -> None:
        nonlocal samples_pending, rows_pending
        if not samples_pending:
            return
        flat = [request for _, requests in samples_pending for request in requests]
        summaries = teacher.summarize(
            flat,
            max_padded_tokens=args.max_padded_tokens,
            max_batch_size=args.max_batch_size,
            max_selected_positions=args.max_selected_positions,
        )
        cursor = 0
        for trajectory, requests in samples_pending:
            stop_cursor = cursor + len(requests)
            arrays, descriptors = combine_sample(requests, summaries[cursor:stop_cursor])
            rows_pending.append(
                {
                    "sample_id": trajectory.sample_id,
                    "split": trajectory.split,
                    "source_manifest_record": trajectory.source_manifest_record,
                    "arrays": arrays,
                    "requests": descriptors,
                }
            )
            counts["records"] += 1
            counts["requests"] += len(requests)
            counts["teacher_positions"] += len(arrays["reference_label"])
            counts["teacher_top1_correct"] += int(
                np.count_nonzero(arrays["top1"] == arrays["reference_label"])
            )
            counts["reference_in_topk"] += int(
                np.count_nonzero(
                    (arrays["indices"] == arrays["reference_label"][:, None]).any(axis=1)
                )
            )
            for request in requests:
                counts[f"family:{request.family}"] += 1
                counts[f"history:{request.history_kind}"] += 1
                counts["content_candidate_tokens"] += request.content_candidate_tokens
                counts["content_selected_tokens"] += request.content_selected_tokens
            cursor = stop_cursor
            if len(rows_pending) >= args.records_per_bundle:
                write_rows(handle)
        if cursor != len(summaries):
            raise AssertionError("Phase3 teacher summary cursor did not close")
        samples_pending = []

    with args.gold.open("rb") as gold_handle, args.rollouts.open(
        "rb"
    ) as rollout_handle, temporary_manifest.open("wb") as manifest_handle:
        for record_index in range(start, stop):
            gold_handle.seek(int(gold_offsets[record_index]))
            trajectory = E2ETrajectory.from_mapping(json.loads(gold_handle.readline()))
            rollout_base = 0 if len(rollout_offsets) == gold_total else selection_start
            rollout_ordinal = record_index - rollout_base
            if not 0 <= rollout_ordinal < len(rollout_offsets):
                raise ValueError("Phase3 teacher rollout selection does not cover gold record")
            rollout_handle.seek(int(rollout_offsets[rollout_ordinal]))
            rollout = V1Rollout.from_mapping(json.loads(rollout_handle.readline()))
            _audit_pair(trajectory, rollout)
            requests = build_phase3_requests(
                trajectory,
                rollout,
                encode_text=lambda text: teacher.tokenizer.encode(
                    text, add_special_tokens=False
                ),
                semantic_stride=args.semantic_stride,
            )
            if not requests:
                raise ValueError(f"Phase3 teacher sample has no requests: {trajectory.sample_id}")
            samples_pending.append((trajectory, requests))
            if len(samples_pending) >= args.sample_group_size:
                process_samples(manifest_handle)
            if args.progress_interval and (record_index - start + 1) % args.progress_interval == 0:
                elapsed = max(1e-9, time.perf_counter() - started)
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "records": record_index - start + 1,
                            "assigned_records": stop - start,
                            "records_per_second": (record_index - start + 1) / elapsed,
                            "teacher_positions": counts["teacher_positions"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        process_samples(manifest_handle)
        write_rows(manifest_handle)
        manifest_handle.flush()
        os.fsync(manifest_handle.fileno())
    os.replace(temporary_manifest, manifest)
    index = write_index(manifest, byte_offsets)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": PART_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "status": "complete",
        "rank": args.rank,
        "world_size": args.world_size,
        "selection_start": selection_start,
        "selection_stop": selection_start + total,
        "assigned_start": start,
        "assigned_stop": stop,
        "gold": str(args.gold.resolve()),
        "rollouts": str(args.rollouts.resolve()),
        "model": str(args.model.resolve()),
        "phase3_hf_sha256": args.phase3_hf_sha256,
        "runtime_sha256": runtime_sha256(),
        "topk": args.topk,
        "temperature": args.temperature,
        "semantic_stride": args.semantic_stride,
        "max_padded_tokens": args.max_padded_tokens,
        "max_batch_size": args.max_batch_size,
        "max_selected_positions": args.max_selected_positions,
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": elapsed,
        "records_per_second": counts["records"] / max(1e-9, elapsed),
        "manifest": str(manifest.resolve()),
        "manifest_bytes": manifest.stat().st_size,
        "index": index,
    }
    atomic_json(output_dir / "PART_COMPLETE.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
