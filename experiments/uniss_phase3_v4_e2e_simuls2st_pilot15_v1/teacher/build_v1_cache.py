#!/usr/bin/env python3
"""Build one GPU part of the same-prefix V1 ASR top-k cache."""

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
    file_sha256,
    partition_bounds,
    selected_total,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.persistent_runtime import (
    _speech_embeddings,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.rollout.schema import (
    V1Rollout,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_cache import (
    V1_CACHE_SCHEMA,
    combine_v1_sample,
    save_v1_bundle,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.teacher.v1_requests import (
    V1TeacherSequence,
    build_v1_teacher_sequences,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (
    make_cached_frontend,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index, write_index


V1_PART_SCHEMA = "uniss_phase3_v4_e2e_v1_asr_teacher_part_v1"


def runtime_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("v1_requests.py"),
        Path(__file__).with_name("v1_cache.py"),
        Path(_speech_embeddings.__code__.co_filename),
        Path(c.__file__),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class V1ASRTeacher:
    def __init__(
        self,
        checkpoint: Path,
        hf_model: Path,
        whispervq_model: Path,
        device: torch.device,
        *,
        topk: int,
        temperature: float,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("V1 teacher cache requires an available CUDA device")
        if topk <= 0 or temperature <= 0:
            raise ValueError("invalid V1 teacher top-k geometry")
        self.device = device
        self.topk = min(int(topk), c.VOCAB_SIZE)
        self.temperature = float(temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model, local_files_only=True
        )
        self.qwen = AutoModelForCausalLM.from_pretrained(
            hf_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval().requires_grad_(False)
        if int(self.qwen.config.vocab_size) < c.VOCAB_SIZE:
            raise ValueError("V1 teacher vocabulary is smaller than UniSS")
        self.objective = stage_a_eval.load_objective(
            checkpoint, whispervq_model, device
        ).eval().requires_grad_(False)
        self.frontend = make_cached_frontend(self.objective, device)
        self.pad_id = int(self.tokenizer.pad_token_id or c.TOKEN_PAD)

    @torch.inference_mode()
    def speech_embeddings(self, trajectory: E2ETrajectory) -> torch.Tensor:
        return _speech_embeddings(
            self.objective,
            self.frontend,
            self.qwen,
            trajectory,
        )

    @torch.inference_mode()
    def summarize(
        self,
        sequences: Sequence[V1TeacherSequence],
        speech_embeddings: torch.Tensor,
    ) -> list[dict[str, np.ndarray]]:
        if not sequences:
            return []
        maximum = max(len(value.token_ids) for value in sequences)
        hidden_size = int(self.qwen.config.hidden_size)
        embeddings = torch.zeros(
            (len(sequences), maximum, hidden_size),
            dtype=self.qwen.get_input_embeddings().weight.dtype,
            device=self.device,
        )
        attention = torch.zeros(
            (len(sequences), maximum), dtype=torch.long, device=self.device
        )
        for row, sequence in enumerate(sequences):
            ids = torch.tensor(
                sequence.token_ids, dtype=torch.long, device=self.device
            )
            values = self.qwen.get_input_embeddings()(ids)
            positions = [
                index
                for index, source in enumerate(sequence.speech_indices)
                if source is not None
            ]
            if positions:
                source = torch.tensor(
                    [
                        int(sequence.speech_indices[index])
                        for index in positions
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                values.index_copy_(
                    0,
                    torch.tensor(positions, dtype=torch.long, device=self.device),
                    speech_embeddings.index_select(0, source).to(values.dtype),
                )
            embeddings[row, : len(sequence.token_ids)] = values
            attention[row, : len(sequence.token_ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.qwen.model(
                inputs_embeds=embeddings,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        selected: list[torch.Tensor] = []
        lengths: list[int] = []
        for row, sequence in enumerate(sequences):
            positions = torch.tensor(
                sequence.selected_predictor_positions,
                dtype=torch.long,
                device=self.device,
            )
            selected.append(hidden[row].index_select(0, positions))
            lengths.append(len(positions))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.qwen.lm_head(torch.cat(selected, dim=0)).float()
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
                    "indices": indices[cursor:stop]
                    .to(torch.int32)
                    .cpu()
                    .numpy(),
                    "probabilities": probabilities[cursor:stop]
                    .to(torch.float16)
                    .cpu()
                    .numpy(),
                    "top1": top1[cursor:stop]
                    .to(torch.int32)
                    .cpu()
                    .numpy(),
                    "confidence": confidence[cursor:stop]
                    .to(torch.float16)
                    .cpu()
                    .numpy(),
                }
            )
            cursor = stop
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--whispervq-model", type=Path, required=True)
    parser.add_argument("--v1-hf-sha256", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--records-per-bundle", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V1 teacher part: {output_dir}")
    output_dir.mkdir(parents=True)
    if args.records_per_bundle <= 0:
        raise ValueError("V1 teacher records per bundle must be positive")
    gold_offsets, gold_total = selected_total(args.gold, None)
    rollout_offsets = load_index(args.rollouts)
    if rollout_offsets is None:
        raise ValueError("V1 rollout is missing its offset index")
    selection_start = int(args.start_index)
    total = gold_total - selection_start
    if args.limit is not None:
        total = min(total, max(0, int(args.limit)))
    rollout_is_full = len(rollout_offsets) == gold_total
    if total <= 0 or (
        rollout_is_full and len(rollout_offsets) < selection_start + total
    ) or (not rollout_is_full and len(rollout_offsets) < total):
        raise ValueError("V1 teacher selection is empty or outside rollouts")
    local_start, local_stop = partition_bounds(total, args.rank, args.world_size)
    start = selection_start + local_start
    stop = selection_start + local_stop
    if start == stop:
        raise ValueError("V1 teacher worker partition is empty")

    teacher = V1ASRTeacher(
        args.checkpoint.resolve(),
        args.hf_model.resolve(),
        args.whispervq_model.resolve(),
        torch.device(args.device),
        topk=args.topk,
        temperature=args.temperature,
    )
    manifest = output_dir / "v1_teacher_cache.jsonl"
    temporary_manifest = output_dir / f".v1_teacher_cache.tmp.{os.getpid()}.jsonl"
    rows_pending: list[dict[str, object]] = []
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
        saved_rows = save_v1_bundle(path, rows_pending)
        bundle_sha256 = file_sha256(path)
        for row in saved_rows:
            row["bundle_sha256"] = bundle_sha256
            encoded = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            byte_offsets.append(byte_offset)
            handle.write(encoded)
            byte_offset += len(encoded)
        rows_pending = []
        bundle_index += 1

    with args.gold.open("rb") as gold_handle, args.rollouts.open(
        "rb"
    ) as rollout_handle, temporary_manifest.open("wb") as manifest_handle:
        for record_index in range(start, stop):
            gold_handle.seek(int(gold_offsets[record_index]))
            trajectory = E2ETrajectory.from_mapping(
                json.loads(gold_handle.readline())
            )
            rollout_base = 0 if rollout_is_full else selection_start
            rollout_ordinal = record_index - rollout_base
            if not 0 <= rollout_ordinal < len(rollout_offsets):
                raise ValueError("V1 teacher rollout selection does not cover gold")
            rollout_handle.seek(int(rollout_offsets[rollout_ordinal]))
            rollout = V1Rollout.from_mapping(json.loads(rollout_handle.readline()))
            _audit_pair(trajectory, rollout)
            sequences = build_v1_teacher_sequences(
                trajectory,
                rollout,
                encode_text=lambda text: teacher.tokenizer.encode(
                    text, add_special_tokens=False
                ),
            )
            speech = teacher.speech_embeddings(trajectory)
            summaries = teacher.summarize(sequences, speech)
            arrays, descriptors = combine_v1_sample(sequences, summaries)
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
            counts["requests"] += len(descriptors)
            counts["teacher_positions"] += len(arrays["reference_label"])
            counts["teacher_top1_correct"] += int(
                np.count_nonzero(arrays["top1"] == arrays["reference_label"])
            )
            counts["reference_in_topk"] += int(
                np.count_nonzero(
                    (
                        arrays["indices"]
                        == arrays["reference_label"][:, None]
                    ).any(axis=1)
                )
            )
            for descriptor in descriptors:
                counts[f"history:{descriptor['history_kind']}"] += 1
                counts["final_requests"] += int(bool(descriptor["final"]))
            if len(rows_pending) >= args.records_per_bundle:
                write_rows(manifest_handle)
            if args.progress_interval and counts["records"] % args.progress_interval == 0:
                elapsed = max(1e-9, time.perf_counter() - started)
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "records": counts["records"],
                            "assigned_records": stop - start,
                            "records_per_second": counts["records"] / elapsed,
                            "teacher_positions": counts["teacher_positions"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        write_rows(manifest_handle)
        manifest_handle.flush()
        os.fsync(manifest_handle.fileno())
    os.replace(temporary_manifest, manifest)
    index = write_index(manifest, byte_offsets)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": V1_PART_SCHEMA,
        "cache_schema": V1_CACHE_SCHEMA,
        "status": "complete",
        "rank": args.rank,
        "world_size": args.world_size,
        "selection_start": selection_start,
        "selection_stop": selection_start + total,
        "assigned_start": start,
        "assigned_stop": stop,
        "gold": str(args.gold.resolve()),
        "rollouts": str(args.rollouts.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "hf_model": str(args.hf_model.resolve()),
        "whispervq_model": str(args.whispervq_model.resolve()),
        "v1_hf_sha256": args.v1_hf_sha256,
        "runtime_sha256": runtime_sha256(),
        "topk": args.topk,
        "temperature": args.temperature,
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": elapsed,
        "records_per_second": counts["records"] / max(1e-9, elapsed),
        "manifest": str(manifest.resolve()),
        "manifest_bytes": manifest.stat().st_size,
        "manifest_sha256": file_sha256(manifest),
        "index": index,
    }
    atomic_json(output_dir / "PART_COMPLETE.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
