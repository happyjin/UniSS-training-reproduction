#!/usr/bin/env python3
"""Build immutable future-safe Phase3 top-k teacher bundles for Stage A v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.dataset import (
    rotated_acoustic_indices,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.same_prefix_teacher import (
    TeacherRequest,
    fixed_speaker_from_pack,
    requests_for_acoustic,
)
from training import constants_uniss as c
from training.simul_uniss.jsonl_index import load_index


CACHE_SCHEMA = "uniss_quality_first_same_prefix_teacher_cache_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if total < 0 or not 0 <= rank < world_size:
        raise ValueError("teacher cache partition geometry is invalid")
    per_rank = math.ceil(total / world_size) if total else 0
    start = min(total, rank * per_rank)
    return start, min(total, start + per_rank)


def selected_acoustic_indices(
    count: int,
    *,
    pack_index: int,
    coverage_epochs: int,
    max_acoustics_per_pack: int,
) -> list[int]:
    selected = {
        index
        for epoch in range(coverage_epochs)
        for index in rotated_acoustic_indices(
            count, max_acoustics_per_pack, epoch, pack_index
        )
    }
    return sorted(selected)


def _pad_sequences(
    sequences: Sequence[Sequence[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty teacher request batch")
    maximum = max(len(value) for value in sequences)
    ids = torch.full(
        (len(sequences), maximum), c.TOKEN_PAD, dtype=torch.long, device=device
    )
    attention = torch.zeros(
        (len(sequences), maximum), dtype=torch.long, device=device
    )
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention[row, : len(sequence)] = 1
    return ids, attention


class Phase3ASRTeacher:
    """Frozen Phase3 teacher evaluated on standalone same-prefix ASR prompts."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        *,
        topk: int,
        temperature: float,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if topk <= 0 or temperature <= 0:
            raise ValueError("teacher top-k and temperature must be positive")
        self.device = device
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def summarize(
        self, requests: Sequence[TeacherRequest]
    ) -> list[dict[str, np.ndarray]]:
        sequences = [
            [*request.prompt_ids, *request.target_ids] for request in requests
        ]
        ids, attention = _pad_sequences(sequences, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.model.model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        selected_hidden: list[torch.Tensor] = []
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
            selected_hidden.append(hidden[row].index_select(0, positions))
            lengths.append(len(positions))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.lm_head(torch.cat(selected_hidden, dim=0)).float()
        if logits.shape[-1] < c.VOCAB_SIZE:
            raise ValueError("Phase3 teacher vocabulary is smaller than UniSS")
        logical_logits = logits[:, : c.VOCAB_SIZE]
        raw_probability = F.softmax(logical_logits, dim=-1)
        confidence, top1 = raw_probability.max(dim=-1)
        scaled = logical_logits / self.temperature
        top_values, top_indices = torch.topk(
            scaled, min(self.topk, c.VOCAB_SIZE), dim=-1, sorted=True
        )
        probabilities = F.softmax(top_values, dim=-1)
        outputs: list[dict[str, np.ndarray]] = []
        cursor = 0
        for length in lengths:
            end = cursor + length
            outputs.append(
                {
                    "indices": top_indices[cursor:end].to(torch.int32).cpu().numpy(),
                    "probabilities": probabilities[cursor:end]
                    .to(torch.float16)
                    .cpu()
                    .numpy(),
                    "top1": top1[cursor:end].to(torch.int32).cpu().numpy(),
                    "confidence": confidence[cursor:end]
                    .to(torch.float16)
                    .cpu()
                    .numpy(),
                }
            )
            cursor = end
        return outputs


def combine_acoustic(
    requests: Sequence[TeacherRequest],
    summaries: Sequence[Mapping[str, np.ndarray]],
    *,
    require_reference_in_topk: bool,
    reference_anchor: float,
) -> tuple[dict[str, np.ndarray], int]:
    if len(requests) != len(summaries) or not requests:
        raise ValueError("teacher acoustic request/result count differs")
    positions = np.concatenate(
        [np.asarray(request.student_positions, dtype=np.int32) for request in requests]
    )
    labels = np.concatenate(
        [np.asarray(request.reference_labels, dtype=np.int32) for request in requests]
    )
    indices = np.concatenate([value["indices"] for value in summaries], axis=0)
    probabilities = np.concatenate(
        [value["probabilities"] for value in summaries], axis=0
    )
    top1 = np.concatenate([value["top1"] for value in summaries], axis=0)
    confidence = np.concatenate(
        [value["confidence"] for value in summaries], axis=0
    )
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    labels = labels[order]
    indices = indices[order]
    probabilities = probabilities[order]
    top1 = top1[order]
    confidence = confidence[order]
    candidate_count = len(positions)
    if not 0.0 <= reference_anchor <= 1.0:
        raise ValueError("teacher reference anchor must be in [0,1]")
    reference_in_topk = (indices == labels[:, None]).any(axis=1)
    if require_reference_in_topk:
        keep = reference_in_topk
        positions = positions[keep]
        labels = labels[keep]
        indices = indices[keep]
        probabilities = probabilities[keep]
        top1 = top1[keep]
        confidence = confidence[keep]
    matches = indices == labels[:, None]
    if len(positions) and not bool(matches.any(axis=1).all()):
        raise ValueError("selected teacher row is missing its reference token")
    probabilities = probabilities.astype(np.float32, copy=False)
    probabilities *= 1.0 - reference_anchor
    probabilities += matches.astype(np.float32) * reference_anchor
    probabilities = probabilities.astype(np.float16)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("teacher acoustic positions are duplicated")
    if indices.shape != probabilities.shape or indices.shape[0] != len(positions):
        raise ValueError("teacher acoustic top-k geometry differs")
    return (
        {
            "positions": positions,
            "labels": labels,
            "indices": indices,
            "probabilities": probabilities,
            "top1": top1,
            "confidence": confidence,
        },
        candidate_count,
    )


def _save_bundle(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    values: dict[str, np.ndarray] = {
        "bundle_schema": np.asarray([CACHE_SCHEMA]),
    }
    manifest: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        arrays = row["arrays"]
        if not isinstance(arrays, dict):
            raise TypeError("teacher bundle row arrays are missing")
        prefix = f"row_{row_index}"
        for name in (
            "positions",
            "labels",
            "indices",
            "probabilities",
            "top1",
            "confidence",
        ):
            values[f"{prefix}_{name}"] = arrays[name]
        manifest.append(
            {
                "schema_version": CACHE_SCHEMA,
                "pack_index": int(row["pack_index"]),
                "acoustic_index": int(row["acoustic_index"]),
                "sample_id": str(row["sample_id"]),
                "task": str(row["task"]),
                "bundle_path": str(path.resolve()),
                "bundle_row": row_index,
                "teacher_candidate_positions": int(row["candidate_positions"]),
                "teacher_positions": len(arrays["positions"]),
                "teacher_top1_correct": int(
                    np.count_nonzero(arrays["top1"] == arrays["labels"])
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)
    return manifest


def atomic_json(path: Path, value: object) -> None:
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


def build(args: argparse.Namespace) -> dict[str, object]:
    packs = args.packs.resolve()
    offsets = load_index(packs)
    if offsets is None:
        raise ValueError(f"missing Stage A pack index: {packs}")
    total = len(offsets)
    if args.limit_packs is not None:
        total = min(total, int(args.limit_packs))
    start, stop = partition(total, args.rank, args.world_size)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite teacher cache part: {output_dir}")
    output_dir.mkdir(parents=True)
    teacher = Phase3ASRTeacher(
        args.model.resolve(),
        torch.device(args.device),
        topk=args.topk,
        temperature=args.temperature,
    )
    manifest_path = output_dir / "teacher_cache.jsonl"
    temporary_manifest = output_dir / f".teacher_cache.tmp.{os.getpid()}.jsonl"
    pending: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    bundle_index = 0
    started = time.time()

    def flush(handle) -> None:
        nonlocal bundle_index
        if not pending:
            return
        path = output_dir / "bundles" / f"bundle-{bundle_index:06d}.npz"
        rows = _save_bundle(path, pending)
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        bundle_index += 1
        pending.clear()

    try:
        with packs.open("rb") as source, temporary_manifest.open(
            "w", encoding="utf-8"
        ) as manifest_handle:
            for pack_index in range(start, stop):
                source.seek(int(offsets[pack_index]))
                pack = json.loads(source.readline())
                acoustics = list(pack.get("acoustics", []))
                selected = selected_acoustic_indices(
                    len(acoustics),
                    pack_index=pack_index,
                    coverage_epochs=args.coverage_epochs,
                    max_acoustics_per_pack=args.max_acoustics_per_pack,
                )
                acoustic_requests: list[tuple[int, Mapping[str, object], list[TeacherRequest]]] = []
                requests: list[TeacherRequest] = []
                for acoustic_index in selected:
                    acoustic = acoustics[acoustic_index]
                    fixed_speaker = fixed_speaker_from_pack(pack, acoustic)
                    current = requests_for_acoustic(
                        pack,
                        acoustic,
                        fixed_speaker=fixed_speaker,
                        encode_text=lambda text: teacher.tokenizer.encode(
                            text, add_special_tokens=False
                        ),
                        decode_text=lambda ids: teacher.tokenizer.decode(
                            ids, skip_special_tokens=False
                        ),
                    )
                    acoustic_requests.append((acoustic_index, acoustic, current))
                    requests.extend(current)
                summaries = teacher.summarize(requests) if requests else []
                cursor = 0
                for acoustic_index, acoustic, current in acoustic_requests:
                    current_summaries = summaries[cursor : cursor + len(current)]
                    cursor += len(current)
                    arrays, candidate_count = combine_acoustic(
                        current,
                        current_summaries,
                        require_reference_in_topk=args.require_reference_in_topk,
                        reference_anchor=args.reference_anchor,
                    )
                    pending.append(
                        {
                            "pack_index": pack_index,
                            "acoustic_index": acoustic_index,
                            "sample_id": acoustic["sample_id"],
                            "task": acoustic["task"],
                            "candidate_positions": candidate_count,
                            "arrays": arrays,
                        }
                    )
                    counts["acoustics"] += 1
                    counts["teacher_candidate_positions"] += candidate_count
                    counts["teacher_positions"] += len(arrays["positions"])
                    counts["teacher_top1_correct"] += int(
                        np.count_nonzero(arrays["top1"] == arrays["labels"])
                    )
                    counts[f"task:{acoustic['task']}"] += 1
                    if len(pending) >= args.records_per_bundle:
                        flush(manifest_handle)
                if cursor != len(summaries):
                    raise AssertionError("teacher result cursor did not close")
                counts["packs"] += 1
                if args.progress_interval and counts["packs"] % args.progress_interval == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "rank": args.rank,
                                "packs": counts["packs"],
                                "assigned": stop - start,
                                "packs_per_second": counts["packs"] / elapsed,
                                "teacher_positions": counts["teacher_positions"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            flush(manifest_handle)
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    result = {
        "schema_version": CACHE_SCHEMA,
        "status": "complete",
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_start": start,
        "assigned_stop": stop,
        "assigned_packs": stop - start,
        "packs": str(packs),
        "packs_sha256": sha256(packs),
        "model": str(args.model.resolve()),
        "speaker_source": "stage_a_pack_prompt",
        "coverage_epochs": args.coverage_epochs,
        "max_acoustics_per_pack": args.max_acoustics_per_pack,
        "topk": args.topk,
        "temperature": args.temperature,
        "require_reference_in_topk": args.require_reference_in_topk,
        "reference_anchor": args.reference_anchor,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "bundles": bundle_index,
        "counts": dict(sorted(counts.items())),
        "teacher_top1_accuracy": counts["teacher_top1_correct"]
        / max(1, counts["teacher_candidate_positions"]),
        "teacher_selection_rate": counts["teacher_positions"]
        / max(1, counts["teacher_candidate_positions"]),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output_dir / "PART_COMPLETE.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coverage-epochs", type=int, default=3)
    parser.add_argument("--max-acoustics-per-pack", type=int, default=2)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument(
        "--require-reference-in-topk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--reference-anchor", type=float, default=0.5)
    parser.add_argument("--records-per-bundle", type=int, default=256)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--limit-packs", type=int)
    args = parser.parse_args()
    if (
        args.coverage_epochs <= 0
        or args.max_acoustics_per_pack <= 0
        or args.records_per_bundle <= 0
        or not 0.0 <= args.reference_anchor <= 1.0
    ):
        raise ValueError("teacher cache geometry must be positive")
    build(args)


if __name__ == "__main__":
    main()
