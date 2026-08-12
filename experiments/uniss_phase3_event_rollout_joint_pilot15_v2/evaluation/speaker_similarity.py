#!/usr/bin/env python3
"""Score generated audio against the fixed BiCodec speaker reference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from evaluation.io_utils import iter_jsonl, write_json
from evaluation.sharding import load_keys, select_shard
from training.generate_unist_eval_audio import write_jsonl_row


SCHEMA = "uniss_event_rollout_fixed_speaker_wavlm_similarity_v1"


def load_audio(path: Path, *, sample_rate: int = 16000) -> np.ndarray:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(value, dtype=np.float32).mean(axis=1)
    if rate != sample_rate:
        import librosa

        mono = librosa.resample(mono, orig_sr=rate, target_sr=sample_rate)
    return np.asarray(mono, dtype=np.float32)


def cosine_scores(reference: torch.Tensor, generated: torch.Tensor) -> list[float]:
    reference = F.normalize(reference.float().reshape(1, -1), dim=-1)
    generated = F.normalize(generated.float(), dim=-1)
    return [float(value) for value in (generated @ reference.T).reshape(-1).cpu()]


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[f"{row.get('mode')}:{row.get('src_lang')}->{row.get('tgt_lang')}"] .append(
            float(row["speaker_similarity"])
        )
    return {
        "schema_version": SCHEMA,
        "definition": (
            "Cosine similarity between WavLM speaker-verification x-vectors for each generated "
            "translation and the single target-audio reference whose BiCodec global speaker "
            "tokens condition every runtime sample. Range [-1,1], higher is more similar."
        ),
        "groups": {
            name: {
                "sample_count": len(values),
                "mean": fmean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in sorted(groups.items())
        },
        "scored_count": len(rows),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_sample_speaker_similarity.jsonl"
    completed = load_keys(args.completed_input)
    existing = []
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite speaker similarity: {output_path}")
        existing = list(iter_jsonl(output_path))
        completed.update((str(row["id"]), str(row["mode"])) for row in existing)
    rows = [
        row
        for row in select_shard(
            iter_jsonl(input_path),
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        if (str(row["id"]), str(row["mode"])) not in completed
        and row.get("audio_path")
        and row.get("fixed_speaker_reference_audio_path")
        and not row.get("error")
    ]
    device = torch.device(args.device)
    extractor = AutoFeatureExtractor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = WavLMForXVector.from_pretrained(
        args.model, local_files_only=args.local_files_only
    ).to(device).eval()
    reference_paths = {str(row["fixed_speaker_reference_audio_path"]) for row in rows}
    if len(reference_paths) > 1:
        raise ValueError(f"runtime results use multiple fixed speaker references: {reference_paths}")
    if not rows:
        report = aggregate(existing)
        write_json(output_dir / "speaker_similarity.json", report)
        return report
    reference_path = Path(next(iter(reference_paths))).resolve()

    def embeddings(audios: Sequence[np.ndarray]) -> torch.Tensor:
        inputs = extractor(
            list(audios), sampling_rate=16000, padding=True, return_tensors="pt"
        )
        values = inputs["input_values"].to(device)
        mask = inputs.get("attention_mask")
        with torch.inference_mode():
            result = model(
                values,
                attention_mask=None if mask is None else mask.to(device),
            ).embeddings
        return result.detach().cpu()

    reference_embedding = embeddings([load_audio(reference_path)])[0]
    scored = list(existing)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        generated = embeddings(
            [load_audio(Path(str(row["audio_path"])).resolve()) for row in batch]
        )
        for row, score in zip(batch, cosine_scores(reference_embedding, generated)):
            output_row = {
                "id": row["id"],
                "mode": row["mode"],
                "src_lang": row["src_lang"],
                "tgt_lang": row["tgt_lang"],
                "audio_path": row["audio_path"],
                "fixed_speaker_reference_audio_path": str(reference_path),
                "speaker_similarity": score,
                "speaker_model": args.model,
            }
            write_jsonl_row(output_path, output_row)
            scored.append(output_row)
        write_json(output_dir / "speaker_similarity.json", aggregate(scored))
    return aggregate(scored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--completed-input", action="append", default=[])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
