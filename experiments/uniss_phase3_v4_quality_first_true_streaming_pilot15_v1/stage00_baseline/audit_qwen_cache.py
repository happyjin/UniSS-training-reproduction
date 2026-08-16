#!/usr/bin/env python3
"""Audit HF Qwen full-forward versus DynamicCache on fixed Phase3 prompts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from training.generate_unist_eval_audio import build_eval_sample, iter_manifest_records, load_hf_text_encoder


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Qwen parity audit: {path}")
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


@torch.inference_mode()
def _parity(model, ids: torch.Tensor, chunk_sizes: list[int]) -> dict[str, object]:
    full = model(input_ids=ids, use_cache=False, return_dict=True).logits.float()
    cache = DynamicCache()
    pieces: list[torch.Tensor] = []
    boundaries: list[dict[str, object]] = []
    start = 0
    chunk_index = 0
    while start < ids.shape[1]:
        size = chunk_sizes[chunk_index % len(chunk_sizes)]
        end = min(ids.shape[1], start + size)
        output = model(
            input_ids=ids[:, start:end],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        pieces.append(output.logits.float())
        recomputed_last = model(
            input_ids=ids[:, :end], use_cache=False, return_dict=True
        ).logits[:, -1].float()
        cached_last = output.logits[:, -1].float()
        cosine = F.cosine_similarity(recomputed_last, cached_last, dim=-1)
        boundaries.append(
            {
                "prefix_tokens": end,
                "full_top1": int(recomputed_last.argmax(dim=-1).item()),
                "cached_top1": int(cached_last.argmax(dim=-1).item()),
                "top1_exact": bool(
                    torch.equal(
                        recomputed_last.argmax(dim=-1), cached_last.argmax(dim=-1)
                    )
                ),
                "logits_cosine": float(cosine.item()),
                "maximum_absolute_logit_error": float(
                    (recomputed_last - cached_last).abs().max().item()
                ),
            }
        )
        start = end
        chunk_index += 1
    cached = torch.cat(pieces, dim=1)
    full_top1 = full.argmax(dim=-1)
    cached_top1 = cached.argmax(dim=-1)
    matches = int((full_top1 == cached_top1).sum().item())
    positions = int(full_top1.numel())
    cosine = F.cosine_similarity(full, cached, dim=-1)
    difference = (full - cached).abs()
    return {
        "boundaries": boundaries,
        "boundary_top1_exact": all(value["top1_exact"] for value in boundaries),
        "minimum_boundary_logits_cosine": min(
            value["logits_cosine"] for value in boundaries
        ),
        "cache_sequence_length": int(cache.get_seq_length()),
        "canonical_sequence_length": int(ids.shape[1]),
        "all_position_numerical_diagnostic": {
            "gate": False,
            "positions": positions,
            "matching_top1": matches,
            "top1_match_ratio": matches / positions,
            "top1_exact": matches == positions,
            "minimum_logits_cosine": float(cosine.min().item()),
            "mean_logits_cosine": float(cosine.mean().item()),
            "maximum_absolute_logit_error": float(difference.max().item()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    manifest = Path(args.validation_manifest).resolve()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    record = next(iter_manifest_records(manifest, limit_records=1))
    encode = load_hf_text_encoder(tokenizer)
    modes = ("quality", "performance", "direct_s2st", "tts")
    results = {}
    for mode in modes:
        sample = build_eval_sample(record, mode=mode, text_encoder=encode)
        ids = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
        results[mode] = _parity(model, ids, [1, 3, 17, 8, 31])
    checks = {
        "all_append_boundary_top1_exact": all(
            value["boundary_top1_exact"] for value in results.values()
        ),
        "all_append_boundary_logits_cosine_ge_0p9999": all(
            value["minimum_boundary_logits_cosine"] >= 0.9999
            for value in results.values()
        ),
        "all_cache_lengths_exact": all(
            value["cache_sequence_length"] == value["canonical_sequence_length"]
            for value in results.values()
        ),
    }
    output = {
        "schema_version": "uniss_stage00_qwen_hf_cache_parity_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "model": str(model_path),
        "validation_manifest": str(manifest),
        "sample_id": record.get("id"),
        "dtype": str(dtype),
        "device": str(device),
        "chunk_sizes": [1, 3, 17, 8, 31],
        "modes": results,
    }
    _atomic_json(Path(args.output_json).resolve(), output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
