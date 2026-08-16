#!/usr/bin/env python3
"""Verify a fresh native->HF export against the canonical Phase3 HF model."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.generate_unist_eval_audio import build_eval_sample, iter_manifest_records, load_hf_text_encoder


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite native/HF audit: {path}")
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


def _tensor_files(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors files in {root}")
    return paths


def _tensor_map(root: Path) -> dict[str, Path]:
    result = {}
    for path in _tensor_files(root):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"duplicate tensor key {key}")
                result[key] = path
    return result


def _compare_tensors(canonical: Path, exported: Path) -> dict[str, object]:
    left = _tensor_map(canonical)
    right = _tensor_map(exported)
    common = sorted(set(left) & set(right))
    exact = 0
    maximum = 0.0
    for key in common:
        with safe_open(left[key], framework="pt", device="cpu") as first:
            left_tensor = first.get_tensor(key)
        with safe_open(right[key], framework="pt", device="cpu") as second:
            right_tensor = second.get_tensor(key)
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            continue
        difference = (left_tensor.float() - right_tensor.float()).abs()
        maximum = max(maximum, float(difference.max().item()))
        exact += int(torch.equal(left_tensor, right_tensor))
    return {
        "canonical_tensors": len(left),
        "reexported_tensors": len(right),
        "common_tensors": len(common),
        "missing_from_reexport": sorted(set(left) - set(right)),
        "extra_in_reexport": sorted(set(right) - set(left)),
        "exact_tensors": exact,
        "maximum_absolute_weight_error": maximum,
        "all_tensors_exact": exact == len(left) == len(right),
    }


@torch.inference_mode()
def _output_parity(
    canonical: Path, exported: Path, manifest: Path, device: torch.device
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(canonical, local_files_only=True)
    record = next(iter_manifest_records(manifest, limit_records=1))
    sample = build_eval_sample(
        record, mode="quality", text_encoder=load_hf_text_encoder(tokenizer)
    )
    ids = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
    models = []
    for path in (canonical, exported):
        models.append(
            AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            ).to(device).eval()
        )
    first = models[0](input_ids=ids, use_cache=False, return_dict=True).logits[:, -1]
    second = models[1](input_ids=ids, use_cache=False, return_dict=True).logits[:, -1]
    cosine = F.cosine_similarity(first.float(), second.float(), dim=-1)
    return {
        "sample_id": record.get("id"),
        "prompt_tokens": int(ids.shape[1]),
        "canonical_top1": int(first.argmax(dim=-1).item()),
        "reexported_top1": int(second.argmax(dim=-1).item()),
        "top1_exact": bool(
            torch.equal(first.argmax(dim=-1), second.argmax(dim=-1))
        ),
        "logits_cosine": float(cosine.item()),
        "maximum_absolute_logit_error": float((first - second).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-hf", required=True)
    parser.add_argument("--reexported-hf", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    canonical = Path(args.canonical_hf).resolve()
    exported = Path(args.reexported_hf).resolve()
    weights = _compare_tensors(canonical, exported)
    outputs = _output_parity(
        canonical,
        exported,
        Path(args.validation_manifest).resolve(),
        torch.device(args.device),
    )
    checks = {
        "all_exported_tensors_exact": weights["all_tensors_exact"],
        "fixed_prompt_top1_exact": outputs["top1_exact"],
        "fixed_prompt_logits_cosine_ge_0p999999": outputs["logits_cosine"]
        >= 0.999999,
    }
    result = {
        "schema_version": "uniss_stage00_native_hf_reexport_parity_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "canonical_hf": str(canonical),
        "reexported_hf": str(exported),
        "weights": weights,
        "fixed_prompt_output": outputs,
    }
    _atomic_json(Path(args.output_json).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

