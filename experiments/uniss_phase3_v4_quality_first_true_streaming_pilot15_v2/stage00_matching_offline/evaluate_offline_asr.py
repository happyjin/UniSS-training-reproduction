#!/usr/bin/env python3
"""Evaluate Phase3 Quality-ASR and stop after the first content region."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from training import constants_uniss as c
from training.generate_unist_eval_audio import (
    build_eval_sample,
    iter_manifest_records,
    load_hf_text_encoder,
)
from evaluation.io_utils import iter_jsonl


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matching offline output: {path}")
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


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite matching offline run: {args.output_dir}")
    if args.max_new_tokens <= 0 or args.repetition_penalty <= 0:
        raise ValueError("invalid matching offline decoding settings")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    text_encoder = load_hf_text_encoder(tokenizer)
    suppressed = list(range(c.VOCAB_SIZE, int(model.config.vocab_size)))
    limit_records = args.limit_records if args.limit_records > 0 else 0
    manifest_rows = list(iter_jsonl(args.manifest))
    if limit_records:
        manifest_rows = manifest_rows[:limit_records]
    source_records = list(iter_manifest_records(args.manifest, limit_records=limit_records))
    if len(manifest_rows) != len(source_records):
        raise ValueError("matching manifest and loaded source record counts differ")
    rows: list[dict[str, object]] = []
    for index, (manifest_row, record) in enumerate(zip(manifest_rows, source_records, strict=True)):
        if str(manifest_row["id"]) != str(record["id"]):
            raise ValueError(f"matching manifest/source order differs at row {index}")
        sample = build_eval_sample(record, mode="quality", text_encoder=text_encoder)
        prompt = torch.tensor([sample.prompt_ids], dtype=torch.long, device=device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=c.TOKEN_PAD,
                eos_token_id=c.TOKEN_END_CONTENT,
                suppress_tokens=suppressed,
            )
        elapsed = time.perf_counter() - started
        tail = generated[0, prompt.shape[1] :].tolist()
        reached_stop = bool(tail and int(tail[-1]) == c.TOKEN_END_CONTENT)
        content = tail[:-1] if reached_stop else tail
        rows.append(
            {
                "index": index,
                "id": str(record["id"]),
                "task": str(manifest_row["task"]),
                "src_lang": str(record["src_lang"]),
                "tgt_lang": str(record["tgt_lang"]),
                "transcription_ref": str(manifest_row["transcription"]),
                "source_transcription": str(record["transcription"]),
                "generated_transcription": tokenizer.decode(
                    content, skip_special_tokens=False
                ).strip(),
                "generated_token_ids": [int(value) for value in tail],
                "reached_end_content": reached_stop,
                "generation_seconds": elapsed,
                "model": str(args.model.resolve()),
            }
        )
    results = args.output_dir / "results.jsonl"
    with results.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    summary = {
        "schema_version": "uniss_quality_first_matching_offline_asr_worker_v1",
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "records": len(rows),
        "reached_end_content": sum(bool(row["reached_end_content"]) for row in rows),
        "generation_seconds": sum(float(row["generation_seconds"]) for row in rows),
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "dtype": args.dtype,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
