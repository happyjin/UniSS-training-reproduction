"""Dump the semantic codes the model actually generates, without decoding them.

Every silence measurement so far has been made on the decoded waveform, which
cannot separate "the model emitted silence codes" from "the vocoder produced a
quiet stretch".  The training pool can only be measured at the code level --
BiCodec maps digital silence to code 7645 -- so to compare the model against
its own training targets on one footing the model's codes have to be read
directly.

Skipping the vocoder also makes this cheap: no BiCodec decode, no wav writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (  # noqa: E501
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    make_cached_frontend,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.data.public_corpus import load_selection
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    P2STCascadeSession,
)

SAMPLE_RATE = 16000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate-hf", required=True)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--read-stride", type=int, default=4)
    parser.add_argument("--length-prior-scale", type=float, default=1.0)
    parser.add_argument("--min-fragment-tokens", type=int, default=0)
    parser.add_argument("--min-final-chunk-ms", type=int, default=0)
    parser.add_argument("--max-semantic-tokens", type=int, default=512)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--source-holdback", type=int, default=1)
    parser.add_argument("--target-holdback", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows = load_selection(args.selection)
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.candidate_hf, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.candidate_hf,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    objective = (
        stage_a_eval.load_objective(
            Path(args.v1_checkpoint), Path(args.whispervq_model), device
        )
        .eval()
        .requires_grad_(False)
    )
    frontend = make_cached_frontend(objective, device)

    out: list[dict] = []
    for position, row in enumerate(rows):
        waveform, rate = sf.read(row.source_audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{row.source_audio} is {rate} Hz")
        session = P2STCascadeSession(
            model=model,
            tokenizer=tokenizer,
            objective=objective,
            frontend=frontend,
            src_lang=row.src_lang,
            tgt_lang=row.tgt_lang,
            speaker_global=row.speaker_global,
            max_semantic_tokens=args.max_semantic_tokens,
            max_text_tokens=args.max_text_tokens,
            length_prior_scale=args.length_prior_scale,
            min_fragment_tokens=args.min_fragment_tokens,
            min_final_chunk_ms=args.min_final_chunk_ms,
            read_stride=args.read_stride,
            source_holdback=args.source_holdback,
            target_holdback=args.target_holdback,
        )
        trace = session.run(waveform)
        speech = [f for f in trace.fragments if f.semantic]
        out.append(
            {
                "sample_id": row.sample_id,
                "src_lang": row.src_lang,
                "tgt_lang": row.tgt_lang,
                "target_hypothesis": trace.target_text,
                # one list per fragment: silence runs must not be counted across
                # a fragment boundary, where the gap is a placement decision
                # rather than something the model generated.
                "fragments": [[int(c) for c in f.semantic] for f in speech],
            }
        )
        print(f"[{position + 1}/{len(rows)}] {row.sample_id} "
              f"frags={len(speech)} codes={sum(len(f.semantic) for f in speech)}",
              flush=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": out}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path} ({len(out)} samples)", flush=True)


if __name__ == "__main__":
    main()
