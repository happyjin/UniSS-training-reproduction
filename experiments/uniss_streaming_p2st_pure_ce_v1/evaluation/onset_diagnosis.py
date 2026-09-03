#!/usr/bin/env python3
"""Decompose C's first-audible delay into the stage that actually causes it.

C's onset on the internal panel is 2395 ms and on RealSI 2544/2573 ms, while
m3 at delta=5 reaches 418-830 ms.  Before proposing a fix, the delay has to be
attributed, and the cascade already records everything needed: ``source_deltas``
and ``target_deltas`` carry one committed-character count per read step, and
``Fragment.read_step`` says which step produced each spoken fragment.  So three
onsets are directly observable per utterance, all in units of 160 ms steps:

* **ASR onset** -- the first step whose ``source_deltas`` entry is non-zero,
  i.e. when the committer first releases *any* source text.  Bounded below by
  the committer's warm-up: ``StablePrefixCommitter`` needs a previous
  hypothesis to intersect with, so step 0 always commits nothing, and holdback
  h then trims h more characters off the agreed prefix.
* **MT onset** -- the first non-zero ``target_deltas`` entry.  MT cannot run on
  an empty source prefix, so this is at least the ASR onset.
* **speech onset** -- ``fragments[0].read_step``, which is what SimulEval's
  StartOffset measures.

Sweeping ``--holdback`` over the same utterances separates "the committer is
withholding text it already has" from "the ASR has not produced stable text
yet".  If onset barely moves with holdback, the delay is the acoustic model
needing more audio, and no commit-policy change can fix it -- the fix has to be
on the model or on the decision to speak before the text is stable, which is
exactly the trade m3's delta=5 makes (and pays for with WER 1.708).

Writes one JSONL row per utterance per holdback.  No existing file is touched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    BLOCK_MS,
    P2STCascadeSession,
)

SAMPLE_RATE = 16_000


def first_nonzero(values: list[int]) -> int | None:
    for index, value in enumerate(values):
        if value:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate-hf", required=True)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdback", type=int, action="append", default=None)
    # The decomposition says the onset is two agreement committers in series --
    # ASR at 960 ms then MT at a further 1120 ms -- and that the source prefix
    # is never revised (0 conflicts at every holdback).  So the MT committer's
    # own stability check may be redundant, which is what "s1t0" tests: keep the
    # source conservative, let the target commit immediately.  Pass pairs as
    # "source:target".
    parser.add_argument("--asymmetric", action="append", default=None,
                        help='"source:target" holdback pairs, e.g. 1:0')
    parser.add_argument("--read-stride", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    pairs: list[tuple[int, int]] = [(h, h) for h in (args.holdback or [])]
    for value in args.asymmetric or []:
        left, _, right = value.partition(":")
        pairs.append((int(left), int(right)))
    if not pairs:
        pairs = [(1, 1)]

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

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows):
            waveform, rate = sf.read(row.source_audio, dtype="float32")
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if int(rate) != SAMPLE_RATE:
                raise ValueError(f"{row.source_audio} is {rate} Hz")
            for source_holdback, target_holdback in pairs:
                started = time.perf_counter()
                session = P2STCascadeSession(
                    model=model,
                    tokenizer=tokenizer,
                    objective=objective,
                    frontend=frontend,
                    src_lang=row.src_lang,
                    tgt_lang=row.tgt_lang,
                    speaker_global=row.speaker_global,
                    source_holdback=source_holdback,
                    target_holdback=target_holdback,
                    read_stride=args.read_stride,
                )
                trace = session.run(waveform)
                speech = [f for f in trace.fragments if f.semantic]
                asr_step = first_nonzero(trace.source_deltas)
                mt_step = first_nonzero(trace.target_deltas)
                step_ms = args.read_stride * BLOCK_MS
                handle.write(
                    json.dumps(
                        {
                            "sample_id": row.sample_id,
                            "direction": row.direction,
                            "holdback": (
                                source_holdback
                                if source_holdback == target_holdback
                                else f"s{source_holdback}t{target_holdback}"
                            ),
                            "source_holdback": source_holdback,
                            "target_holdback": target_holdback,
                            "source_duration_ms": row.source_duration_ms,
                            "read_steps": trace.blocks,
                            "asr_onset_step": asr_step,
                            "mt_onset_step": mt_step,
                            "speech_onset_step": speech[0].read_step if speech else None,
                            "asr_onset_ms": None if asr_step is None else (asr_step + 1) * step_ms,
                            "mt_onset_ms": None if mt_step is None else (mt_step + 1) * step_ms,
                            "speech_onset_ms": speech[0].source_end_ms if speech else None,
                            "fragments": len(speech),
                            "semantic_tokens": sum(len(f.semantic) for f in speech),
                            "source_chars": len(trace.source_text),
                            "target_chars": len(trace.target_text),
                            "reference_chars": len(row.reference_translation),
                            "source_hypothesis": trace.source_text,
                            "target_hypothesis": trace.target_text,
                            "transcription_reference": row.reference_transcription,
                            "translation_reference": row.reference_translation,
                            "revision_conflicts": {
                                "source": session.source_committer.revision_conflicts,
                                "target": session.target_committer.revision_conflicts,
                            },
                            "wall_seconds": time.perf_counter() - started,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            if position % 10 == 0:
                print(f"  [onset shard{args.shard_index}] {position + 1}/{len(rows)}", flush=True)
    print(f"shard {args.shard_index}: {written} rows -> {out}")


if __name__ == "__main__":
    main()
