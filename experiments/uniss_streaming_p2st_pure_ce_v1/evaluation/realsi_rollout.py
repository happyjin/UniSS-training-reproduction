#!/usr/bin/env python3
"""Run the p2st cascade over a public corpus and emit what the scorers need.

A sibling of ``timeline_demos.py``, not an edit to it: that file produced every
published PANEL64/FIXED16/FINAL number and has to keep doing so byte for byte.
It is imported here for ``decode_fragments`` so the two share one decode path.

Three things this adds:

*   a **mono** translation wav.  ``evaluation/asr_transcribe.load_audio_array``
    means over channels, so handing it the stereo listening demo would mix the
    source speech into the ASR input and inflate ASR-BLEU.  Both the placed
    stream (silences where the system had nothing ready, SimulEval's own
    convention) and the back-to-back concatenation are written, so the
    sensitivity of ASR to inserted silence is measurable rather than assumed.
*   per-fragment ``delays`` / ``durations`` / ``intervals`` / ``elapsed``, which
    are the four lists SimulEval's speech latency scorers consume.
*   ``--read-stride``, and the MT delta character lengths per read step, because
    the length prior buckets text at 24 characters and a large stride commits
    more text per step -- if the deltas run past 24 the prior is saturating and
    the arm's numbers mean something different.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_e2e_speak_decision_v1.evaluation.timeline_stereo import (
    place_on_timeline,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (  # noqa: E501
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    make_cached_frontend,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.data.public_corpus import load_selection
from experiments.uniss_streaming_p2st_pure_ce_v1.evaluation.timeline_demos import (
    SAMPLE_RATE,
    decode_fragments,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    P2STCascadeSession,
    SEMANTIC_MS_PER_TOKEN,
)
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer


def write_mono(path: Path, audio: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
    return len(audio) / SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate-hf", required=True)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--bicodec-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arm", required=True, help="name for this operating point")
    parser.add_argument("--read-stride", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-semantic-tokens", type=int, default=512)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--length-prior-scale", type=float, default=1.0)
    parser.add_argument("--keep-stereo", action="store_true")
    args = parser.parse_args()

    rows = load_selection(args.selection)
    if args.num_shards > 1:
        rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
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
    codec = BiCodecTokenizer(args.bicodec_model, device=device)
    codec.model.eval().requires_grad_(False)

    out = Path(args.output_root) / args.arm
    manifest: list[dict] = []
    for position, row in enumerate(rows):
        waveform, rate = sf.read(row.source_audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if int(rate) != SAMPLE_RATE:
            raise ValueError(f"{row.source_audio} is {rate} Hz")
        started = time.perf_counter()
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
            read_stride=args.read_stride,
        )
        trace = session.run(waveform)
        wall = time.perf_counter() - started
        speech = [f for f in trace.fragments if f.semantic]
        mono = decode_fragments(
            codec, row.speaker_global, [tuple(f.semantic) for f in speech]
        )
        schedule = [(int(f.source_end_ms), len(f.semantic)) for f in speech]
        placed, placement = place_on_timeline(mono, schedule, len(waveform))

        placed_seconds = write_mono(
            out / "translation_placed" / f"{row.sample_id}.wav", placed
        )
        concat_seconds = write_mono(
            out / "translation_concat" / f"{row.sample_id}.wav", mono
        )
        if args.keep_stereo:
            total = max(len(waveform), len(placed))
            stereo = np.zeros((total, 2), dtype=np.float32)
            stereo[: len(waveform), 0] = waveform
            stereo[: len(placed), 1] = placed
            path = out / "stereo" / f"{row.sample_id}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), stereo, SAMPLE_RATE, subtype="PCM_16")

        # SimulEval's speech-output lists: one entry per emitted chunk.
        delays = [float(f.source_end_ms) for f in speech]
        durations = [SEMANTIC_MS_PER_TOKEN * len(f.semantic) for f in speech]
        intervals = [
            [float(f.start_ms), SEMANTIC_MS_PER_TOKEN * len(f.semantic)] for f in speech
        ]
        elapsed = [float(f.elapsed_ms) for f in speech]
        silences: list[float] = []
        previous_end = 0.0
        for fragment in speech:
            silences.append(max(0.0, float(fragment.start_ms) - previous_end))
            previous_end = float(fragment.end_ms)

        tts_stages = [s for s in trace.stages if s.task == "tts"]
        mt_delta_chars: list[int] = []
        previous_text = ""
        for fragment in speech:
            mt_delta_chars.append(max(0, len(fragment.text) - len(previous_text)))
            previous_text = fragment.text
        manifest.append(
            {
                "sample_id": row.sample_id,
                "arm": args.arm,
                "read_stride": args.read_stride,
                "read_step_ms": args.read_stride * 160,
                "direction": row.direction,
                "src_lang": row.src_lang,
                "tgt_lang": row.tgt_lang,
                "source_duration_ms": row.source_duration_ms,
                "read_steps": trace.blocks,
                "audio_blocks": trace.audio_blocks,
                "fragments": len(speech),
                "semantic_tokens": sum(len(f.semantic) for f in speech),
                "translation_placed": str(out / "translation_placed" / f"{row.sample_id}.wav"),
                "translation_concat": str(out / "translation_concat" / f"{row.sample_id}.wav"),
                "placed_seconds": placed_seconds,
                "concat_seconds": concat_seconds,
                "delays": delays,
                "durations": durations,
                "intervals": intervals,
                "elapsed": elapsed,
                "silences": silences,
                "source_hypothesis": trace.source_text,
                "target_hypothesis": trace.target_text,
                "transcription_reference": row.reference_transcription,
                "translation_reference": row.reference_translation,
                "terminator_rate": (
                    sum(1 for s in tts_stages if s.stopped_on_terminator)
                    / max(1, len(tts_stages))
                ),
                "capped_stages": sum(
                    1 for s in tts_stages if not s.stopped_on_terminator
                ),
                # The length prior buckets text at 24 characters, so a delta
                # longer than that lands in the widest bucket -- median 374
                # codes, support to 1328 -- and END is heavily suppressed.  A
                # large read stride commits more text per step, so this is the
                # number that says whether an arm's result is the stride or the
                # prior saturating.  Fragment.text is the accumulated committed
                # target, so successive differences give the per-step delta.
                "mt_delta_chars": mt_delta_chars,
                "mt_delta_chars_over_prior_bucket": sum(
                    1 for value in mt_delta_chars if value > 24
                ),
                "revision_conflicts": {
                    "source": session.source_committer.revision_conflicts,
                    "target": session.target_committer.revision_conflicts,
                },
                "placement": placement,
                "wall_seconds": wall,
                "rtf": wall / max(row.source_duration_ms / 1000.0, 1e-9),
            }
        )
        if position % 20 == 0:
            print(
                f"  [{args.arm} shard{args.shard_index}] {position + 1}/{len(rows)} "
                f"{row.sample_id} steps={trace.blocks} frags={len(speech)} "
                f"sem={sum(len(f.semantic) for f in speech)} rtf={manifest[-1]['rtf']:.2f}",
                flush=True,
            )

    target = out / f"MANIFEST_g{args.shard_index}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"arm": args.arm, "samples": manifest}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"shard {args.shard_index}: {len(manifest)} rows -> {target}")


if __name__ == "__main__":
    main()
