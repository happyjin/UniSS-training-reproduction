#!/usr/bin/env python3
"""Stereo listening demos for the p2st cascade, on the true emission timeline.

Left channel is the source, right channel is the translation, and every spoken
fragment is written where it could actually first be heard -- at the later of
its event's ``source_end_ms`` and the end of the previous fragment, because one
speaker cannot play two fragments at once.  Gaps where the system had nothing
ready stay silent, which is what a listener hears.

This matters because the m3 run's own report
(``longform_delta5/TIMELINE_RENDERING_BUG.zh-CN.md``) found that
``build_stereo_demos`` wrote the translation into ``stereo[:len(x), 1]`` from
sample zero after concatenating every fragment back to back, discarding the
emission schedule.  On one sample that was a 6.2 second error and made the
translation appear to run ahead of the source.  Only m3's ``audio_timeline/``
directories are honest, so those are the ones this run is compared against.

The cascade already computes each fragment's placement, and the placement rule
here is imported from the speak-decision experiment's renderer so the two runs
are laid out by the same code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from experiments.uniss_phase3_e2e_speak_decision_v1.evaluation.timeline_stereo import (
    place_on_timeline,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (  # noqa: E501
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    make_cached_frontend,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.runtime.p2st_cascade import (
    BLOCK_SAMPLES,
    P2STCascadeSession,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from uniss.streaming.bicodec_streamer import (
    StreamingBiCodecDecoder,
    bicodec_decode_function,
)

SAMPLE_RATE = 16000


def decode_fragments(codec, speaker_global, fragments) -> np.ndarray:
    """One continuous stream, exactly as run_worker._decode_semantic_fragments."""
    streamer = StreamingBiCodecDecoder(bicodec_decode_function(codec))
    streamer.set_speaker_tokens(speaker_global)
    chunks: list[np.ndarray] = []
    for fragment in fragments:
        if fragment:
            chunk = streamer.push(fragment, is_final=False)
            if len(chunk):
                chunks.append(chunk)
    if fragments:
        tail = streamer.push((), is_final=True)
        if len(tail):
            chunks.append(tail)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--candidate-hf", required=True)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--whispervq-model", required=True)
    parser.add_argument("--bicodec-model", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tts-text-scope", default="delta")
    parser.add_argument("--max-semantic-tokens", type=int, default=384)
    parser.add_argument("--semantic-penalty", type=float, default=1.1)
    # The default of 8 in p2st_cascade is too narrow.  Measured on
    # emilia_zh_0006795452: the TTS stage enters a repeating code loop whose
    # period is 28, so a window of 8 cannot see it; the stage runs to the 384
    # cap and BiCodec decodes the loop to silence (peak 0.0020) while gold
    # codes through the same decoder and the same speaker tokens give peak
    # 0.9949.  At window 32 it terminates on its own at 194 codes (peak
    # 0.8950) and at 64 at 177 codes (peak 0.9900) against gold's 163.  The m3
    # run had already found this -- its report directory is named
    # audio_timeline_rp11w64.
    parser.add_argument("--semantic-penalty-window", type=int, default=64)
    parser.add_argument("--holdback", type=int, default=2)
    parser.add_argument("--label", default="p2st")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-pace", action="store_true")
    parser.add_argument("--pace-margin-ms", type=float, default=2000.0)
    parser.add_argument("--length-prior-scale", type=float, default=1.0)
    parser.add_argument("--pace-tail-ms", type=float, default=2000.0)
    args = parser.parse_args()

    wanted = set(args.sample_ids)
    rows: list[dict] = []
    with open(args.gold) as handle:
        for line in handle:
            if '"sample_id"' not in line:
                continue
            value = json.loads(line)
            if value.get("sample_id") in wanted:
                rows.append(value)
    missing = wanted - {r["sample_id"] for r in rows}
    if missing:
        raise SystemExit(f"sample ids not in {args.gold}: {sorted(missing)}")

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

    audio_dir = Path(args.output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for value in sorted(rows, key=lambda r: r["sample_id"]):
        trajectory = E2ETrajectory.from_mapping(value)
        waveform, rate = sf.read(trajectory.source_audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform[:, 0]
        if int(rate) != SAMPLE_RATE:
            raise ValueError("source audio must be 16 kHz")
        blocks = max(1, (len(waveform) + BLOCK_SAMPLES - 1) // BLOCK_SAMPLES)
        session = P2STCascadeSession(
            model=model,
            tokenizer=tokenizer,
            objective=objective,
            frontend=frontend,
            src_lang=trajectory.src_lang,
            tgt_lang=trajectory.tgt_lang,
            speaker_global=trajectory.speaker_global,
            holdback=args.holdback,
            max_semantic_tokens=args.max_semantic_tokens,
            semantic_penalty=args.semantic_penalty,
            semantic_penalty_window=args.semantic_penalty_window,
            tts_text_scope=args.tts_text_scope,
            pace=not args.no_pace,
            pace_margin_ms=args.pace_margin_ms,
            pace_tail_ms=args.pace_tail_ms,
            length_prior_scale=args.length_prior_scale,
        )
        trace = session.run(waveform, max_blocks=blocks)
        speech = [f for f in trace.fragments if f.semantic]
        mono = decode_fragments(
            codec, trajectory.speaker_global, [tuple(f.semantic) for f in speech]
        )
        schedule = [(int(f.source_end_ms), len(f.semantic)) for f in speech]
        placed, stats = place_on_timeline(mono, schedule, len(waveform))
        total = max(len(waveform), len(placed))
        stereo = np.zeros((total, 2), dtype=np.float32)
        stereo[: len(waveform), 0] = waveform
        stereo[: len(placed), 1] = placed
        name = f"{trajectory.sample_id}__{args.label}__stereo.wav"
        sf.write(str(audio_dir / name), stereo, SAMPLE_RATE)
        timeline = [
            {
                "fragment": index,
                "block_index": int(f.block_index),
                "source_end_ms": int(f.source_end_ms),
                "placed_start_ms": round(f.start_ms, 1),
                "placed_end_ms": round(f.end_ms, 1),
                "semantic_tokens": len(f.semantic),
                "text": f.text,
            }
            for index, f in enumerate(speech)
        ]
        (audio_dir.parent / f"TIMELINE_{trajectory.sample_id}.json").write_text(
            json.dumps(timeline, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "sample_id": trajectory.sample_id,
                "direction": f"{trajectory.src_lang}->{trajectory.tgt_lang}",
                "audio": str(audio_dir / name),
                "source_duration_ms": int(trajectory.source_duration_ms),
                "blocks": blocks,
                "fragments": len(speech),
                "semantic_tokens": sum(len(f.semantic) for f in speech),
                "first_audible_ms": round(speech[0].start_ms, 1) if speech else None,
                "last_audible_ms": round(speech[-1].end_ms, 1) if speech else None,
                "terminator_rate": (
                    sum(1 for s in trace.stages if s.task == "tts" and s.stopped_on_terminator)
                    / max(1, sum(1 for s in trace.stages if s.task == "tts"))
                ),
                "source_hypothesis": trace.source_text,
                "target_hypothesis": trace.target_text,
                "transcription_reference": trajectory.full_transcription,
                "translation_reference": trajectory.full_translation,
                "placement": stats,
                "pace_budgets": session.pace_budgets,
                "timeline": timeline,
            }
        )
        print(
            f"  {trajectory.sample_id:<26} src={trajectory.source_duration_ms:>6}ms "
            f"frags={len(speech):>2d} sem={sum(len(f.semantic) for f in speech):>4d} "
            f"first={manifest[-1]['first_audible_ms']} "
            f"queue_max={stats['queueing_delay_ms_max']:.0f}ms",
            flush=True,
        )
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(
        json.dumps({"label": args.label, "samples": manifest}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print("manifest=", args.manifest)


if __name__ == "__main__":
    main()
