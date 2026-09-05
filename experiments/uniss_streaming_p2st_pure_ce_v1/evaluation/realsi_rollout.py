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
import os
import tempfile


def out_silence_dir() -> str:
    return os.environ.get("TMPDIR") or tempfile.gettempdir()


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
    # 1 keeps the greedy path this lineage was measured on.  SimulS2ST-Omni
    # uses 4 on its text stage (paper section 4.1).
    parser.add_argument("--text-num-beams", type=int, default=1)
    parser.add_argument("--text-length-penalty", type=float, default=1.0)
    parser.add_argument("--text-penalty", type=float, default=1.0)
    parser.add_argument("--text-penalty-window", type=int, default=0)
    # 16 codes is 320 ms at 20 ms per code.
    parser.add_argument("--min-fragment-tokens", type=int, default=0)
    # SimulS2ST-Omni gates the source tail at 320 ms.
    parser.add_argument("--min-final-chunk-ms", type=int, default=0)
    # SimulS2ST-Omni's talker: top_p 0.8, top_k 20, temperature 1.0, rep 1.4.
    # temperature 0 keeps the greedy path.
    parser.add_argument("--semantic-temperature", type=float, default=0.0)
    parser.add_argument("--semantic-top-k", type=int, default=0)
    parser.add_argument("--semantic-top-p", type=float, default=1.0)
    parser.add_argument("--semantic-penalty", type=float, default=None)
    parser.add_argument("--semantic-penalty-window", type=int, default=None)
    # Defaults are None so not passing them leaves P2STCascadeSession on its own
    # holdback=1, keeping every published k1/k25/offline number byte-identical.
    # onset_diagnosis measured why these matter: speech onset equals the MT
    # committer's first release in 80/80 samples, of a 2080 ms onset 960 ms is
    # the ASR committer and 1120 ms the MT committer, and there are zero commit
    # conflicts at any setting -- so the target committer can be released
    # without the source committer losing anything.
    # The BiCodec speed token is a training-time conditioning token the cascade
    # already passes (p2st_cascade.py:425) but has always held at 1.0.  It is
    # the only knob that redistributes speech over time instead of changing how
    # much of it there is, which is what the measured timeline calls for: on the
    # demo sample C emits 6000 ms of speech carrying 3680 ms of internal holes
    # and a 3630 ms tail overhang, while the same audio spread from
    # first-audible to source end would fit at 1.06 occupancy.  Default 1.0
    # leaves every published number unchanged.
    parser.add_argument("--speed", type=float, default=1.0)
    # SimulS2ST-Omni's --enable-wait-silence-decode, ported.  Their agent
    # synthesizes cached silence codes through the vocoder on wait/idle chunks
    # "instead of emitting no audio"; without it their own comment says wait
    # chunks emit nothing, which is what this cascade does and what makes the
    # placed timeline 20-44% exact digital zeros with hard edges.  Measured on
    # their demo audio: quiet frames sit at 0.0005-0.0007 RMS, not at zero.
    # Filling the gaps with in-distribution silence codes and decoding the whole
    # timeline as ONE stream also lets the decoder's 80 ms crossfade span the
    # silence-to-speech transitions, which post-hoc pasting cannot.
    parser.add_argument("--silence-fill", action="store_true")
    parser.add_argument("--source-holdback", type=int, default=None)
    parser.add_argument("--target-holdback", type=int, default=None)
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
    silence_codes: list[int] = []
    if args.silence_fill:
        # One second of digital silence tokenized by the same BiCodec, so the
        # filler is in-distribution rather than a constant.
        scratch = Path(out_silence_dir()) / f"silence_{os.getpid()}.wav"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(scratch), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE,
                 subtype="PCM_16")
        _, semantic = codec.tokenize(str(scratch))
        silence_codes = [int(v) for v in torch.as_tensor(semantic).reshape(-1).tolist()]
        if not silence_codes:
            raise ValueError("BiCodec produced no semantic codes for silence")
        print(f"silence filler: {len(silence_codes)} codes for 1 s", flush=True)

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
            text_num_beams=args.text_num_beams,
            text_length_penalty=args.text_length_penalty,
            text_penalty=args.text_penalty,
            text_penalty_window=args.text_penalty_window,
            min_fragment_tokens=args.min_fragment_tokens,
            min_final_chunk_ms=args.min_final_chunk_ms,
            semantic_temperature=args.semantic_temperature,
            semantic_top_k=args.semantic_top_k,
            semantic_top_p=args.semantic_top_p,
            **(
                {}
                if args.semantic_penalty is None
                else {"semantic_penalty": args.semantic_penalty}
            ),
            **(
                {}
                if args.semantic_penalty_window is None
                else {"semantic_penalty_window": args.semantic_penalty_window}
            ),
            speed=args.speed,
            read_stride=args.read_stride,
            **(
                {}
                if args.source_holdback is None and args.target_holdback is None
                else {
                    "source_holdback": args.source_holdback,
                    "target_holdback": args.target_holdback,
                }
            ),
        )
        trace = session.run(waveform)
        wall = time.perf_counter() - started
        speech = [f for f in trace.fragments if f.semantic]
        mono = decode_fragments(
            codec, row.speaker_global, [tuple(f.semantic) for f in speech]
        )
        isochronous = None
        if args.silence_fill and speech:
            runs: list[tuple[int, ...]] = []
            cursor_ms = 0.0
            for fragment in speech:
                gap_ms = max(0.0, float(fragment.source_end_ms) - cursor_ms)
                n_silence = int(round(gap_ms / SEMANTIC_MS_PER_TOKEN))
                if n_silence > 0:
                    pool = silence_codes
                    repeats = -(-n_silence // max(1, len(pool)))
                    runs.append(tuple((pool * repeats)[:n_silence]))
                    cursor_ms += n_silence * SEMANTIC_MS_PER_TOKEN
                runs.append(tuple(fragment.semantic))
                cursor_ms += len(fragment.semantic) * SEMANTIC_MS_PER_TOKEN
            isochronous = decode_fragments(codec, row.speaker_global, runs)
        schedule = [(int(f.source_end_ms), len(f.semantic)) for f in speech]
        placed, placement = place_on_timeline(mono, schedule, len(waveform))

        placed_seconds = write_mono(
            out / "translation_placed" / f"{row.sample_id}.wav", placed
        )
        concat_seconds = write_mono(
            out / "translation_concat" / f"{row.sample_id}.wav", mono
        )
        if isochronous is not None:
            write_mono(
                out / "translation_isochronous" / f"{row.sample_id}.wav", isochronous
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
                "text_num_beams": args.text_num_beams,
                "text_length_penalty": args.text_length_penalty,
                "text_penalty": args.text_penalty,
                "text_penalty_window": args.text_penalty_window,
                "min_fragment_tokens": args.min_fragment_tokens,
                "min_final_chunk_ms": args.min_final_chunk_ms,
                "semantic_temperature": args.semantic_temperature,
                "semantic_top_k": args.semantic_top_k,
                "semantic_top_p": args.semantic_top_p,
                "speed": args.speed,
                "source_holdback": args.source_holdback,
                "target_holdback": args.target_holdback,
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
