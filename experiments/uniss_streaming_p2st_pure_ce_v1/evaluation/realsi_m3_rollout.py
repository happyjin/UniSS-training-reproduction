#!/usr/bin/env python3
"""Run the m3 interleaved runtime over RealSI on a uniform read grid.

A sibling of ``realsi_rollout.py`` (which runs C's cascade), and an edit to
nothing.  Every piece of m3's inference path is imported and driven as-is:
``PacedInterleavedSession`` from ``uniss_phase3_e2e_commit_policy_v1``, the
``PersistentInterleavedSession`` event grammar it inherits, the shared causal
WhisperVQ frontend, and the streaming BiCodec decoder.  m3's own files stay
byte-identical so its published gate numbers keep reproducing.

WHY THIS RUN EXISTS, AND WHAT IT CORRECTS
-----------------------------------------
m3's published latency -- first-audible 883/978 ms, LAAL 518/2184 ms -- was
measured on ``trajectory.events``, the *gold* event list.  Reading
``run_worker.py`` settles what that means: the free-running loop is
``for event in trajectory.events``, and ``run_event`` consumes only
``source_glm_start`` / ``source_glm_end`` / ``source_end_ms`` /
``source_final`` from each event.  No gold text reaches the model -- WAIT,
WRITE_ASR, WRITE_MT, WRITE_SEMANTIC and EOS are all its own choices -- but the
*read schedule* is entirely gold.

And the gold schedule is not a uniform grid.  On the formal gold trajectories
the events land on 160 ms boundaries yet skip most of them: sample
``NCSSD_R_EN_0000000083`` is 6940 ms (43 possible 160 ms read points) with 23
events, at 160, 320, 960, 1280, 1440, ... -- and some carry
``source_glm_start == source_glm_end``, a read step that admits no new audio at
all.  Those boundaries are the gold word/event boundaries.  So m3 was asked
"decide now" exactly at the moments a word had just finished.  That is oracle
segmentation, and it flatters latency: the model never has to discover a
boundary, and it is never asked mid-word.

RealSI has no gold events, so a schedule has to be supplied.  The honest choice
for comparing against C is the same grid C reads on: one event per 160 ms block
(``--read-stride 1``, C's k1), which is 2 GLM frames per event because the GLM
frame is exactly 80 ms -- measured on the gold trajectories as
``source_duration_ms / source_glm_length`` = 79.51-80.00 with
``glm_length == ceil(samples / 1280)`` on every sample checked.

The two arms therefore share: the 777 frozen RealSI segments, the 160 ms read
grid, the WhisperVQ frontend and its GLM embedding math (identical three lines
in ``p2st_cascade.py:352-357`` and ``persistent_runtime.py:_speech_embeddings``),
the BiCodec streaming decoder, the emission-timeline placement, the SimulEval
scorers and the ASR-BLEU chain.  They differ in the weights and in the runtime
those weights were trained for.  That difference is the point; it is not
something this script can or should paper over.

WHAT IS REPLICATED RATHER THAN IMPORTED, AND WHY
------------------------------------------------
``_speech_embeddings`` is three lines of math wrapped in a length reconciliation
against a *pre-recorded* ``source_glm_length``.  Calling it would need that
length up front, and the only way to know it is to run the frontend -- which is
the expensive part, and ``run_cached_frontend`` caches within a pass, not across
passes.  So the frontend runs once here and the same three lines are applied to
its output.  The reconciliation branch is unreachable by construction when the
length comes from the live frontend, which is why it is absent.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.uniss_phase3_e2e_commit_policy_v1.runtime.semantic_pacing import (
    PacedInterleavedSession,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr import (  # noqa: E501
    evaluate_checkpoint as stage_a_eval,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.checkpoint_runtime import (  # noqa: E501
    make_cached_frontend,
    run_cached_frontend,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.data.public_corpus import load_selection
from experiments.uniss_streaming_p2st_pure_ce_v1.evaluation.timeline_demos import (
    SAMPLE_RATE,
    decode_fragments,
)
from training import constants_uniss as c
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer


BLOCK_MS = 160
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000
# Measured on the formal gold trajectories: source_duration_ms /
# source_glm_length is 79.51-80.00 and glm_length == ceil(samples / 1280).
GLM_FRAME_SAMPLES = 1280
GLM_FRAMES_PER_BLOCK = BLOCK_SAMPLES // GLM_FRAME_SAMPLES
SEMANTIC_MS_PER_TOKEN = 20.0


@dataclass(frozen=True)
class ReadEvent:
    """The four fields ``PersistentInterleavedSession.run_event`` reads."""

    event_index: int
    source_end_ms: int
    source_final: bool
    source_glm_start: int
    source_glm_end: int


@dataclass(frozen=True)
class InterleavedInput:
    """The three fields the interleaved session reads off a trajectory.

    ``PersistentInterleavedSession.__init__`` uses ``tgt_lang`` and
    ``speaker_global`` for the header; ``run_event`` uses ``src_lang`` for the
    ASR family's language token.  Nothing else on ``E2ETrajectory`` is touched
    on the run path, so a 29-field record is not synthesized -- its
    ``*_sha256`` and ``*_audit_status`` fields would be fabricated provenance
    for values no code reads.  The duck-typed loader follows the precedent in
    ``uniss_phase3_v4_e2e_simuls2st_pilot15_v1/tests/test_rollout.py:169``.
    """

    src_lang: str
    tgt_lang: str
    speaker_global: tuple[int, ...]


def build_read_grid(
    *, source_samples: int, glm_length: int, read_stride: int
) -> list[ReadEvent]:
    """One event per ``read_stride`` blocks of 160 ms, contiguous in GLM frames.

    ``_append_source`` requires monotone, contiguous spans, so each event picks
    up exactly where the last left off and the final event is clamped to the
    frontend's true frame count.  The last block always gets an event, and only
    it carries ``source_final``, which is what makes EOS legal there and only
    there.
    """
    if read_stride < 1:
        raise ValueError("read_stride must be at least 1")
    total_blocks = max(1, math.ceil(source_samples / BLOCK_SAMPLES))
    steps = list(range(read_stride - 1, total_blocks, read_stride))
    if not steps or steps[-1] != total_blocks - 1:
        steps.append(total_blocks - 1)
    events: list[ReadEvent] = []
    cursor = 0
    for position, block_index in enumerate(steps):
        samples = min(source_samples, (block_index + 1) * BLOCK_SAMPLES)
        frames = min(glm_length, (block_index + 1) * GLM_FRAMES_PER_BLOCK)
        events.append(
            ReadEvent(
                event_index=position,
                source_end_ms=int(round(1000.0 * samples / SAMPLE_RATE)),
                source_final=position == len(steps) - 1,
                source_glm_start=cursor,
                source_glm_end=max(cursor, frames),
            )
        )
        cursor = max(cursor, frames)
    return events



def load_gold_grid(
    events: list[dict], *, glm_length: int, source_duration_ms: int
) -> list[ReadEvent]:
    """m3's own gold read schedule, clamped to the live frontend's frame count.

    This exists to isolate the read schedule from everything else.  Running the
    same weights, the same session, the same scorers and the same data under
    both this and ``build_read_grid`` differs in exactly one thing -- when the
    model is asked to decide -- which is the only way to say how much of m3's
    published latency was the gold boundaries rather than the policy.

    The stored ``source_glm_end`` can exceed what the live frontend produces by
    a frame (``_speech_embeddings`` carries a terminal-extension branch for the
    same reason), so ends are clamped and the last event is extended to cover
    whatever remains.  Clamping can leave an event empty, which is legal: 15.8%
    of gold events already admit no new audio.
    """
    grid: list[ReadEvent] = []
    cursor = 0
    for position, event in enumerate(events):
        end = min(int(event["source_glm_end"]), glm_length)
        grid.append(
            ReadEvent(
                event_index=position,
                source_end_ms=min(int(event["source_end_ms"]), int(source_duration_ms)),
                source_final=position == len(events) - 1,
                source_glm_start=cursor,
                source_glm_end=max(cursor, end),
            )
        )
        cursor = max(cursor, end)
    if not grid:
        raise ValueError("gold event list is empty")
    if grid[-1].source_glm_end < glm_length:
        last = grid[-1]
        grid[-1] = ReadEvent(
            event_index=last.event_index,
            source_end_ms=int(source_duration_ms),
            source_final=True,
            source_glm_start=last.source_glm_start,
            source_glm_end=glm_length,
        )
    return grid

class InstrumentedPacedSession(PacedInterleavedSession):
    """m3's paced session, with per-fragment termination recorded.

    ``InterleavedEvent`` reports ``malformed_segments`` as one integer that
    pools text and semantic faults, so it cannot answer "did this speech
    fragment stop on END_SEMANTIC or hit the cap" -- the quantity C reports as
    ``terminator_rate``.  Wrapping ``_generate_semantic`` records exactly that,
    changes no behaviour, and lives here rather than in m3's file.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_terminated: list[bool] = []

    def _generate_semantic(self, *, max_tokens: int):
        values, ended = super()._generate_semantic(max_tokens=max_tokens)
        if values:
            self.semantic_terminated.append(bool(ended))
        return values, ended


@torch.inference_mode()
def glm_embeddings(objective, qwen, hidden: torch.Tensor) -> torch.Tensor:
    """``persistent_runtime._speech_embeddings``'s math, minus the length check."""
    device = next(objective.parameters()).device
    bridge_dtype = objective.bridge_norm.weight.dtype
    hidden = hidden.to(device=device, dtype=bridge_dtype)
    codes = objective._nearest_codes(hidden)
    residual = objective.bridge_projection(objective.bridge_norm(hidden))
    base = qwen.get_input_embeddings()(codes.long() + c.GLM_SEMANTIC_OFFSET)
    return base + residual.to(base.dtype)


def place(
    translation: np.ndarray, schedule: Sequence[tuple[int, int]], total_samples: int
) -> tuple[np.ndarray, list[float], dict]:
    """``timeline_stereo.place_on_timeline``, also returning each start in ms.

    The recurrence is the same one: a fragment cannot begin before the source
    time that produced it, and cannot overlap the fragment still playing, so
    ``start = max(earliest, cursor_out)``.  ``place_on_timeline`` returns only
    aggregate stats; the per-fragment starts are needed for ``intervals`` and
    ``silences``, which is why the two lines are rewritten here.
    """
    samples_per_token = int(SAMPLE_RATE * SEMANTIC_MS_PER_TOKEN / 1000.0)
    placed = np.zeros(max(total_samples, 1), dtype=np.float32)
    cursor_in = 0
    cursor_out = 0
    starts: list[float] = []
    late: list[float] = []
    for source_end_ms, count in schedule:
        piece = translation[cursor_in : cursor_in + count * samples_per_token]
        cursor_in += count * samples_per_token
        if not len(piece):
            continue
        earliest = int(round(source_end_ms * SAMPLE_RATE / 1000.0))
        start = max(earliest, cursor_out)
        starts.append(1000.0 * start / SAMPLE_RATE)
        late.append(1000.0 * (start - earliest) / SAMPLE_RATE)
        end = start + len(piece)
        if end > len(placed):
            placed = np.concatenate(
                [placed, np.zeros(end - len(placed), dtype=np.float32)]
            )
        placed[start:end] += piece
        cursor_out = end
    stats = {
        "fragments": len(schedule),
        "placed_seconds": cursor_out / SAMPLE_RATE,
        "concatenated_seconds": len(translation) / SAMPLE_RATE,
        "queueing_delay_ms_mean": float(np.mean(late)) if late else 0.0,
        "queueing_delay_ms_max": float(np.max(late)) if late else 0.0,
        "unused_translation_samples": max(0, len(translation) - cursor_in),
    }
    return placed, starts, stats


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
    parser.add_argument("--arm", required=True)
    parser.add_argument("--read-stride", type=int, default=1)
    parser.add_argument(
        "--gold-events",
        default=None,
        help=(
            "JSON of {sample_id: [event, ...]}.  When given, m3 reads on its own "
            "gold schedule instead of the uniform grid, which isolates the read "
            "schedule from the data and the metric."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    # m3's gate configuration, from
    # uniss_phase3_e2e_continue_end_v1/scripts/wait_then_export_and_gate.sh
    # (MAX_S2S_SEMANTIC_TOKENS=384, PACE_MARGIN_MS=1200, UNISS_E2E_SEMANTIC_PACE=1)
    # plus run_worker.py's own defaults for the two it does not override.
    parser.add_argument("--max-s2s-semantic-tokens", type=int, default=384)
    parser.add_argument("--max-s2s-fragments", type=int, default=4)
    parser.add_argument("--max-s2s-text-tokens", type=int, default=48)
    parser.add_argument("--pace-margin-ms", type=float, default=1200.0)
    parser.add_argument("--pace-tail-ms", type=float, default=2000.0)
    parser.add_argument("--keep-stereo", action="store_true")
    args = parser.parse_args()

    rows = load_selection(args.selection)
    gold_events = (
        json.loads(Path(args.gold_events).read_text(encoding='utf-8'))
        if args.gold_events
        else None
    )
    if gold_events is not None:
        missing = [r.sample_id for r in rows if r.sample_id not in gold_events]
        if missing:
            raise KeyError(f'no gold events for {len(missing)} samples, e.g. {missing[0]}')
    if args.num_shards > 1:
        rows = [
            row
            for index, row in enumerate(rows)
            if index % args.num_shards == args.shard_index
        ]
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
    if int(model.config.vocab_size) < len(tokenizer):
        raise ValueError("m3 vocabulary is smaller than its tokenizer")
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
        cached = run_cached_frontend(frontend, waveform)
        hidden = cached.hidden[0]
        embeddings = glm_embeddings(objective, model, hidden)
        if gold_events is not None:
            events = load_gold_grid(
                gold_events[row.sample_id],
                glm_length=len(hidden),
                source_duration_ms=row.source_duration_ms,
            )
        else:
            events = build_read_grid(
                source_samples=len(waveform),
                glm_length=len(hidden),
                read_stride=args.read_stride,
            )
        session = InstrumentedPacedSession(
            model,
            tokenizer,
            embeddings,
            InterleavedInput(
                src_lang=row.src_lang,
                tgt_lang=row.tgt_lang,
                speaker_global=tuple(row.speaker_global),
            ),
            pace_margin_ms=args.pace_margin_ms,
            pace_tail_ms=args.pace_tail_ms,
        )

        fragments: list[tuple[int, ...]] = []
        malformed = 0
        continuations: list[str] = []
        eos_event: int | None = None
        # (source_end_ms, tokens, elapsed_ms, committed target text)
        speech: list[tuple[int, tuple[int, ...], float, str]] = []
        for event in events:
            if session.closed:
                malformed += 1
                break
            record = session.run_event(
                event,
                max_fragments=args.max_s2s_fragments,
                max_text_tokens=args.max_s2s_text_tokens,
                max_semantic_tokens=args.max_s2s_semantic_tokens,
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            malformed += record.malformed_segments
            continuations.extend(record.chosen_continuations)
            if "EOS" in record.chosen_continuations:
                eos_event = record.event_index
            if record.semantic_tokens:
                fragments.append(record.semantic_tokens)
                speech.append(
                    (
                        int(record.source_end_ms),
                        record.semantic_tokens,
                        elapsed_ms,
                        session.target_text,
                    )
                )
        malformed += int(not session.closed)

        mono = decode_fragments(codec, tuple(row.speaker_global), fragments)
        schedule = [(end_ms, len(tokens)) for end_ms, tokens, _, _ in speech]
        placed, starts, placement = place(mono, schedule, len(waveform))
        wall = time.perf_counter() - started

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

        delays = [float(end_ms) for end_ms, _, _, _ in speech]
        durations = [SEMANTIC_MS_PER_TOKEN * len(tokens) for _, tokens, _, _ in speech]
        elapsed = [value for _, _, value, _ in speech]
        intervals = [
            [start, SEMANTIC_MS_PER_TOKEN * len(tokens)]
            for start, (_, tokens, _, _) in zip(starts, speech)
        ]
        silences: list[float] = []
        previous_end = 0.0
        for start, (_, tokens, _, _) in zip(starts, speech):
            silences.append(max(0.0, start - previous_end))
            previous_end = start + SEMANTIC_MS_PER_TOKEN * len(tokens)
        mt_delta_chars: list[int] = []
        previous_text = ""
        for _, _, _, text in speech:
            mt_delta_chars.append(max(0, len(text) - len(previous_text)))
            previous_text = text

        manifest.append(
            {
                "sample_id": row.sample_id,
                "arm": args.arm,
                "read_stride": args.read_stride,
                "read_schedule": "gold" if gold_events is not None else "uniform",
                "read_step_ms": args.read_stride * BLOCK_MS,
                "direction": row.direction,
                "src_lang": row.src_lang,
                "tgt_lang": row.tgt_lang,
                "source_duration_ms": row.source_duration_ms,
                "read_steps": len(events),
                "audio_blocks": max(1, math.ceil(len(waveform) / BLOCK_SAMPLES)),
                "fragments": len(speech),
                "semantic_tokens": sum(len(tokens) for _, tokens, _, _ in speech),
                "translation_placed": str(
                    out / "translation_placed" / f"{row.sample_id}.wav"
                ),
                "translation_concat": str(
                    out / "translation_concat" / f"{row.sample_id}.wav"
                ),
                "placed_seconds": placed_seconds,
                "concat_seconds": concat_seconds,
                "delays": delays,
                "durations": durations,
                "intervals": intervals,
                "elapsed": elapsed,
                "silences": silences,
                "source_hypothesis": session.source_text,
                "target_hypothesis": session.target_text,
                "transcription_reference": row.reference_transcription,
                "translation_reference": row.reference_translation,
                "terminator_rate": (
                    sum(1 for value in session.semantic_terminated if value)
                    / max(1, len(session.semantic_terminated))
                ),
                "capped_stages": sum(
                    1 for value in session.semantic_terminated if not value
                ),
                "mt_delta_chars": mt_delta_chars,
                "mt_delta_chars_over_prior_bucket": sum(
                    1 for value in mt_delta_chars if value > 24
                ),
                # The interleaved session appends monotone deltas and has no
                # revision committer, so there is nothing to conflict.  Null
                # rather than 0: "no conflicts" and "no mechanism that could
                # conflict" are different facts and the report must not read
                # the second as the first.
                "revision_conflicts": {"source": None, "target": None},
                "placement": placement,
                # m3-only structure, for reading the latency honestly.
                "wait_events": sum(1 for value in continuations if value == "WAIT"),
                "read_next_events": sum(
                    1 for value in continuations if value == "READ_NEXT"
                ),
                "write_asr_events": sum(
                    1 for value in continuations if value == "WRITE_ASR"
                ),
                "write_mt_events": sum(
                    1 for value in continuations if value == "WRITE_MT"
                ),
                "write_semantic_events": sum(
                    1 for value in continuations if value == "WRITE_SEMANTIC"
                ),
                "eos_event_index": eos_event,
                "eos_reached": session.closed,
                "malformed_segments": malformed,
                "pace_margin_ms": args.pace_margin_ms,
                "pace_tail_ms": args.pace_tail_ms,
                "wall_seconds": wall,
                "rtf": wall / max(row.source_duration_ms / 1000.0, 1e-9),
            }
        )
        if position % 20 == 0:
            last = manifest[-1]
            print(
                f"  [{args.arm} shard{args.shard_index}] {position + 1}/{len(rows)} "
                f"{row.sample_id} steps={last['read_steps']} "
                f"frags={last['fragments']} sem={last['semantic_tokens']} "
                f"mt={last['write_mt_events']} eos={last['eos_reached']} "
                f"rtf={last['rtf']:.2f}",
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
