"""Fixed-chunk task samples with explicit IDLE supervision.

Why this exists
---------------
Every builder in ``task_samples_p2st.py`` opens the same way::

    for event in trajectory.events:
        if not event.gold_source_delta.strip():   # ASR
            continue
        if not event.target_text_delta.strip():   # MT
            continue
        if not event.target_semantic_delta:       # TTS
            continue

so a training sample exists **only** where gold says something was committed,
and the read points are the gold word boundaries.  Two consequences follow,
and both are measured rather than argued:

1. **The model has never been shown a step on which the right answer is
   "nothing".**  At inference the runtime asks it once per fixed clock tick,
   and most ticks commit nothing.  Re-binning 60,000 15-shard gold
   trajectories onto fixed grids (481,882 windows at 640 ms):

   =========  ==========  ==========  ==========
   chunk      no target   no source   no event
              content     transcript  at all
   =========  ==========  ==========  ==========
   160 ms     0.840       0.704       0.593
   320 ms     0.701       0.497       0.371
   **640 ms** **0.508**   **0.299**   0.152
   960 ms     0.407       0.232       0.088
   1920 ms    0.219       0.141       0.000
   =========  ==========  ==========  ==========

   The first column is the paper's own criterion -- "chunks without newly
   committed *target* content" -- and it is the one the MT and TTS families
   see.  The ASR family reads the second column, because its content is the
   source transcript.  Either way the overwhelmingly common case at inference
   is the one case training never contained.
2. **The read schedule itself is gold.**  Measured over 3000 trajectories the
   gold events use a median 0.429 of the available 160 ms read points and
   15.8% of them admit no new audio at all, so the intervals are irregular and
   content-aligned in a way no deployment clock can be.

This module re-bins the same gold trajectories onto a fixed chunk grid and
supervises both cases: a chunk that committed new content is supervised with
the *merged* delta of the events that fall inside it, and a chunk that
committed nothing is supervised with its terminator alone.

Where this comes from
---------------------
SimulS2ST-Omni (arXiv 2607.19810) §3.2 Step 2, verbatim:

    "group adjacent target words and their codes whose boundaries fall within
    the same pre-defined source chunk intervals of 1 second ... Chunks without
    newly committed target content act as read/wait steps"

and the runtime form of the second sentence is in their repository at
``src/agents/simuleval_omni_talker_s2st_agent.py``, which defines
``DEFAULT_IDLE_TOKEN``, tests for it in ``_is_wait()``, and at line 350 writes
``assistant_text = text if text else DEFAULT_IDLE_TOKEN``.

Where this deliberately differs
-------------------------------
* **Chunk size 640 ms, not their 1 s.**  Chosen from the table above: on the
  paper's own criterion 640 ms is the grid where supervision is closest to
  balanced (0.508), where 1 s sits near 0.40 and 160 ms near 0.84, and it is
  simultaneously the coarsest grid on which the ASR family still sees a
  substantial read/wait share (0.299).  It is also a whole number of frontend
  blocks, which 1 s is not: ``BLOCK_MS`` is 160, so 640 ms is exactly four
  read steps and 1 s is 6.25.  Their 1 s figure is a choice for their system,
  not a result, and §4.5 of the same paper reports the sampling schedule is
  "negligible" next to pool curation.
* **No new vocabulary entry for IDLE.**  They add a literal ``IDLE`` string to
  a text LM's prompt, which costs nothing.  Our 180k-entry vocabulary is
  shared with an existing checkpoint lineage, so a new id means resizing the
  embedding and the output head.  Instead an IDLE chunk is trained as *emit
  the terminator immediately* -- ``END_CONTENT`` for ASR and MT -- which is a
  token the model already has and already knows how to produce.  The cost is
  that the IDLE lesson lands in the ``boundary_eos`` bucket rather than in a
  bucket of its own; see reports/uniss_streaming_p2st_traj_v1/STEP2_DESIGN.zh-CN.md
* **MT emits no IDLE when ``mt_idle_ratio`` is 0, which is the measured
  recommendation.**  Two numbers decide this, taken over 3000 trajectories
  and 33,777 chunks.  On a chunk that admits at least one gold event -- 90%
  of them, and the only case inference presents -- the ASR label is nearly
  determined, ``P(idle) = 0.089``, because "was anything said in this 640 ms"
  is a property of the audio.  The MT label on the same chunks is
  ``P(idle) = 0.469``: a coin flip, because it encodes when the *aligner*
  chose to commit a translation, which is not visible in the input at all.
  Training on it cannot teach when to wait; it can only raise the model's
  prior for terminating.  And it is redundant besides -- ``switch_rule``
  already routes an empty MT delta to ``TASK_READ`` -- so at inference the
  runtime expresses the read/wait step whether or not the model was taught
  to.  Measured consequence of leaving it on: at 200 steps the MT stage
  terminated on 26 of 27 read steps and the translation collapsed from 164
  characters to 2, while the ASR stage was untouched.

* **TTS emits no IDLE by default.**  Their Talker is asked on every chunk; our
  cascade invokes the TTS stage only when the MT stage produced text, so a
  "speak nothing" sample would train a condition inference never presents.
  ``tts_idle=True`` exists so that can be tested rather than assumed.

What is reused unchanged
------------------------
The prompt layouts, the separator convention and the acoustic sidecar
contract all come from ``task_samples_p2st`` by import, including its private
splitters.  That is deliberate: ``_split_running_text`` documents a measured
failure -- a dropped space made the cascade transcribe "I can't think what
takes" as ``'Ithinksomethingwilltake'`` -- and re-deriving the convention here
would be exactly the way to reintroduce it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    TrajectoryEvent,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_MT,
    LOSS_SEMANTIC,
)
from experiments.uniss_streaming_p2st_pure_ce_v1.training.task_samples_p2st import (
    FAMILY_P2ST_ASR,
    FAMILY_P2ST_MT,
    FAMILY_P2ST_TTS,
    SOURCE_PREFIX_GOLD,
    TASK_TOKENS,
    TEXT_SCOPE_DELTA,
    TEXT_SCOPE_PREFIX,
    EncodeText,
    P2STTaskSample,
    _Builder,
    _semantic_prefix,
    _split_source_text,
    _split_target_text,
    causal_glm_token_count,
)
from training import constants_uniss as c

SAMPLE_RATE = 16_000

# 640 ms, from the measured IDLE rates in the module docstring.  It is four
# 160 ms frontend blocks exactly -- BLOCK_MS is 160 in
# shared_causal_frontend.py and a GLM frame is 80 ms -- so the grid is a whole
# number of read steps and eight GLM frames.
DEFAULT_CHUNK_MS = 640

# At most this many IDLE samples per content sample, per family, per
# trajectory.  1.0 is above the natural rate at 640 ms for both criteria --
# 0.508 IDLE is 1.03 per content chunk, 0.299 is 0.43 -- so it binds on almost
# nothing and exists so a pathological utterance, a long silence with three
# words at the end, cannot flood the pool.
DEFAULT_IDLE_RATIO = 1.0


@dataclass(frozen=True)
class ChunkWindow:
    """One tick of the fixed read clock.

    ``end_ms`` is the clock boundary clamped to the utterance, ``pcm_end`` is
    the audio cut that boundary implies, and ``events`` are the gold events
    whose commitment point falls inside ``(start_ms, end_ms]``.
    """

    chunk_index: int
    start_ms: int
    end_ms: int
    pcm_end: int
    glm_stop: int
    events: tuple[TrajectoryEvent, ...]


def chunk_windows(
    trajectory: E2ETrajectory,
    *,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    glm_token_count: Callable[[int], int] = causal_glm_token_count,
) -> list[ChunkWindow]:
    """The trajectory's gold events re-binned onto a fixed ``chunk_ms`` grid.

    The grid is absolute -- boundary ``j`` is at ``(j + 1) * chunk_ms`` -- so
    it is reproducible from the clock alone and carries none of the gold
    schedule's content alignment.  Only the last boundary is clamped, to the
    end of the audio.
    """
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive")
    events = tuple(trajectory.events)
    if not events:
        return []
    final_ms = max(int(trajectory.source_duration_ms),
                   max(int(e.source_end_ms) for e in events))
    final_pcm = max(int(e.source_pcm_end) for e in events)
    if final_ms <= 0 or final_pcm <= 0:
        return []
    count = -(-final_ms // int(chunk_ms))
    windows: list[ChunkWindow] = []
    for index in range(count):
        start_ms = index * int(chunk_ms)
        end_ms = min((index + 1) * int(chunk_ms), final_ms)
        pcm_end = min(round(end_ms * SAMPLE_RATE / 1000), final_pcm)
        glm_stop = int(glm_token_count(pcm_end))
        if glm_stop <= 0 or pcm_end <= 0:
            continue
        members = tuple(
            event
            for event in events
            if int(event.source_end_ms) <= end_ms
            and (index == 0 or int(event.source_end_ms) > start_ms)
        )
        windows.append(
            ChunkWindow(
                chunk_index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                pcm_end=pcm_end,
                glm_stop=glm_stop,
                events=members,
            )
        )
    return windows


def _glm_ids_through(
    trajectory: E2ETrajectory, end_ms: int, glm_stop: int
) -> tuple[int, ...]:
    """Recorded GLM ids for the prefix, resized to the frontend's count.

    The same contract as ``task_samples_p2st._glm_ids_for_prefix``, addressed
    by clock time instead of by event index: contents feed only the
    ``diagnostic/causal_glm_agreement`` log, so what has to be right is the
    length and the value range.
    """
    recorded = [
        int(value)
        for event in trajectory.events
        if int(event.source_end_ms) <= int(end_ms)
        for value in event.source_glm_delta
    ]
    if len(recorded) >= glm_stop:
        return tuple(recorded[:glm_stop])
    filler = recorded[-1] if recorded else 0
    return tuple(recorded + [filler] * (glm_stop - len(recorded)))


def _keep_idle(idle_indices: Sequence[int], content_count: int,
               idle_ratio: float) -> set[int]:
    """An evenly spaced subset of the IDLE chunks to keep.

    Evenly spaced rather than random so the pool is reproducible from the
    trajectory alone, and so the kept IDLE chunks stay interleaved with the
    content chunks rather than clustering in the leading silence.
    """
    if idle_ratio <= 0 or content_count <= 0 or not idle_indices:
        return set()
    budget = min(len(idle_indices), math.ceil(content_count * idle_ratio))
    if budget >= len(idle_indices):
        return set(int(v) for v in idle_indices)
    step = len(idle_indices) / budget
    return {int(idle_indices[min(len(idle_indices) - 1, int(k * step))])
            for k in range(budget)}


def _source_text_through(
    trajectory: E2ETrajectory, end_ms: int, kind: str
) -> str:
    """Committed source running text at clock time ``end_ms``."""
    text = ""
    for event in trajectory.events:
        if int(event.source_end_ms) > int(end_ms):
            break
        candidate = (
            event.gold_source_prefix
            if kind == SOURCE_PREFIX_GOLD
            else (event.v1_source_prefix or "")
        )
        if candidate:
            text = candidate
    return text


def _target_text_through(trajectory: E2ETrajectory, end_ms: int) -> str:
    """Committed target running text at clock time ``end_ms``."""
    text = ""
    for event in trajectory.events:
        if int(event.source_end_ms) > int(end_ms):
            break
        if event.target_text_prefix:
            text = event.target_text_prefix
    return text


def build_uniform_chunk_asr_tasks(
    trajectory: E2ETrajectory,
    *,
    task_token: int = TASK_TOKENS[FAMILY_P2ST_ASR],
    encode_text: EncodeText,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    idle_ratio: float = DEFAULT_IDLE_RATIO,
    glm_token_count: Callable[[int], int] = causal_glm_token_count,
) -> list[P2STTaskSample]:
    """One sample per chunk of the fixed read clock.

    Prompt: the causal source GLM prefix cut at the chunk boundary, plus the
    transcript committed before the chunk.  Target: the merged transcript
    delta of the events inside the chunk, then ``END_CONTENT`` and EOS -- or,
    on a chunk that committed nothing, ``END_CONTENT`` and EOS alone.

    The GLM length comes from ``glm_token_count(pcm_end)`` where ``pcm_end``
    is the clock boundary, never from ``event.source_glm_end``, for the same
    reason the event-level builder gives: the trajectory's own offset is not
    the causal frontend's count.
    """
    windows = chunk_windows(
        trajectory, chunk_ms=chunk_ms, glm_token_count=glm_token_count
    )
    plan: list[tuple[ChunkWindow, str, tuple[int, ...]]] = []
    content: list[int] = []
    idle: list[int] = []
    for window in windows:
        members = [e for e in window.events if e.gold_source_delta.strip()]
        delta: tuple[int, ...] = ()
        committed = ""
        if members:
            committed = _split_source_text(members[0], SOURCE_PREFIX_GOLD)[0]
            delta_text = "".join(
                _split_source_text(m, SOURCE_PREFIX_GOLD)[1] for m in members
            )
            delta = tuple(int(v) for v in encode_text(delta_text)) if delta_text else ()
        if delta:
            content.append(window.chunk_index)
        else:
            committed = _source_text_through(
                trajectory, window.end_ms, SOURCE_PREFIX_GOLD
            )
            idle.append(window.chunk_index)
        plan.append((window, committed, delta))

    keep = _keep_idle(idle, len(content), idle_ratio)
    output: list[P2STTaskSample] = []
    for window, committed, delta in plan:
        if not delta and window.chunk_index not in keep:
            continue
        builder = _Builder()
        builder.observe(
            [
                task_token,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.src_lang),
                *c.wrap_global_tokens(trajectory.speaker_global),
            ]
        )
        builder.observe(
            [
                c.TOKEN_START_GLM,
                *([c.glm_semantic_id(0)] * window.glm_stop),
                c.TOKEN_END_GLM,
            ],
            [None, *range(window.glm_stop), None],
        )
        builder.observe(
            [
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(trajectory.src_lang),
                c.TOKEN_START_CONTENT,
                *(tuple(int(v) for v in encode_text(committed)) if committed else ()),
            ]
        )
        builder.supervise(delta, LOSS_ASR)
        builder.finish(c.TOKEN_END_CONTENT)
        output.append(
            P2STTaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=(
                    f"{trajectory.sample_id}:traj_asr:{int(chunk_ms)}"
                    f":{window.chunk_index}"
                ),
                source_manifest_record=trajectory.source_manifest_record,
                family=FAMILY_P2ST_ASR,
                token_ids=tuple(builder.tokens),
                loss_kinds=tuple(builder.loss_kinds),
                speech_indices=tuple(builder.speech_indices),
                source_audio=trajectory.source_audio,
                source_glm_length=window.glm_stop,
                source_glm_ids=_glm_ids_through(
                    trajectory, window.end_ms, window.glm_stop
                ),
                source_pcm_end=window.pcm_end,
            )
        )
    return output


def build_uniform_chunk_mt_tasks(
    trajectory: E2ETrajectory,
    *,
    task_token: int = TASK_TOKENS[FAMILY_P2ST_MT],
    encode_text: EncodeText,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    idle_ratio: float = DEFAULT_IDLE_RATIO,
    mt_idle_ratio: float | None = None,
    source_prefix_kind: str = SOURCE_PREFIX_GOLD,
    glm_token_count: Callable[[int], int] = causal_glm_token_count,
) -> list[P2STTaskSample]:
    """Incremental translation on the fixed clock, IDLE chunks included.

    A chunk on which no source word has yet been read produces no sample at
    all -- content or IDLE -- because the prompt would carry an empty source
    block, which is a condition the cascade never presents: it runs the MT
    stage only once the ASR stage has committed something.  A chunk that has
    source text but committed no *new translation* is exactly the read/wait
    step, and does produce an IDLE sample.
    """
    windows = chunk_windows(
        trajectory, chunk_ms=chunk_ms, glm_token_count=glm_token_count
    )
    plan: list[tuple[ChunkWindow, str, str, tuple[int, ...]]] = []
    content: list[int] = []
    idle: list[int] = []
    for window in windows:
        source_text = _source_text_through(
            trajectory, window.end_ms, source_prefix_kind
        )
        if not source_text.strip():
            continue
        members = [e for e in window.events if e.target_text_delta.strip()]
        delta: tuple[int, ...] = ()
        committed = ""
        if members:
            committed = _split_target_text(members[0])[0]
            delta_text = "".join(_split_target_text(m)[1] for m in members)
            delta = tuple(int(v) for v in encode_text(delta_text)) if delta_text else ()
        if delta:
            content.append(window.chunk_index)
        else:
            committed = _target_text_through(trajectory, window.end_ms)
            idle.append(window.chunk_index)
        plan.append((window, source_text, committed, delta))

    keep = _keep_idle(
        idle,
        len(content),
        idle_ratio if mt_idle_ratio is None else float(mt_idle_ratio),
    )
    output: list[P2STTaskSample] = []
    for window, source_text, committed, delta in plan:
        if not delta and window.chunk_index not in keep:
            continue
        builder = _Builder()
        builder.observe(
            [
                task_token,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.tgt_lang),
                c.TOKEN_START_CONTENT,
                *(int(v) for v in encode_text(source_text)),
                c.TOKEN_END_CONTENT,
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(trajectory.tgt_lang),
                c.TOKEN_START_CONTENT,
                *(tuple(int(v) for v in encode_text(committed)) if committed else ()),
            ]
        )
        builder.supervise(delta, LOSS_MT)
        builder.finish(c.TOKEN_END_CONTENT)
        output.append(
            P2STTaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=(
                    f"{trajectory.sample_id}:traj_mt:{int(chunk_ms)}"
                    f":{window.chunk_index}:{source_prefix_kind}"
                ),
                source_manifest_record=trajectory.source_manifest_record,
                family=FAMILY_P2ST_MT,
                token_ids=tuple(builder.tokens),
                loss_kinds=tuple(builder.loss_kinds),
                speech_indices=tuple(builder.speech_indices),
                source_audio=None,
                source_glm_length=0,
            )
        )
    return output


def build_uniform_chunk_tts_tasks(
    trajectory: E2ETrajectory,
    *,
    task_token: int = TASK_TOKENS[FAMILY_P2ST_TTS],
    encode_text: EncodeText,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    idle_ratio: float = DEFAULT_IDLE_RATIO,
    speed: float = 1.0,
    text_scope: str = TEXT_SCOPE_DELTA,
    tts_idle: bool = False,
    glm_token_count: Callable[[int], int] = causal_glm_token_count,
) -> list[P2STTaskSample]:
    """Speak one chunk's worth of target words, ending on a word boundary.

    This is also where the third improvement lands, and it lands by
    construction rather than by a new loss.  ``target_semantic_delta`` is
    already cut at target-word blocks -- the Stage-A ``micro_write_events``
    carry ``target_word_start/end`` beside ``semantic_start/end``, and the
    spans measure 16 codes for ``'我'`` and 52 for ``'完全同意'`` -- so merging
    *whole* deltas and terminating after the last one means ``END_SEMANTIC``
    can only ever fall on a word-block boundary.  The event-level builder had
    the same property per event; what changes here is that the chunk grid no
    longer lets a read boundary fall mid-block.

    The contiguity of the merged span is checked rather than assumed: a gap
    would mean the codes spoken are not the codes the text names.
    """
    if text_scope not in (TEXT_SCOPE_DELTA, TEXT_SCOPE_PREFIX):
        raise ValueError(f"unknown text scope {text_scope!r}")
    windows = chunk_windows(
        trajectory, chunk_ms=chunk_ms, glm_token_count=glm_token_count
    )
    plan: list[tuple[ChunkWindow, str, tuple[int, ...], tuple[int, ...]]] = []
    content: list[int] = []
    idle: list[int] = []
    for window in windows:
        members = [e for e in window.events if e.target_semantic_delta]
        if not members:
            idle.append(window.chunk_index)
            plan.append((window, "", (), ()))
            continue
        spoken = _semantic_prefix(trajectory.events, members[0].event_index)
        if len(spoken) != int(members[0].target_semantic_start):
            raise ValueError(
                "semantic prefix length does not match target_semantic_start "
                f"for {trajectory.sample_id} chunk {window.chunk_index}: "
                f"{len(spoken)} != {members[0].target_semantic_start}"
            )
        delta = tuple(int(v) for m in members for v in m.target_semantic_delta)
        if int(members[-1].target_semantic_end) != len(spoken) + len(delta):
            raise ValueError(
                "merged semantic span is not contiguous for "
                f"{trajectory.sample_id} chunk {window.chunk_index}: "
                f"{members[-1].target_semantic_end} != {len(spoken) + len(delta)}"
            )
        if text_scope == TEXT_SCOPE_DELTA:
            fragment_text = "".join(_split_target_text(m)[1] for m in members)
        else:
            fragment_text = members[-1].target_text_prefix
        if not fragment_text.strip():
            continue
        content.append(window.chunk_index)
        plan.append((window, fragment_text, spoken, delta))

    keep = _keep_idle(idle, len(content), idle_ratio) if tts_idle else set()
    output: list[P2STTaskSample] = []
    for window, fragment_text, spoken, delta in plan:
        if not delta:
            if window.chunk_index not in keep:
                continue
            spoken = _semantic_prefix_through(trajectory, window.end_ms)
        builder = _Builder()
        builder.observe(
            [
                task_token,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.tgt_lang),
                *c.wrap_global_tokens(trajectory.speaker_global),
                c.TOKEN_START_CONTENT,
                *(int(v) for v in encode_text(fragment_text)),
                c.TOKEN_END_CONTENT,
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(trajectory.tgt_lang),
                c.speed_token_id(speed),
                c.TOKEN_START_SEMANTIC,
                *c.encode_bicodec_semantic(spoken),
            ]
        )
        builder.supervise(c.encode_bicodec_semantic(delta), LOSS_SEMANTIC)
        builder.finish(c.TOKEN_END_SEMANTIC)
        output.append(
            P2STTaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=(
                    f"{trajectory.sample_id}:traj_tts:{int(chunk_ms)}"
                    f":{window.chunk_index}"
                ),
                source_manifest_record=trajectory.source_manifest_record,
                family=FAMILY_P2ST_TTS,
                token_ids=tuple(builder.tokens),
                loss_kinds=tuple(builder.loss_kinds),
                speech_indices=tuple(builder.speech_indices),
                source_audio=None,
                source_glm_length=0,
            )
        )
    return output


def _semantic_prefix_through(
    trajectory: E2ETrajectory, end_ms: int
) -> tuple[int, ...]:
    """Semantic codes committed at or before clock time ``end_ms``."""
    return tuple(
        int(value)
        for event in trajectory.events
        if int(event.source_end_ms) <= int(end_ms)
        for value in event.target_semantic_delta
    )


def build_uniform_chunk_samples(
    trajectory: E2ETrajectory,
    *,
    encode_text: EncodeText,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    idle_ratio: float = DEFAULT_IDLE_RATIO,
    mt_idle_ratio: float | None = None,
    source_prefix_kind: str = SOURCE_PREFIX_GOLD,
    tts_idle: bool = False,
) -> dict[str, list[P2STTaskSample]]:
    """The three streaming families for one trajectory, keyed by family.

    The two phase3 replay families are not built here: they are
    whole-utterance and carry no read schedule at all, so the fixed clock does
    not apply to them and ``build_traj_pools`` takes them unchanged from
    ``task_samples_p2st.build_p2st_phase3_replay_tasks``.
    """
    return {
        FAMILY_P2ST_ASR: build_uniform_chunk_asr_tasks(
            trajectory,
            encode_text=encode_text,
            chunk_ms=chunk_ms,
            idle_ratio=idle_ratio,
        ),
        FAMILY_P2ST_MT: build_uniform_chunk_mt_tasks(
            trajectory,
            encode_text=encode_text,
            chunk_ms=chunk_ms,
            idle_ratio=idle_ratio,
            mt_idle_ratio=mt_idle_ratio,
            source_prefix_kind=source_prefix_kind,
        ),
        FAMILY_P2ST_TTS: build_uniform_chunk_tts_tasks(
            trajectory,
            encode_text=encode_text,
            chunk_ms=chunk_ms,
            idle_ratio=idle_ratio,
            tts_idle=tts_idle,
        ),
    }
