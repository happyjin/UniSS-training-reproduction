"""Prefix-to-prefix task samples with the speak decision removed.

Why this exists
---------------
Four training runs have now tried to teach the interleaved model *when* to
speak, and all four failed at inference while succeeding on their own loss:

===========================  ===================  =========================
run                          post-fragment gap    inference result
===========================  ===================  =========================
baseline iter_0002264        -2.88                WRITE_MT 0.168
continue margin 1.0          -3.75  (wrong way)   unmoved
continue margin 3.0          -4.97  (wrong way)   unmoved
uniform CE, boundary 0.1->1  boundary_ce fell in  WRITE_MT 0.189,
                             every stratum        natural_eos 0.500 exactly
===========================  ===================  =========================

The uniform-CE run is the decisive one because it carried no margin at all,
so neither "the margin had the wrong shape" nor "the weight was too small"
survives it.  A following inference-side bias sweep on that checkpoint showed
the decision is not calibratable either: delta=1 leaves the median text
length ratio at 0.829 with natural_eos 0.500, and delta=2 jumps it to 1.631
with natural_eos 1.000, stepping straight over the [0.9, 1.2] band, and
delta 3/4/5 are bit-identical to each other (saturated).

So the decision is bimodal, cannot be trained teacher-forced, and cannot be
calibrated at inference.  This module removes it: every sample here is a
self-contained sequence that ends with its own terminator, and nothing in any
target span is a WAIT/WRITE choice.  Which task runs next is decided outside
the model at inference time, the way SpeakStream's Scheme 1 does it.

What it costs
-------------
Nothing in data and nothing in vocabulary.  The three task tokens are already
allocated -- ``TOKEN_TASK_STREAMING_ASR`` 180383,
``TOKEN_TASK_STREAMING_TEXT_TRANSLATION`` 180398 and
``TOKEN_TASK_STREAMING_TTS`` 180382 -- they simply have never been emitted by
the interleaved builders, so an existing checkpoint already carries
embeddings for them.  The fragment alignment the TTS task needs is already in
the 15-shard gold trajectories: of the events that carry a
``target_text_delta``, every single one also carries the matching
``target_semantic_delta`` with explicit ``target_semantic_start`` /
``target_semantic_end`` offsets.

Loss shape
----------
Pure cross-entropy on the target span only, exactly like the offline phase3
``build_tts_sample`` whose whole-utterance duration ratio is a healthy 1.039.
Content tokens carry their own kind (``LOSS_ASR`` / ``LOSS_MT`` /
``LOSS_SEMANTIC``), each sequence's single terminator carries
``LOSS_BOUNDARY``, and the final ``TOKEN_EOS`` carries ``LOSS_EOS``.  That
means the established objective needs no extension: the boundary bucket here
holds exactly one unambiguous token per sequence instead of the interleaved
task's mixture of WAIT/WRITE/TASK/language/speed tokens, which is what let a
32.8% token share collapse onto a 4.2% sub-class.

Causality
---------
Each sample's prompt holds only what a real streaming session would have
heard or emitted by that event: source GLM positions ``0 .. source_glm_end``,
the source text committed before the supervised delta, and the target text
and semantic tokens committed before it.  Nothing from a later event appears
in a prompt.

.. warning::

   The audio-prefix wiring (``speech_indices`` / ``source_glm_length`` over a
   truncated GLM prefix) is new here: the established
   ``build_streaming_asr_task`` always passes the whole trajectory.  Validate
   it with the bridge-parity gate before any GPU training.  The unit tests in
   this experiment cover token layout, loss placement, causality and
   alignment, which is what is checkable without a frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.data.schema import (
    E2ETrajectory,
    TrajectoryEvent,
)
from experiments.uniss_phase3_v4_e2e_simuls2st_pilot15_v1.training.task_samples import (
    LOSS_ASR,
    LOSS_BOUNDARY,
    LOSS_EOS,
    LOSS_MT,
    LOSS_NONE,
    LOSS_SEMANTIC,
)
from training import constants_uniss as c

FAMILY_P2ST_ASR = "p2st_streaming_asr"
FAMILY_P2ST_MT = "p2st_incremental_mt"
FAMILY_P2ST_TTS = "p2st_streaming_tts"
P2ST_FAMILIES = (FAMILY_P2ST_ASR, FAMILY_P2ST_MT, FAMILY_P2ST_TTS)

SOURCE_PREFIX_GOLD = "gold"
SOURCE_PREFIX_V1 = "v1"

EncodeText = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class P2STTaskSample:
    """One self-contained prefix-to-prefix sequence.

    Field-for-field compatible with the interleaved pool's ``E2ETaskSample``
    so the established packer and collator can consume it unchanged, but it
    carries its own family whitelist.  Defining it here rather than widening
    ``TASK_FAMILIES`` keeps the base experiment -- whose objective and Stage-A
    tensors are under a bit-for-bit frozen audit -- untouched.

    ``teacher_bindings`` and ``commit_positions`` exist only so the shapes
    match; this pool is pure CE and always leaves them empty.
    """

    sample_id: str
    sequence_id: str
    source_manifest_record: int
    family: str
    token_ids: tuple[int, ...]
    loss_kinds: tuple[int, ...]
    speech_indices: tuple[int | None, ...]
    source_audio: str | None
    source_glm_length: int
    source_glm_ids: tuple[int, ...] = ()
    teacher_bindings: tuple[object, ...] = ()
    commit_key: str | None = None
    commit_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in P2ST_FAMILIES:
            raise ValueError(f"unknown p2st task family {self.family!r}")
        if not (
            len(self.token_ids)
            == len(self.loss_kinds)
            == len(self.speech_indices)
        ):
            raise ValueError("token, loss and speech arrays must be parallel")
        if not self.token_ids:
            raise ValueError("a task sample cannot be empty")


class _Builder:
    """Accumulates one sequence, tracking the prompt/target split."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.loss_kinds: list[int] = []
        self.speech_indices: list[int | None] = []

    def observe(
        self,
        values: Sequence[int],
        speech: Sequence[int | None] | None = None,
    ) -> None:
        """Append prompt tokens, which never carry loss."""
        if speech is not None and len(speech) != len(values):
            raise ValueError("speech index count must match token count")
        self.tokens.extend(int(value) for value in values)
        self.loss_kinds.extend([LOSS_NONE] * len(values))
        self.speech_indices.extend(
            [None] * len(values) if speech is None else list(speech)
        )

    def supervise(self, values: Sequence[int], kind: int) -> None:
        """Append target tokens carrying ``kind``."""
        self.tokens.extend(int(value) for value in values)
        self.loss_kinds.extend([kind] * len(values))
        self.speech_indices.extend([None] * len(values))

    def finish(self, terminator: int) -> None:
        """Append the sequence's own terminator and EOS."""
        self.supervise([terminator], LOSS_BOUNDARY)
        self.supervise([c.TOKEN_EOS], LOSS_EOS)


def _semantic_prefix(
    events: Sequence[TrajectoryEvent], stop_index: int
) -> tuple[int, ...]:
    """Target semantic tokens committed strictly before ``stop_index``."""
    return tuple(
        int(value)
        for event in events
        if event.event_index < stop_index
        for value in event.target_semantic_delta
    )


def _source_prefix_before(event: TrajectoryEvent, kind: str) -> str:
    """The source text committed before this event's supervised delta.

    ``gold_source_prefix`` already includes the event's own delta, so the
    prompt has to be the prefix with that delta removed -- otherwise the
    answer would be visible in the question.
    """
    if kind == SOURCE_PREFIX_GOLD:
        prefix, delta = event.gold_source_prefix, event.gold_source_delta
    elif kind == SOURCE_PREFIX_V1:
        prefix = event.v1_source_prefix or ""
        delta = event.v1_source_delta or ""
    else:
        raise ValueError(f"unknown source prefix kind {kind!r}")
    if delta and prefix.endswith(delta):
        return prefix[: len(prefix) - len(delta)].rstrip()
    return prefix


def build_p2st_streaming_asr_tasks(
    trajectory: E2ETrajectory,
    *,
    encode_text: EncodeText,
) -> list[E2ETaskSample]:
    """One sample per event that transcribes new source text.

    Prompt: the causal source GLM prefix plus the transcript committed so far.
    Target: this event's transcript delta, then ``END_CONTENT`` and EOS.
    """
    output: list[E2ETaskSample] = []
    for event in trajectory.events:
        if not event.gold_source_delta.strip():
            continue
        delta = tuple(int(v) for v in encode_text(event.gold_source_delta))
        if not delta:
            continue
        glm_stop = int(event.source_glm_end)
        if glm_stop <= 0:
            continue
        builder = _Builder()
        builder.observe(
            [
                c.TOKEN_TASK_STREAMING_ASR,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.src_lang),
            ]
        )
        builder.observe(
            [
                c.TOKEN_START_GLM,
                *([c.glm_semantic_id(0)] * glm_stop),
                c.TOKEN_END_GLM,
            ],
            [None, *range(glm_stop), None],
        )
        prefix = _source_prefix_before(event, SOURCE_PREFIX_GOLD)
        committed = tuple(int(v) for v in encode_text(prefix)) if prefix else ()
        builder.observe([c.TOKEN_START_CONTENT, *committed])
        builder.supervise(delta, LOSS_ASR)
        builder.finish(c.TOKEN_END_CONTENT)
        output.append(
            P2STTaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=f"{trajectory.sample_id}:p2st_asr:{event.event_index}",
                source_manifest_record=trajectory.source_manifest_record,
                family=FAMILY_P2ST_ASR,
                token_ids=tuple(builder.tokens),
                loss_kinds=tuple(builder.loss_kinds),
                speech_indices=tuple(builder.speech_indices),
                source_audio=trajectory.source_audio,
                source_glm_length=glm_stop,
                source_glm_ids=tuple(
                    int(value)
                    for item in trajectory.events
                    if item.event_index <= event.event_index
                    for value in item.source_glm_delta
                ),
            )
        )
    return output


def build_p2st_incremental_mt_tasks(
    trajectory: E2ETrajectory,
    *,
    encode_text: EncodeText,
    source_prefix_kind: str = SOURCE_PREFIX_GOLD,
) -> list[E2ETaskSample]:
    """One sample per event that commits new target text.

    Prompt: the source transcript available at this event and the translation
    committed so far.  Target: this event's translation delta, then
    ``END_CONTENT`` and EOS.

    ``source_prefix_kind`` selects the gold transcript or the model's own V1
    ASR output, which the trajectories already carry as ``v1_source_prefix``.
    The V1 form is a genuine roll-in for this task at no data cost, so the
    exposure-bias question can be answered by a single flag rather than a new
    data build.
    """
    output: list[E2ETaskSample] = []
    for event in trajectory.events:
        if not event.target_text_delta.strip():
            continue
        delta = tuple(int(v) for v in encode_text(event.target_text_delta))
        if not delta:
            continue
        if source_prefix_kind == SOURCE_PREFIX_GOLD:
            source_text = event.gold_source_prefix
        else:
            source_text = event.v1_source_prefix or ""
        if not source_text.strip():
            continue
        target_prefix = event.target_text_prefix
        if target_prefix.endswith(event.target_text_delta):
            target_prefix = target_prefix[
                : len(target_prefix) - len(event.target_text_delta)
            ]
        builder = _Builder()
        builder.observe(
            [
                c.TOKEN_TASK_STREAMING_TEXT_TRANSLATION,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.tgt_lang),
                c.TOKEN_START_CONTENT,
                *(int(v) for v in encode_text(source_text)),
                c.TOKEN_END_CONTENT,
                c.TOKEN_START_CONTENT,
                *(tuple(int(v) for v in encode_text(target_prefix))
                  if target_prefix else ()),
            ]
        )
        builder.supervise(delta, LOSS_MT)
        builder.finish(c.TOKEN_END_CONTENT)
        output.append(
            P2STTaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=(
                    f"{trajectory.sample_id}:p2st_mt:{event.event_index}"
                    f":{source_prefix_kind}"
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


def build_p2st_streaming_tts_tasks(
    trajectory: E2ETrajectory,
    *,
    encode_text: EncodeText,
    speed: float = 1.0,
) -> list[E2ETaskSample]:
    """One sample per event that speaks a new target fragment.

    This is the builder the interleaved pool never had: ``task_samples.py``
    emits ``TOKEN_TASK_STREAMING_TTS`` nowhere, so the speech side has only
    ever been trained inside the interleaved sequence, where its terminator
    competes with WAIT/WRITE for the same boundary bucket.

    Prompt: the speaker identity, the target text committed through this
    event, and the semantic tokens already spoken.  Target: this event's
    semantic delta, then ``END_SEMANTIC`` and EOS.
    """
    output: list[E2ETaskSample] = []
    for event in trajectory.events:
        if not event.target_semantic_delta:
            continue
        spoken = _semantic_prefix(trajectory.events, event.event_index)
        if len(spoken) != int(event.target_semantic_start):
            raise ValueError(
                "semantic prefix length does not match target_semantic_start "
                f"for {trajectory.sample_id} event {event.event_index}: "
                f"{len(spoken)} != {event.target_semantic_start}"
            )
        builder = _Builder()
        builder.observe(
            [
                c.TOKEN_TASK_STREAMING_TTS,
                c.TOKEN_STREAMING_MODE,
                c.language_token_id(trajectory.tgt_lang),
                c.speed_token_id(speed),
                *c.wrap_global_tokens(trajectory.speaker_global),
                c.TOKEN_START_CONTENT,
                *(int(v) for v in encode_text(event.target_text_prefix)),
                c.TOKEN_END_CONTENT,
                c.TOKEN_START_SEMANTIC,
                *c.encode_bicodec_semantic(spoken),
            ]
        )
        builder.supervise(
            c.encode_bicodec_semantic(event.target_semantic_delta), LOSS_SEMANTIC
        )
        builder.finish(c.TOKEN_END_SEMANTIC)
        output.append(
            P2STTaskSample(
                sample_id=trajectory.sample_id,
                sequence_id=f"{trajectory.sample_id}:p2st_tts:{event.event_index}",
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


__all__ = [
    "P2STTaskSample",
    "FAMILY_P2ST_ASR",
    "FAMILY_P2ST_MT",
    "FAMILY_P2ST_TTS",
    "P2ST_FAMILIES",
    "SOURCE_PREFIX_GOLD",
    "SOURCE_PREFIX_V1",
    "build_p2st_incremental_mt_tasks",
    "build_p2st_streaming_asr_tasks",
    "build_p2st_streaming_tts_tasks",
]
