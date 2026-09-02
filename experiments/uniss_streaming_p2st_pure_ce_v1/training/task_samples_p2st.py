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

The audio prefix
----------------
``StageAObjective._inject_causal_glm`` hard-raises unless the frontend's token
count for the row's waveform equals ``glm_lengths``, tolerating only a single
terminal codec slot.  A prefix-to-prefix ASR sample therefore has to cut the
*audio* as well, which is what ``source_pcm_end`` carries, and its
``source_glm_length`` has to be the count the frontend will actually return
for that cut.

That count is a closed form.  Tokens arrive one per 80 ms hop
(``TOKEN_HOP_SAMPLES`` = 16000 x 80 / 1000 = 1280), rounding a partial hop up:

    tokens = ceil(samples / TOKEN_HOP_SAMPLES)

verified against the frontend on 201 event boundaries from 12 real
trajectories with no exception, and with no collisions across the 60 distinct
cut points, so the count is a pure function of the sample count.  The same
measurement showed the frontend is genuinely block-causal -- a prefix cut at
``source_pcm_end`` reproduces the full run's tokens bit for bit -- so nothing
in the prompt is conditioned on audio the session has not heard.

The trajectory's own ``source_glm_end`` is *not* usable for this.  Checked on
5000 trajectories, it agrees with the closed form only 84.6% of the time and
is otherwise exactly 2 short, because its bookkeeping lags the frontend by one
160 ms block.  Using it would make ``causal_length != length`` and raise.

``glm_token_count`` is injectable so a caller can substitute a real frontend
measurement, and ``_inject_causal_glm``'s own check is the backstop: a wrong
count fails loudly on the first acoustic batch rather than training on a
silent mismatch.
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
    FAMILY_PHASE3_PERFORMANCE,
    FAMILY_PHASE3_QUALITY,
    LOSS_MT,
    LOSS_NONE,
    LOSS_REPLAY,
    LOSS_SEMANTIC,
    build_phase3_replay_tasks,
)
from training import constants_uniss as c

FAMILY_P2ST_ASR = "p2st_streaming_asr"
FAMILY_P2ST_MT = "p2st_incremental_mt"
FAMILY_P2ST_TTS = "p2st_streaming_tts"
P2ST_FAMILIES = (FAMILY_P2ST_ASR, FAMILY_P2ST_MT, FAMILY_P2ST_TTS)

# The two phase3 replay families are the anti-forgetting mechanism this
# experiment's own README specified and the first pool omitted: "防遗忘沿用
# phase3 自己的方式:靠数据混合,把两个 phase3_*_replay 家族按权重 1.0 混进来,
# 不加任何 loss 项".  They are reused verbatim from the interleaved pool -- same
# builder, same family names, same LOSS_REPLAY tag -- so ``replay_ce`` and the
# base experiment's family ids need no new registration.  Without them C has no
# anchor at all to the offline phase3 quality it inherits, while B' carried
# replay at 0.50 plus two teacher KLs.
REPLAY_FAMILIES = (FAMILY_PHASE3_QUALITY, FAMILY_PHASE3_PERFORMANCE)
POOL_FAMILIES = P2ST_FAMILIES + REPLAY_FAMILIES

SOURCE_PREFIX_GOLD = "gold"
SOURCE_PREFIX_V1 = "v1"

# Which task token heads each family's prompt.
#
# Measured across the repository: TOKEN_TASK_STREAMING_ASR is emitted by the
# Stage-A training pool, whose prompt header is
# ``[TASK_STREAMING_ASR, STREAMING_MODE, lang, *speaker]`` -- nearly this
# module's ASR header -- so that embedding is trained in this lineage.  The
# other two streaming task tokens are not: TASK_STREAMING_TEXT_TRANSLATION
# appears nowhere at all, and TASK_STREAMING_TTS only in a different lineage.
# Their trained counterparts are heavily used: TASK_S2T_TRANSLATION in 31
# places and TASK_TTS in 22, by both the interleaved family and offline
# phase3 -- including the phase3 ``tts`` mode whose whole-utterance duration
# ratio is a healthy 1.039.
#
# So the MT and TTS families head their prompts with the trained tokens.  That
# is not a loss of distinction: every prompt here also carries
# TOKEN_STREAMING_MODE, and the interleaved family's own TTS segment is headed
# by TASK_TTS too, so this task inherits what that segment already taught and
# differs only in being an isolated sequence that ends with its own END rather
# than competing with WAIT/WRITE for the same bucket.
TASK_TOKENS = {
    FAMILY_P2ST_ASR: c.TOKEN_TASK_STREAMING_ASR,
    # C's MT stage reads the committed source *text*, not audio, so its
    # trained counterpart is phase3's build_mt_sample, which heads a
    # text-to-text translation with TASK_T2T_TRANSLATION.  Using the
    # speech-to-text token would name a task this prompt does not perform.
    FAMILY_P2ST_MT: c.TOKEN_TASK_T2T_TRANSLATION,
    FAMILY_P2ST_TTS: c.TOKEN_TASK_TTS,
}
# The never-trained alternative, kept addressable so the choice can be tested
# rather than assumed.
UNTRAINED_TASK_TOKENS = {
    FAMILY_P2ST_ASR: c.TOKEN_TASK_STREAMING_ASR,
    FAMILY_P2ST_MT: c.TOKEN_TASK_STREAMING_TEXT_TRANSLATION,
    FAMILY_P2ST_TTS: c.TOKEN_TASK_STREAMING_TTS,
}

SAMPLE_RATE = 16_000
TOKEN_HOP_MS = 80
TOKEN_HOP_SAMPLES = SAMPLE_RATE * TOKEN_HOP_MS // 1000


def causal_glm_token_count(samples: int) -> int:
    """Tokens the causal frontend returns for ``samples`` of audio.

    One token per 80 ms hop with a partial hop rounded up.  See the module
    docstring for the measurement this reproduces.
    """
    if samples <= 0:
        return 0
    return -(-int(samples) // TOKEN_HOP_SAMPLES)

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
    source_pcm_end: int = 0
    teacher_bindings: tuple[object, ...] = ()
    commit_key: str | None = None
    commit_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in POOL_FAMILIES:
            raise ValueError(f"unknown p2st task family {self.family!r}")
        if not (
            len(self.token_ids)
            == len(self.loss_kinds)
            == len(self.speech_indices)
        ):
            raise ValueError("token, loss and speech arrays must be parallel")
        if not self.token_ids:
            raise ValueError("a task sample cannot be empty")
        speech = [value for value in self.speech_indices if value is not None]
        if self.source_audio is None:
            if speech or self.source_glm_length or self.source_pcm_end:
                raise ValueError("a text sample must carry no acoustic sidecar")
            return
        # An acoustic sample promises the trainer three things at once: where
        # to cut the audio, how many tokens that cut yields, and that the
        # prompt binds exactly those positions in order.
        if self.source_pcm_end <= 0:
            raise ValueError("an acoustic sample needs a positive source_pcm_end")
        if speech != list(range(self.source_glm_length)):
            raise ValueError("speech indices must cover the GLM prefix in order")
        if len(self.source_glm_ids) != self.source_glm_length:
            raise ValueError("source_glm_ids must have one id per GLM position")

    @property
    def shifted_length(self) -> int:
        """Supervised length after the next-token shift.

        ``pack_task_samples`` reads this to lay samples out, so it has to
        mean exactly what it means on ``E2ETaskSample``.
        """
        return len(self.token_ids) - 1


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
    return _split_source_text(event, kind)[0]


def _split_running_text(prefix: str, delta: str) -> tuple[str, str]:
    """Split running text into committed prompt and supervised delta.

    The separator between the two belongs to neither field.  Measured on 3710
    English deltas in the 15-shard gold trajectories, ``gold_source_prefix``
    is running text -- ``'I completely'`` -- while ``gold_source_delta`` is the
    bare word -- ``'completely'`` -- and **0.0%** of deltas carry a leading
    space.  Cutting at ``len(prefix) - len(delta)`` and then stripping the
    committed side therefore deletes the space outright, and the pair the model
    sees is ``'I'`` -> ``'completely'``.

    That is what shipped in the first pool, and 400 training steps were enough
    to make it permanent: the cascade transcribed "I can't think what takes" as
    ``'Ithinksomethingwilltake'``.  Because the MT stage is prompted with
    ``gold_source_prefix``, which *does* have spaces, it then received an
    out-of-distribution string, which is the mechanism behind
    "Well I'm glad I can inspire you" being translated as 忘恩负义激发你.

    The separator has to be *supervised*, not injected by the prompt builder:
    at inference the running hypothesis is the concatenation of the model's own
    deltas, so a space the model never emits is a space that never appears.
    Hence the whitespace goes to the target side, which is also the natural BPE
    segmentation of running text.
    """
    if delta and prefix.endswith(delta):
        committed = prefix[: len(prefix) - len(delta)].rstrip()
        return committed, prefix[len(committed) :]
    return prefix, delta


def _split_source_text(event: TrajectoryEvent, kind: str) -> tuple[str, str]:
    """``(committed prompt text, supervised delta)`` for the ASR task."""
    if kind == SOURCE_PREFIX_GOLD:
        prefix, delta = event.gold_source_prefix, event.gold_source_delta
    elif kind == SOURCE_PREFIX_V1:
        prefix = event.v1_source_prefix or ""
        delta = event.v1_source_delta or ""
    else:
        raise ValueError(f"unknown source prefix kind {kind!r}")
    return _split_running_text(prefix, delta)


def _split_target_text(event: TrajectoryEvent) -> tuple[str, str]:
    """``(committed prompt text, supervised delta)`` for the MT and TTS tasks.

    The target side needs the same treatment as the source side, for the same
    reason: on cmn->eng the target text is English running text, so a dropped
    separator would break the translation the way it broke the transcript.
    """
    return _split_running_text(event.target_text_prefix, event.target_text_delta)


def _glm_ids_for_prefix(
    trajectory: E2ETrajectory, event: TrajectoryEvent, glm_stop: int
) -> tuple[int, ...]:
    """Recorded GLM ids for the prefix, resized to the frontend's count.

    Contents do not affect learning.  The trainer feeds the model
    ``embedding(causal_codes + offset) + bridge_residual`` computed from the
    waveform and uses ``glm_ids`` only to log
    ``diagnostic/causal_glm_agreement``, which has read about 0.001 for the
    whole of this lineage because the recorded codes come from the offline
    GLM-4 tokenizer while the model consumes the causal frontend's codes.
    What must be right is the length and the value range, so a prefix short of
    the frontend's count is extended by repeating its last recorded code
    rather than by inventing one.
    """
    recorded = [
        int(value)
        for item in trajectory.events
        if item.event_index <= event.event_index
        for value in item.source_glm_delta
    ]
    if len(recorded) >= glm_stop:
        return tuple(recorded[:glm_stop])
    filler = recorded[-1] if recorded else 0
    return tuple(recorded + [filler] * (glm_stop - len(recorded)))


def build_p2st_streaming_asr_tasks(
    trajectory: E2ETrajectory,
    *,
    task_token: int = TASK_TOKENS[FAMILY_P2ST_ASR],
    encode_text: EncodeText,
    glm_token_count: Callable[[int], int] = causal_glm_token_count,
) -> list[P2STTaskSample]:
    """One sample per event that transcribes new source text.

    Prompt: the causal source GLM prefix plus the transcript committed so far.
    Target: this event's transcript delta, then ``END_CONTENT`` and EOS.

    The GLM length comes from ``glm_token_count(event.source_pcm_end)``, not
    from ``event.source_glm_end`` -- see the module docstring for why the
    trajectory's own offset cannot be used.
    """
    output: list[P2STTaskSample] = []
    for event in trajectory.events:
        if not event.gold_source_delta.strip():
            continue
        prefix, delta_text = _split_source_text(event, SOURCE_PREFIX_GOLD)
        delta = tuple(int(v) for v in encode_text(delta_text))
        if not delta:
            continue
        pcm_end = int(event.source_pcm_end)
        glm_stop = int(glm_token_count(pcm_end))
        if glm_stop <= 0 or pcm_end <= 0:
            continue
        builder = _Builder()
        # Match Stage-A's own layout around the supervised span.  That pool is
        # what trained TOKEN_TASK_STREAMING_ASR, and its sample is
        #   [TASK_STREAMING_ASR, STREAMING_MODE, lang, *speaker]
        #   ... START_GLM glm END_GLM WRITE_GENERATE lang START_CONTENT delta
        # Three pieces were missing here, and the middle one is the likely
        # cause of a measured failure: the cascade transcribed English audio as
        # Chinese, and the only language cue was in the header, roughly ninety
        # GLM tokens away from where the text begins.
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
                *([c.glm_semantic_id(0)] * glm_stop),
                c.TOKEN_END_GLM,
            ],
            [None, *range(glm_stop), None],
        )
        committed = tuple(int(v) for v in encode_text(prefix)) if prefix else ()
        # WRITE_GENERATE is unconditional here and lives in the *prompt*: it is
        # never a supervised target and the model never chooses it, so the
        # property that matters -- that no WAIT-versus-WRITE decision is ever
        # generated -- is untouched.  Stage-A's ASR has no WAIT branch either;
        # every event with a text delta writes.
        builder.observe(
            [
                c.TOKEN_WRITE_GENERATE,
                c.language_token_id(trajectory.src_lang),
                c.TOKEN_START_CONTENT,
                *committed,
            ]
        )
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
                source_glm_ids=_glm_ids_for_prefix(trajectory, event, glm_stop),
                source_pcm_end=pcm_end,
            )
        )
    return output


def build_p2st_incremental_mt_tasks(
    trajectory: E2ETrajectory,
    *,
    task_token: int = TASK_TOKENS[FAMILY_P2ST_MT],
    encode_text: EncodeText,
    source_prefix_kind: str = SOURCE_PREFIX_GOLD,
) -> list[P2STTaskSample]:
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
    output: list[P2STTaskSample] = []
    for event in trajectory.events:
        if not event.target_text_delta.strip():
            continue
        target_prefix, delta_text = _split_target_text(event)
        delta = tuple(int(v) for v in encode_text(delta_text))
        if not delta:
            continue
        if source_prefix_kind == SOURCE_PREFIX_GOLD:
            source_text = event.gold_source_prefix
        else:
            source_text = event.v1_source_prefix or ""
        if not source_text.strip():
            continue
        builder = _Builder()
        # phase3's build_mt_sample, whose whole-utterance BLEU is 33-52, lays
        # this out as
        #   [TASK_T2T_TRANSLATION, lang, START_CONTENT src END_CONTENT,
        #    WRITE_GENERATE, lang, START_CONTENT] -> target
        # The WRITE_GENERATE and the repeated language token are the separator
        # between the two content blocks.  Without them the second
        # START_CONTENT reads as a continuation of the first, and the measured
        # consequence was the cascade copying the source instead of
        # translating it: "I think a" -> "I think a" on eng->cmn.
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


TEXT_SCOPE_PREFIX = "prefix"
TEXT_SCOPE_DELTA = "delta"


def build_p2st_streaming_tts_tasks(
    trajectory: E2ETrajectory,
    *,
    task_token: int = TASK_TOKENS[FAMILY_P2ST_TTS],
    encode_text: EncodeText,
    speed: float = 1.0,
    text_scope: str = TEXT_SCOPE_DELTA,
) -> list[P2STTaskSample]:
    """One sample per event that speaks a new target fragment.

    This is the builder the interleaved pool never had: ``task_samples.py``
    emits ``TOKEN_TASK_STREAMING_TTS`` nowhere, so the speech side has only
    ever been trained inside the interleaved sequence, where its terminator
    competes with WAIT/WRITE for the same boundary bucket.

    Prompt: the speaker identity, the target text for this fragment, and the
    semantic tokens already spoken.  Target: this event's semantic delta, then
    ``END_SEMANTIC`` and EOS.

    ``text_scope`` decides how much text the prompt shows, and it decides
    whether ``END_SEMANTIC`` is predictable at all.  With ``prefix`` the prompt
    carries every word committed so far while the target is only the newest
    fragment's codes, so "how much to speak" has to be inferred from how long
    the semantic prefix already is -- and the measured consequence was
    END_SEMANTIC never arriving: every TTS stage in the cascade ran to
    ``max_semantic_tokens`` exactly.  With ``delta`` the prompt carries just
    this fragment's words, which is what SpeakStream's Scheme 1 feeds, so the
    stopping point is determined by the prompt rather than inferred from it.
    The earlier speech is still in the prompt as the semantic prefix, so
    acoustic continuity is not lost.
    """
    output: list[P2STTaskSample] = []
    for event in trajectory.events:
        if not event.target_semantic_delta:
            continue
        if text_scope == TEXT_SCOPE_DELTA:
            # The separator-carrying form, because at inference this prompt is
            # filled with the delta the MT stage just generated, which does
            # carry it.  Only cmn->eng is affected -- Chinese has no spaces --
            # but a mismatch there would be invisible in the eng->cmn panel.
            fragment_text = _split_target_text(event)[1]
        elif text_scope == TEXT_SCOPE_PREFIX:
            fragment_text = event.target_text_prefix
        else:
            raise ValueError(f"unknown text scope {text_scope!r}")
        if not fragment_text.strip():
            continue
        spoken = _semantic_prefix(trajectory.events, event.event_index)
        if len(spoken) != int(event.target_semantic_start):
            raise ValueError(
                "semantic prefix length does not match target_semantic_start "
                f"for {trajectory.sample_id} event {event.event_index}: "
                f"{len(spoken)} != {event.target_semantic_start}"
            )
        builder = _Builder()
        # phase3's build_tts_sample, whose whole-utterance duration ratio is a
        # healthy 1.039, lays this out as
        #   [TASK_TTS, lang, *speaker, START_CONTENT text END_CONTENT,
        #    WRITE_GENERATE, lang, speed, START_SEMANTIC] -> semantic
        # so the speed token belongs after WRITE_GENERATE, not in the header,
        # and the separator is required here for the same reason as in MT.
        # The measured consequence of omitting it was END_SEMANTIC never
        # arriving: every TTS stage ran to max_semantic_tokens exactly.
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


def build_p2st_phase3_replay_tasks(
    trajectory: E2ETrajectory,
    *,
    encode_text: EncodeText,
) -> list[P2STTaskSample]:
    """The two whole-utterance phase3 replay samples for one trajectory.

    Delegates to the interleaved pool's ``build_phase3_replay_tasks`` and only
    re-wraps the result, so the replayed prompts are bit-identical to the ones
    the lineage has always trained on.  Both samples are text-only --
    ``source_audio`` is ``None`` and ``source_glm_length`` is 0 -- so they cost
    no audio read and take no part in the causal-GLM injection.
    """
    output: list[P2STTaskSample] = []
    for sample in build_phase3_replay_tasks(trajectory, encode_text=encode_text):
        if not sample.token_ids:
            continue
        if any(kind not in (LOSS_NONE, LOSS_REPLAY) for kind in sample.loss_kinds):
            raise ValueError("a replay sample may only carry replay supervision")
        output.append(
            P2STTaskSample(
                sample_id=sample.sample_id,
                sequence_id=sample.sequence_id,
                source_manifest_record=sample.source_manifest_record,
                family=sample.family,
                token_ids=tuple(sample.token_ids),
                loss_kinds=tuple(sample.loss_kinds),
                speech_indices=tuple(sample.speech_indices),
                source_audio=None,
                source_glm_length=0,
            )
        )
    return output


__all__ = [
    "P2STTaskSample",
    "TASK_TOKENS",
    "UNTRAINED_TASK_TOKENS",
    "TOKEN_HOP_SAMPLES",
    "causal_glm_token_count",
    "FAMILY_PHASE3_PERFORMANCE",
    "FAMILY_PHASE3_QUALITY",
    "POOL_FAMILIES",
    "REPLAY_FAMILIES",
    "build_p2st_phase3_replay_tasks",
    "FAMILY_P2ST_ASR",
    "FAMILY_P2ST_MT",
    "FAMILY_P2ST_TTS",
    "P2ST_FAMILIES",
    "SOURCE_PREFIX_GOLD",
    "TEXT_SCOPE_DELTA",
    "TEXT_SCOPE_PREFIX",
    "SOURCE_PREFIX_V1",
    "build_p2st_incremental_mt_tasks",
    "build_p2st_streaming_asr_tasks",
    "build_p2st_streaming_tts_tasks",
]
