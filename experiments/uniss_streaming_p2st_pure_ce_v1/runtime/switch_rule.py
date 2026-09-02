"""The external switch rule: what runs next, decided without a logit.

This is the whole of what C moves out of the model.  Four training runs tried
to teach the interleaved model when to speak and all four moved their own loss
correctly while leaving the inference decision where it was; a bias sweep then
showed the decision is bimodal rather than calibratable -- delta=1 flipped none
of 181 post-fragment decisions and delta=2 flipped 58 of 218, stepping straight
over the target band.  So the decision is not learned here and not calibrated
here.  It is this function, and it consults nothing but how much *committed*
content each stage produced.

The rule is prefix-to-prefix with bounded waiting, the shape CSSEL-P2P and
SpeakStream's Scheme 1 use: transcribe every new block; translate only when the
transcript grew; speak only when the translation grew; otherwise read more
audio.  "Grew" means the local-agreement committer released new tokens, not
that the model emitted something -- an unstable hypothesis is not new content.

Kept as a pure function of three integers so it can be exercised exhaustively
without a model, which is the only way to be sure the rule itself is not where
a bug hides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

TASK_ASR: Final = "asr"
TASK_MT: Final = "mt"
TASK_TTS: Final = "tts"
TASK_READ: Final = "read"
TASK_DONE: Final = "done"

STAGE_ORDER: Final = (TASK_ASR, TASK_MT, TASK_TTS)


@dataclass(frozen=True)
class SwitchState:
    """What the rule is allowed to look at.

    ``stage`` is how far this block has got through ASR -> MT -> TTS.  The
    deltas are token counts the committer released in this block, so zero means
    "nothing safe to pass on", not "the model said nothing".
    """

    stage: str
    source_delta: int
    target_delta: int
    source_exhausted: bool

    def __post_init__(self) -> None:
        if self.stage not in (*STAGE_ORDER, TASK_READ, TASK_DONE):
            raise ValueError(f"unknown cascade stage {self.stage!r}")
        if self.source_delta < 0 or self.target_delta < 0:
            raise ValueError("committed deltas cannot be negative")


def next_task(state: SwitchState) -> str:
    """The next task to run, or ``read``/``done``.

    Reading the branches in order is the whole policy:

    * at the start of a block there is new audio, so transcribe it;
    * a transcript that did not grow means nothing new was heard reliably, so
      go back for more audio rather than re-translating the same prefix;
    * a translation that did not grow means nothing is safe to say yet -- this
      is the bounded wait, and it is the branch the model used to have to make;
    * once the translation grew, speak exactly that much and no more.
    """
    if state.stage == TASK_DONE:
        return TASK_DONE
    if state.stage == TASK_READ:
        return TASK_DONE if state.source_exhausted else TASK_ASR
    if state.stage == TASK_ASR:
        if state.source_delta <= 0:
            return TASK_DONE if state.source_exhausted else TASK_READ
        return TASK_MT
    if state.stage == TASK_MT:
        if state.target_delta <= 0:
            return TASK_DONE if state.source_exhausted else TASK_READ
        return TASK_TTS
    # After speaking, a block is finished whether or not more audio remains.
    return TASK_DONE if state.source_exhausted else TASK_READ


def rule_trace(
    source_deltas: list[int], target_deltas: list[int], *, blocks: int
) -> list[str]:
    """Replay the rule over recorded deltas, for tests and for auditing a run.

    Takes the deltas a real session observed and returns the task sequence the
    rule would produce, so a run's own trace can be checked against the rule
    rather than trusted.
    """
    if len(source_deltas) != blocks or len(target_deltas) != blocks:
        raise ValueError("one source and target delta per block is required")
    output: list[str] = []
    for index in range(blocks):
        exhausted = index == blocks - 1
        stage = TASK_ASR
        while True:
            output.append(stage)
            state = SwitchState(
                stage=stage,
                source_delta=source_deltas[index],
                target_delta=target_deltas[index],
                source_exhausted=exhausted,
            )
            stage = next_task(state)
            if stage in (TASK_READ, TASK_DONE):
                break
    return output


__all__ = [
    "STAGE_ORDER",
    "TASK_ASR",
    "TASK_DONE",
    "TASK_MT",
    "TASK_READ",
    "TASK_TTS",
    "SwitchState",
    "next_task",
    "rule_trace",
]
