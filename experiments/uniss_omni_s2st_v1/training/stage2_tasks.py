"""The five Stage-2 tasks, and what each one supervises.

The mixture and its ratio come from arXiv 2607.19810 §4.1:
``ASR : S2TT : MT : TTS : S2ST = 0.2 : 1 : 0.5 : 1 : 1.5``. The S2ST task
uses **Quality mode only** -- transcript, then translation, then codes --
decided on 2026-10-10 and argued in ``OMNI_REPLACEMENT_PLAN.zh-CN.md``.

Each task says three things: what the Thinker sees, what text it is
supervised on, and whether the Talker is supervised at all.

    task    Thinker input          text target              code target
    ----    -------------          -----------              -----------
    ASR     audio                  source transcript        --
    S2TT    audio                  target text              --
    MT      source text            target text              --
    TTS     target text            **none**                 target codes
    S2ST    audio                  transcript + target text target codes

TTS carries no text loss on purpose. Its prompt already contains the text
the assistant would "produce", so supervising it would train the Thinker
to copy its own input -- harmless while the Thinker was frozen in Stage 1,
actively wrong now that it learns. The Talker still needs those hidden
states, so the forward runs; only the text term is dropped.
"""

from __future__ import annotations

import os

from dataclasses import dataclass

LANGUAGE = {"cmn": "Chinese", "eng": "English", "zh": "Chinese", "en": "English"}

SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
    " capable of perceiving auditory and visual inputs, as well as generating"
    " text and speech."
)

# The ratio the paper trains at. Keys are the task names below.
MIXTURE = {"asr": 0.2, "s2tt": 1.0, "mt": 0.5, "tts": 1.0, "s2st": 1.5}

# Which of the Thinker's projections the Stage 2 LoRA wraps. It lives here
# rather than beside the model because evaluation has to inject the same
# four to load the adapter, and this module is the one with no Megatron
# import for an eval environment to trip over.
THINKER_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class TaskSample:
    """One training example, before tokenisation."""

    task: str
    audio_path: str | None          # a wds:// URI, or None for text-only tasks
    user_text: str                  # the instruction, and any text input
    text_target: str | None         # None when the task has no text loss
    codes: object | None            # target BiCodec codes, or None
    globals: object | None          # the 32 speaker tokens, or None


# The reply shape the s2st task supervises. "transcript+translation" is
# what stage 2 and phase E v1 trained under; it cost phase E its end-to-end
# score, because inference asks for the translation alone and the talker
# had never read a reply without a foreign transcript in front of it.
# Left unset nothing changes, so stage 2 and every finished run keep the
# behaviour they were measured with.
S2ST_REPLY = os.environ.get("UNISS_S2ST_REPLY", "transcript+translation")


def _instruction(task: str, src_lang: str, tgt_lang: str, text: str | None) -> str:
    source = LANGUAGE.get(src_lang, src_lang)
    target = LANGUAGE.get(tgt_lang, tgt_lang)
    if task == "asr":
        return f"Listen to the {source} speech and transcribe it. Reply with the {source} transcription only."
    if task == "a2s":
        # Same conditioning as asr -- audio in, transcript out -- but the
        # talker speaks that transcript instead of the task ending at text.
        # The point is the state family: this is the only task that puts the
        # talker on audio-conditioned states with a target it can reach, so
        # it can learn to read them before s2st asks it to translate too.
        return f"Listen to the {source} speech and transcribe it. Reply with the {source} transcription only."
    if task == "s2tt":
        return f"Listen to the {source} speech and translate it into {target}. Reply with the {target} translation only."
    if task == "mt":
        return f"Translate the following {source} text into {target}. Reply with the {target} translation only.\n{text}"
    if task == "tts":
        return f"Read this {target} text aloud: {text}"
    if task == "s2st":
        if S2ST_REPLY == "translation":
            # Word for word the s2tt instruction, which is what the e2e
            # probe sends. Identical prompt, identical reply shape.
            return (f"Listen to the {source} speech and translate it into"
                    f" {target}. Reply with the {target} translation only.")
        return (
            f"Listen to the {source} speech, transcribe it, translate it into"
            f" {target}, and speak the translation."
        )
    raise ValueError(f"unknown task {task!r}")


def build(task: str, row: dict) -> TaskSample:
    """Turn one corpus row into the sample a given task needs.

    ``row`` carries ``source_transcript``, ``target_text``, ``source_lang``,
    ``target_lang``, optionally ``audio`` (a wds:// URI), ``codes`` and
    ``globals``.
    """
    src, tgt = row["source_lang"], row["target_lang"]
    transcript = (row.get("source_transcript") or "").strip()
    translation = (row.get("target_text") or "").strip()

    if task == "asr":
        return TaskSample(task, row["audio"], _instruction(task, src, tgt, None),
                          transcript, None, None)
    if task == "a2s":
        if row.get("source_codes") is None:
            raise SystemExit(
                "a2s needs source_codes; build ParquetCodeIndex(with_source=True)")
        return TaskSample(task, row["audio"], _instruction(task, src, tgt, None),
                          transcript, row["source_codes"], row["globals"])
    if task == "s2tt":
        return TaskSample(task, row["audio"], _instruction(task, src, tgt, None),
                          translation, None, None)
    if task == "mt":
        return TaskSample(task, None, _instruction(task, src, tgt, transcript),
                          translation, None, None)
    if task == "tts":
        return TaskSample(task, None, _instruction(task, src, tgt, translation),
                          None, row["codes"], row["globals"])
    if task == "s2st":
        # Quality mode: the transcript precedes the translation, so the
        # model plans through what was said before what to say. That plan
        # is only worth its cost if inference reproduces it -- phase E v1
        # did not, and scored 12.06 where the same weights scored 26.27 on
        # the reply they were trained on.
        reply = translation
        if transcript and S2ST_REPLY == "transcript+translation":
            reply = f"{transcript}\n{translation}"
        return TaskSample(
            task, row["audio"], _instruction(task, src, tgt, None),
            reply, row["codes"], row["globals"],
        )
    raise ValueError(f"unknown task {task!r}")


def conversation(sample: TaskSample) -> list[dict]:
    """The chat-template messages for a sample."""
    content: list[dict] = []
    if sample.audio_path is not None:
        content.append({"type": "audio", "audio": "x.wav"})
    content.append({"type": "text", "text": sample.user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]


def needs_audio(task: str) -> bool:
    return task in ("asr", "s2tt", "s2st", "a2s")


def uses_tts_cache(task: str) -> bool:
    """Which pool a task draws from, which is not the same question.

    MT needs no waveform but does need the source transcript and the
    language pair, and the TTS cache carries neither -- it holds token ids
    and codes, built so the TTS task never re-tokenises. So MT draws from
    the manifest pool and simply does not load the audio.
    """
    return task == "tts"


def supervises_text(task: str) -> bool:
    return task != "tts"


def supervises_codes(task: str) -> bool:
    return task in ("tts", "s2st", "a2s")
