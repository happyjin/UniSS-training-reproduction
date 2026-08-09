"""Token-level sample builders for the full198 streaming curriculum."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from training import constants_uniss as c


@dataclass(frozen=True)
class TokenSample:
    prompt_ids: list[int]
    target_ids: list[int]
    task: str

    @property
    def input_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.target_ids]


def _ints(record: Mapping[str, object], key: str) -> list[int]:
    values = record[key]
    if not isinstance(values, list) or not values or not all(isinstance(value, int) for value in values):
        raise ValueError(f"record field {key} must be a non-empty list of ints")
    return [int(value) for value in values]


def _lang(record: Mapping[str, object], key: str) -> str:
    return str(record[key])


def source_prefix(record: Mapping[str, object], ratio: float) -> list[int]:
    source = _ints(record, "source_glm")
    length = max(1, min(len(source), int(math.ceil(float(ratio) * len(source)))))
    return source[:length]


def _source_header(task_tokens: Sequence[int], record: Mapping[str, object], source: Sequence[int]) -> list[int]:
    return [
        *task_tokens,
        c.language_token_id(_lang(record, "tgt_lang")),
        *c.wrap_global_tokens(_ints(record, "bicodec_global")),
        *c.encode_glm_semantic(source),
    ]


def build_streaming_s2tt(record: Mapping[str, object], ratio: float) -> TokenSample:
    prompt = [
        *_source_header(
            [c.TOKEN_TASK_STREAMING_S2TT, c.TOKEN_STREAMING_MODE],
            record,
            source_prefix(record, ratio),
        ),
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(_lang(record, "tgt_lang")),
        c.TOKEN_START_CONTENT,
    ]
    target = [*_ints(record, "translation_ids"), c.TOKEN_END_CONTENT, c.TOKEN_EOS]
    return TokenSample(prompt, target, "prefix")


def build_teacher_s2tt(record: Mapping[str, object]) -> TokenSample:
    prompt = [
        *_source_header([c.TOKEN_TASK_S2T_TRANSLATION], record, _ints(record, "source_glm")),
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(_lang(record, "tgt_lang")),
        c.TOKEN_START_CONTENT,
    ]
    target = [*_ints(record, "translation_ids"), c.TOKEN_END_CONTENT, c.TOKEN_EOS]
    return TokenSample(prompt, target, "teacher_s2tt")


def build_action_prompt(record: Mapping[str, object], ratio: float) -> list[int]:
    return _source_header(
        [c.TOKEN_TASK_STREAMING_S2ST, c.TOKEN_STREAMING_MODE, c.TOKEN_DYNAMIC_MODE],
        record,
        source_prefix(record, ratio),
    )


def build_streaming_tts(
    record: Mapping[str, object],
    *,
    text_ratio: float,
    semantic_cut: int,
    block_size: int,
    history_tokens: int = 200,
) -> TokenSample:
    text = _ints(record, "translation_ids")
    text_end = max(1, min(len(text), int(math.ceil(text_ratio * len(text)))))
    semantic = _ints(record, "target_bicodec")
    semantic_cut = max(1, min(len(semantic) - 1, semantic_cut))
    block_size = max(1, min(block_size, len(semantic) - semantic_cut))
    history = semantic[max(0, semantic_cut - history_tokens) : semantic_cut]
    prompt = [
        c.TOKEN_TASK_STREAMING_TTS,
        c.TOKEN_STREAMING_MODE,
        c.language_token_id(_lang(record, "tgt_lang")),
        *c.wrap_global_tokens(_ints(record, "bicodec_global")),
        c.TOKEN_START_CONTENT,
        *text[:text_end],
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_SEMANTIC,
        *c.encode_bicodec_semantic(history),
        c.TOKEN_END_SEMANTIC,
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_START_SEMANTIC,
    ]
    target = [
        *c.encode_bicodec_semantic(semantic[semantic_cut : semantic_cut + block_size]),
        c.TOKEN_END_SEMANTIC,
        c.TOKEN_EOS,
    ]
    return TokenSample(prompt, target, "semantic")


def build_teacher_tts(
    record: Mapping[str, object], *, semantic_cut: int, block_size: int, history_tokens: int = 200
) -> TokenSample:
    semantic = _ints(record, "target_bicodec")
    semantic_cut = max(1, min(len(semantic) - 1, semantic_cut))
    block_size = max(1, min(block_size, len(semantic) - semantic_cut))
    history = semantic[max(0, semantic_cut - history_tokens) : semantic_cut]
    prompt = [
        c.TOKEN_TASK_TTS,
        c.language_token_id(_lang(record, "tgt_lang")),
        *c.wrap_global_tokens(_ints(record, "bicodec_global")),
        c.TOKEN_START_CONTENT,
        *_ints(record, "translation_ids"),
        c.TOKEN_END_CONTENT,
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(_lang(record, "tgt_lang")),
        c.speed_token_id(1.0),
        c.TOKEN_START_SEMANTIC,
        *c.encode_bicodec_semantic(history),
    ]
    target = [
        *c.encode_bicodec_semantic(semantic[semantic_cut : semantic_cut + block_size]),
        c.TOKEN_END_SEMANTIC,
        c.TOKEN_EOS,
    ]
    return TokenSample(prompt, target, "teacher_tts")


def build_replay(record: Mapping[str, object], mode: str) -> TokenSample:
    source = _ints(record, "source_glm")
    target_semantic = c.encode_bicodec_semantic(_ints(record, "target_bicodec"))
    tgt_lang = _lang(record, "tgt_lang")
    if mode == "performance":
        prompt = [
            *_source_header(
                [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_BALANCE_MODE], record, source
            ),
            c.TOKEN_WRITE_GENERATE,
            c.TOKEN_TASK_S2T_TRANSLATION,
            c.language_token_id(tgt_lang),
            c.speed_token_id(1.0),
            c.TOKEN_START_CONTENT,
        ]
        target = [
            *_ints(record, "translation_ids"),
            c.TOKEN_END_CONTENT,
            c.TOKEN_START_SEMANTIC,
            *target_semantic,
            c.TOKEN_END_SEMANTIC,
            c.TOKEN_EOS,
        ]
        return TokenSample(prompt, target, "replay_performance")
    if mode != "quality":
        raise ValueError(f"unsupported replay mode: {mode}")
    prompt = [
        *_source_header([c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_SLOW_MODE], record, source),
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_TASK_ASR,
        c.language_token_id(_lang(record, "src_lang")),
        c.speed_token_id(1.0),
        c.TOKEN_START_CONTENT,
    ]
    target = [
        *_ints(record, "transcription_ids"),
        c.TOKEN_END_CONTENT,
        c.TOKEN_TASK_S2T_TRANSLATION,
        c.language_token_id(tgt_lang),
        c.speed_token_id(1.0),
        c.TOKEN_START_CONTENT,
        *_ints(record, "translation_ids"),
        c.TOKEN_END_CONTENT,
        c.TOKEN_START_SEMANTIC,
        *target_semantic,
        c.TOKEN_END_SEMANTIC,
        c.TOKEN_EOS,
    ]
    return TokenSample(prompt, target, "replay_quality")

