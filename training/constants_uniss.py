"""UniSS token constants and small encoding helpers.

The IDs in this file are derived from the public UniSS tokenizer and the paper's
implementation details: the Qwen2.5 vocabulary is expanded to 180,407 tokens to
include BiCodec, GLM semantic, speed, language, and task/control tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence


VOCAB_SIZE = 180_407

QWEN_BASE_VOCAB_END = 151_664

BICODEC_GLOBAL_OFFSET = 151_665
BICODEC_GLOBAL_SIZE = 4_096

BICODEC_SEMANTIC_OFFSET = 155_761
BICODEC_SEMANTIC_SIZE = 8_192

GLM_SEMANTIC_OFFSET = 163_953
GLM_SEMANTIC_SIZE = 16_384

SPEED_OFFSET = 180_337
SPEED_SIZE = 35

TOKEN_PAD = 151_643
TOKEN_EOS = 151_645

TOKEN_CMN = 180_372
TOKEN_ENG = 180_373
TOKEN_YUE = 180_374

TOKEN_TASK_TTS = 180_375
TOKEN_TASK_ASR = 180_376
TOKEN_TASK_S2S = 180_377
TOKEN_TASK_T2S_TRANSLATION = 180_378
TOKEN_TASK_S2S_TRANSLATION = 180_379
TOKEN_TASK_S2T_TRANSLATION = 180_380
TOKEN_TASK_T2T_TRANSLATION = 180_381
TOKEN_TASK_STREAMING_TTS = 180_382
TOKEN_TASK_STREAMING_ASR = 180_383
TOKEN_TASK_STREAMING_S2ST = 180_384
TOKEN_TASK_STREAMING_S2TT = 180_385

TOKEN_START_CONTENT = 180_386
TOKEN_START_GLOBAL = 180_387
TOKEN_START_SEMANTIC = 180_388
TOKEN_START_GLM = 180_389
TOKEN_END_CONTENT = 180_390
TOKEN_END_GLOBAL = 180_391
TOKEN_END_SEMANTIC = 180_392
TOKEN_END_GLM = 180_393
TOKEN_GLOBAL_PADDING = 180_394
TOKEN_WAIT_READ = 180_395
TOKEN_WRITE_GENERATE = 180_396
TOKEN_TASK_TEXT_TRANSLATION = 180_397
TOKEN_TASK_STREAMING_TEXT_TRANSLATION = 180_398

TOKEN_SLOW_MODE = 180_399
TOKEN_BALANCE_MODE = 180_400
TOKEN_FAST_MODE = 180_401
TOKEN_DYNAMIC_MODE = 180_402
TOKEN_AUDIO_PLACEHOLDER_1 = 180_403
TOKEN_AUDIO_PLACEHOLDER_2 = 180_404
TOKEN_AUDIO_PLACEHOLDER_3 = 180_405
TOKEN_STREAMING_MODE = 180_406

SUPPORTED_LANGUAGES = {"eng", "cmn"}

_LANGUAGE_ALIASES = {
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "en_xx": "eng",
    "en-xx": "eng",
    "zh": "cmn",
    "zh_cn": "cmn",
    "zh-cn": "cmn",
    "cn": "cmn",
    "cmn": "cmn",
    "mandarin": "cmn",
    "chinese": "cmn",
}

LANGUAGE_TOKEN_IDS = {
    "eng": TOKEN_ENG,
    "cmn": TOKEN_CMN,
}

TASK_TOKEN_IDS = {
    "tts": TOKEN_TASK_TTS,
    "asr": TOKEN_TASK_ASR,
    "s2s": TOKEN_TASK_S2S,
    "s2s_translation": TOKEN_TASK_S2S_TRANSLATION,
    "s2t_translation": TOKEN_TASK_S2T_TRANSLATION,
    "t2t_translation": TOKEN_TASK_T2T_TRANSLATION,
}

MODE_TOKEN_IDS = {
    "slow": TOKEN_SLOW_MODE,
    "quality": TOKEN_SLOW_MODE,
    "balance": TOKEN_BALANCE_MODE,
    "performance": TOKEN_BALANCE_MODE,
    "fast": TOKEN_FAST_MODE,
    "direct": TOKEN_FAST_MODE,
}


@dataclass(frozen=True)
class TokenSpan:
    name: str
    offset: int
    size: int

    @property
    def last_id(self) -> int:
        return self.offset + self.size - 1

    def id_for(self, value: int) -> int:
        validate_range(value, self.size, self.name)
        return self.offset + value

    def value_for(self, token_id: int) -> int:
        if not self.offset <= token_id <= self.last_id:
            raise ValueError(
                f"{token_id} is outside {self.name} token id span "
                f"[{self.offset}, {self.last_id}]"
            )
        return token_id - self.offset


BICODEC_GLOBAL_SPAN = TokenSpan(
    "bicodec_global", BICODEC_GLOBAL_OFFSET, BICODEC_GLOBAL_SIZE
)
BICODEC_SEMANTIC_SPAN = TokenSpan(
    "bicodec_semantic", BICODEC_SEMANTIC_OFFSET, BICODEC_SEMANTIC_SIZE
)
GLM_SEMANTIC_SPAN = TokenSpan("glm_semantic", GLM_SEMANTIC_OFFSET, GLM_SEMANTIC_SIZE)
SPEED_SPAN = TokenSpan("speed", SPEED_OFFSET, SPEED_SIZE)


def validate_range(value: int, size: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} token value must be int, got {type(value).__name__}")
    if value < 0 or value >= size:
        raise ValueError(f"{name} token value {value} is outside [0, {size - 1}]")


def normalize_language(language: str) -> Literal["eng", "cmn"]:
    key = language.strip().lower().replace(" ", "_")
    try:
        return _LANGUAGE_ALIASES[key]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported language {language!r}; expected one of {sorted(SUPPORTED_LANGUAGES)}"
        ) from exc


def language_token_id(language: str) -> int:
    return LANGUAGE_TOKEN_IDS[normalize_language(language)]


def opposite_language(language: str) -> Literal["eng", "cmn"]:
    normalized = normalize_language(language)
    return "cmn" if normalized == "eng" else "eng"


def speed_to_index(speed: float) -> int:
    """Convert speed ratio to UniSS speed token index.

    Current inference code uses int((speed - 0.1) / 0.1). This helper preserves
    that behavior while guarding against float precision around one decimal.
    """

    index = int(round((speed - 0.1) / 0.1))
    validate_range(index, SPEED_SIZE, "speed")
    return index


def speed_token_id(speed: float = 1.0) -> int:
    return SPEED_SPAN.id_for(speed_to_index(speed))


def bicodec_global_id(value: int) -> int:
    return BICODEC_GLOBAL_SPAN.id_for(value)


def bicodec_semantic_id(value: int) -> int:
    return BICODEC_SEMANTIC_SPAN.id_for(value)


def glm_semantic_id(value: int) -> int:
    return GLM_SEMANTIC_SPAN.id_for(value)


def encode_bicodec_global(values: Sequence[int]) -> list[int]:
    if len(values) != 32:
        raise ValueError(f"bicodec global token list must have length 32, got {len(values)}")
    return [bicodec_global_id(value) for value in values]


def encode_bicodec_semantic(values: Iterable[int]) -> list[int]:
    return [bicodec_semantic_id(value) for value in values]


def encode_glm_semantic(values: Iterable[int]) -> list[int]:
    return [glm_semantic_id(value) for value in values]


def wrap_global_tokens(values: Sequence[int]) -> list[int]:
    return [TOKEN_START_GLOBAL, *encode_bicodec_global(values), TOKEN_END_GLOBAL]


def wrap_semantic_tokens(values: Iterable[int], include_eos: bool = True) -> list[int]:
    ids = [TOKEN_START_SEMANTIC, *encode_bicodec_semantic(values), TOKEN_END_SEMANTIC]
    if include_eos:
        ids.append(TOKEN_EOS)
    return ids


def validate_token_id(token_id: int) -> None:
    validate_range(token_id, VOCAB_SIZE, "vocab")
