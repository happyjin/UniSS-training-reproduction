"""Build UniSS task samples as token-id sequences.

These builders implement the prompt/target layouts described in the UniSS paper
and mirrored by ``uniss/cli/prompt.py`` for inference. Text tokenization is
injected so this module can be tested without loading a Hugging Face tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from training import constants_uniss as c


TextEncoder = Callable[[str], list[int]]


@dataclass(frozen=True)
class TrainingSample:
    task: str
    prompt_ids: list[int]
    target_ids: list[int]
    segment_spans: dict[str, tuple[int, int]]
    source_id: str | None = None

    @property
    def input_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.target_ids]

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_ids)

    @property
    def target_length(self) -> int:
        return len(self.target_ids)


def _segment_spans(segments: Sequence[tuple[str, Sequence[int]]]) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, ids in segments:
        start = offset
        offset += len(ids)
        spans[name] = (start, offset)
    return spans


def _text_ids(text: str, text_encoder: TextEncoder, field_name: str) -> list[int]:
    if text is None or text == "":
        raise ValueError(f"{field_name} must be non-empty")
    ids = text_encoder(text)
    if not ids:
        raise ValueError(f"{field_name} encoded to an empty token list")
    for token_id in ids:
        c.validate_token_id(token_id)
    return ids


def _source_speech_prompt(
    task_tokens: Sequence[int],
    language: str,
    bicodec_global: Sequence[int],
    source_glm: Iterable[int],
) -> list[int]:
    return [
        *task_tokens,
        c.language_token_id(language),
        *c.wrap_global_tokens(bicodec_global),
        *c.encode_glm_semantic(source_glm),
    ]


def build_asr_sample(
    *,
    source_glm: Sequence[int],
    bicodec_global: Sequence[int],
    src_lang: str,
    transcription: str,
    text_encoder: TextEncoder,
    source_id: str | None = None,
) -> TrainingSample:
    prompt = [
        *_source_speech_prompt([c.TOKEN_TASK_ASR], src_lang, bicodec_global, source_glm),
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(src_lang),
        c.TOKEN_START_CONTENT,
    ]
    text = _text_ids(transcription, text_encoder, "transcription")
    target = [*text, c.TOKEN_END_CONTENT, c.TOKEN_EOS]
    segments = [("transcription_text", text), ("end_content", [c.TOKEN_END_CONTENT]), ("eos", [c.TOKEN_EOS])]
    return TrainingSample("asr", prompt, target, _segment_spans(segments), source_id)


def build_s2tt_sample(
    *,
    source_glm: Sequence[int],
    bicodec_global: Sequence[int],
    tgt_lang: str,
    translation: str,
    text_encoder: TextEncoder,
    source_id: str | None = None,
) -> TrainingSample:
    prompt = [
        *_source_speech_prompt(
            [c.TOKEN_TASK_S2T_TRANSLATION], tgt_lang, bicodec_global, source_glm
        ),
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(tgt_lang),
        c.TOKEN_START_CONTENT,
    ]
    text = _text_ids(translation, text_encoder, "translation")
    target = [*text, c.TOKEN_END_CONTENT, c.TOKEN_EOS]
    segments = [("translation_text", text), ("end_content", [c.TOKEN_END_CONTENT]), ("eos", [c.TOKEN_EOS])]
    return TrainingSample("s2tt", prompt, target, _segment_spans(segments), source_id)


def build_tts_sample(
    *,
    bicodec_global: Sequence[int],
    src_lang: str,
    transcription: str,
    source_bicodec: Sequence[int],
    text_encoder: TextEncoder,
    speed: float = 1.0,
    source_id: str | None = None,
) -> TrainingSample:
    text = _text_ids(transcription, text_encoder, "transcription")
    semantic = c.encode_bicodec_semantic(source_bicodec)
    prompt = [
        c.TOKEN_TASK_TTS,
        c.language_token_id(src_lang),
        *c.wrap_global_tokens(bicodec_global),
        c.TOKEN_START_CONTENT,
        *text,
        c.TOKEN_END_CONTENT,
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(src_lang),
        c.speed_token_id(speed),
        c.TOKEN_START_SEMANTIC,
    ]
    target = [*semantic, c.TOKEN_END_SEMANTIC, c.TOKEN_EOS]
    segments = [
        ("source_semantic", semantic),
        ("end_semantic", [c.TOKEN_END_SEMANTIC]),
        ("eos", [c.TOKEN_EOS]),
    ]
    return TrainingSample("tts", prompt, target, _segment_spans(segments), source_id)


def build_mt_sample(
    *,
    src_lang: str,
    tgt_lang: str,
    source_text: str,
    target_text: str,
    text_encoder: TextEncoder,
    source_id: str | None = None,
) -> TrainingSample:
    src_text = _text_ids(source_text, text_encoder, "source_text")
    tgt_text = _text_ids(target_text, text_encoder, "target_text")
    prompt = [
        c.TOKEN_TASK_T2T_TRANSLATION,
        c.language_token_id(tgt_lang),
        c.TOKEN_START_CONTENT,
        *src_text,
        c.TOKEN_END_CONTENT,
        c.TOKEN_WRITE_GENERATE,
        c.language_token_id(tgt_lang),
        c.TOKEN_START_CONTENT,
    ]
    target = [*tgt_text, c.TOKEN_END_CONTENT, c.TOKEN_EOS]
    segments = [("target_text", tgt_text), ("end_content", [c.TOKEN_END_CONTENT]), ("eos", [c.TOKEN_EOS])]
    return TrainingSample("mt", prompt, target, _segment_spans(segments), source_id)


def build_quality_sample(
    *,
    source_glm: Sequence[int],
    bicodec_global: Sequence[int],
    src_lang: str,
    tgt_lang: str,
    transcription: str,
    translation: str,
    target_bicodec: Sequence[int],
    text_encoder: TextEncoder,
    speed: float = 1.0,
    source_id: str | None = None,
) -> TrainingSample:
    transcription_ids = _text_ids(transcription, text_encoder, "transcription")
    translation_ids = _text_ids(translation, text_encoder, "translation")
    semantic_ids = c.encode_bicodec_semantic(target_bicodec)
    prompt = [
        *_source_speech_prompt(
            [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_SLOW_MODE],
            tgt_lang,
            bicodec_global,
            source_glm,
        ),
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_TASK_ASR,
        c.language_token_id(src_lang),
        c.speed_token_id(speed),
        c.TOKEN_START_CONTENT,
    ]
    target_segments = [
        ("quality_transcription_text", transcription_ids),
        ("quality_transcription_end", [c.TOKEN_END_CONTENT]),
        (
            "quality_translation_prompt",
            [
                c.TOKEN_TASK_S2T_TRANSLATION,
                c.language_token_id(tgt_lang),
                c.speed_token_id(speed),
                c.TOKEN_START_CONTENT,
            ],
        ),
        ("quality_translation_text", translation_ids),
        ("quality_translation_end", [c.TOKEN_END_CONTENT]),
        ("quality_semantic_start", [c.TOKEN_START_SEMANTIC]),
        ("quality_semantic", semantic_ids),
        ("quality_semantic_end", [c.TOKEN_END_SEMANTIC]),
        ("eos", [c.TOKEN_EOS]),
    ]
    target = [token for _, ids in target_segments for token in ids]
    return TrainingSample("quality", prompt, target, _segment_spans(target_segments), source_id)


def build_performance_sample(
    *,
    source_glm: Sequence[int],
    bicodec_global: Sequence[int],
    tgt_lang: str,
    translation: str,
    target_bicodec: Sequence[int],
    text_encoder: TextEncoder,
    speed: float = 1.0,
    source_id: str | None = None,
) -> TrainingSample:
    translation_ids = _text_ids(translation, text_encoder, "translation")
    semantic_ids = c.encode_bicodec_semantic(target_bicodec)
    prompt = [
        *_source_speech_prompt(
            [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_BALANCE_MODE],
            tgt_lang,
            bicodec_global,
            source_glm,
        ),
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_TASK_S2T_TRANSLATION,
        c.language_token_id(tgt_lang),
        c.speed_token_id(speed),
        c.TOKEN_START_CONTENT,
    ]
    target_segments = [
        ("performance_translation_text", translation_ids),
        ("performance_translation_end", [c.TOKEN_END_CONTENT]),
        ("performance_semantic_start", [c.TOKEN_START_SEMANTIC]),
        ("performance_semantic", semantic_ids),
        ("performance_semantic_end", [c.TOKEN_END_SEMANTIC]),
        ("eos", [c.TOKEN_EOS]),
    ]
    target = [token for _, ids in target_segments for token in ids]
    return TrainingSample(
        "performance", prompt, target, _segment_spans(target_segments), source_id
    )


def build_direct_s2st_sample(
    *,
    source_glm: Sequence[int],
    bicodec_global: Sequence[int],
    tgt_lang: str,
    target_bicodec: Sequence[int],
    speed: float = 1.0,
    source_id: str | None = None,
) -> TrainingSample:
    semantic_ids = c.encode_bicodec_semantic(target_bicodec)
    prompt = [
        *_source_speech_prompt(
            [c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_FAST_MODE],
            tgt_lang,
            bicodec_global,
            source_glm,
        ),
        c.TOKEN_WRITE_GENERATE,
        c.TOKEN_FAST_MODE,
        c.language_token_id(tgt_lang),
        c.speed_token_id(speed),
        c.TOKEN_START_SEMANTIC,
    ]
    target = [*semantic_ids, c.TOKEN_END_SEMANTIC, c.TOKEN_EOS]
    segments = [
        ("direct_semantic", semantic_ids),
        ("end_semantic", [c.TOKEN_END_SEMANTIC]),
        ("eos", [c.TOKEN_EOS]),
    ]
    return TrainingSample("direct_s2st", prompt, target, _segment_spans(segments), source_id)


def build_phase1_samples_from_record(
    record: Mapping[str, object],
    text_encoder: TextEncoder,
) -> list[TrainingSample]:
    source_id = str(record.get("id", "")) or None
    src_lang = str(record["src_lang"])
    tgt_lang = str(record["tgt_lang"])
    return [
        build_asr_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            src_lang=src_lang,
            transcription=str(record["transcription"]),
            text_encoder=text_encoder,
            source_id=source_id,
        ),
        build_s2tt_sample(
            source_glm=record["source_glm"],  # type: ignore[arg-type]
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            tgt_lang=tgt_lang,
            translation=str(record["translation"]),
            text_encoder=text_encoder,
            source_id=source_id,
        ),
        build_tts_sample(
            bicodec_global=record["bicodec_global"],  # type: ignore[arg-type]
            src_lang=src_lang,
            transcription=str(record["transcription"]),
            source_bicodec=record["source_bicodec"],  # type: ignore[arg-type]
            text_encoder=text_encoder,
            source_id=source_id,
        ),
    ]


def build_s2st_samples_from_record(
    record: Mapping[str, object],
    text_encoder: TextEncoder,
    include_direct: bool,
) -> list[TrainingSample]:
    source_id = str(record.get("id", "")) or None
    common = {
        "source_glm": record["source_glm"],
        "bicodec_global": record["bicodec_global"],
        "tgt_lang": str(record["tgt_lang"]),
        "target_bicodec": record["target_bicodec"],
        "source_id": source_id,
    }
    samples = [
        build_quality_sample(
            **common,  # type: ignore[arg-type]
            src_lang=str(record["src_lang"]),
            transcription=str(record["transcription"]),
            translation=str(record["translation"]),
            text_encoder=text_encoder,
        ),
        build_performance_sample(
            **common,  # type: ignore[arg-type]
            translation=str(record["translation"]),
            text_encoder=text_encoder,
        ),
    ]
    if include_direct:
        samples.append(build_direct_s2st_sample(**common))  # type: ignore[arg-type]
    return samples
