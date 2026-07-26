"""Parse text and semantic regions from UniSS generated token sequences."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from training import constants_uniss as c


TextDecoder = Callable[[Sequence[int]], str]


def _first_index(values: Sequence[int], token_id: int, start: int = 0) -> int | None:
    for index in range(start, len(values)):
        if int(values[index]) == token_id:
            return index
    return None


def _decode(ids: Sequence[int], decoder: TextDecoder) -> str:
    return decoder([int(token_id) for token_id in ids]).strip()


def parse_generated_fields(
    token_ids: Sequence[int],
    *,
    mode: str,
    text_decoder: TextDecoder,
) -> dict[str, object]:
    ids = [int(token_id) for token_id in token_ids]
    transcription: str | None = None
    translation: str | None = None

    if mode == "quality":
        transcription_end = _first_index(ids, c.TOKEN_END_CONTENT)
        if transcription_end is not None:
            transcription = _decode(ids[:transcription_end], text_decoder)
            translation_start = _first_index(ids, c.TOKEN_START_CONTENT, transcription_end + 1)
            if translation_start is not None:
                translation_end = _first_index(ids, c.TOKEN_END_CONTENT, translation_start + 1)
                if translation_end is not None:
                    translation = _decode(ids[translation_start + 1 : translation_end], text_decoder)
    elif mode == "performance":
        translation_end = _first_index(ids, c.TOKEN_END_CONTENT)
        if translation_end is not None:
            translation = _decode(ids[:translation_end], text_decoder)

    semantic_start = _first_index(ids, c.TOKEN_START_SEMANTIC)
    semantic_end = _first_index(ids, c.TOKEN_END_SEMANTIC, (semantic_start or -1) + 1)
    semantic_ids = [
        c.BICODEC_SEMANTIC_SPAN.value_for(token_id)
        for token_id in ids
        if c.BICODEC_SEMANTIC_OFFSET <= token_id <= c.BICODEC_SEMANTIC_SPAN.last_id
    ]
    return {
        "generated_transcription": transcription,
        "generated_translation": translation,
        "semantic_values": semantic_ids,
        "has_semantic_start": semantic_start is not None,
        "has_semantic_end": semantic_end is not None,
        "has_eos": c.TOKEN_EOS in ids,
    }


def parse_with_tokenizer(token_ids: Sequence[int], *, mode: str, tokenizer) -> Mapping[str, object]:
    return parse_generated_fields(
        token_ids,
        mode=mode,
        text_decoder=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
    )
