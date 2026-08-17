from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_CAUSAL_FULL_ASR,
    LOSS_STREAMING_ASR,
    build_stage_a_sample,
    pack_stage_a_samples,
    supervision_kind,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage_a_causal_whisper_asr.same_prefix_teacher import (
    fixed_speaker_from_pack,
    requests_for_acoustic,
)
from training import constants_uniss as c


def _id_for_kind(kind: int) -> str:
    for index in range(10_000):
        value = f"teacher-{kind}-{index}"
        if supervision_kind(value) == kind:
            return value
    raise AssertionError("missing deterministic supervision ID")


def _record(kind: int) -> dict[str, object]:
    return {
        "id": _id_for_kind(kind),
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "source_duration_ms": 1600,
        "source_words": [
            {"text": "Good", "start_ms": 80, "end_ms": 320},
            {"text": "morning", "start_ms": 480, "end_ms": 960},
        ],
        "source_glm": list(range(20)),
        "source_glm_end_ms": [80 * (index + 1) for index in range(20)],
        "source_audio": "/tmp/teacher.flac",
        "transcription": "Good morning.",
        "translation": "早上好。",
        "target_bicodec": list(range(24)),
        "bicodec_global": list(range(32)),
    }


def _encode(text: str) -> list[int]:
    return [1000 + (ord(value) % 1000) for value in text]


def _decode(ids) -> str:
    return "".join(chr(int(value) - 1000) for value in ids)


def _pack(kind: int) -> tuple[dict[str, object], dict[str, object]]:
    sample = build_stage_a_sample(
        _record(kind), _encode, list(range(100, 132))
    )
    pack = list(pack_stage_a_samples([sample], seq_length=512))[0]
    return pack, pack["acoustics"][0]


def test_streaming_teacher_uses_growing_prefix_and_only_delta_positions() -> None:
    pack, acoustic = _pack(LOSS_STREAMING_ASR)
    requests = requests_for_acoustic(
        pack,
        acoustic,
        fixed_speaker=list(range(100, 132)),
        encode_text=_encode,
        decode_text=_decode,
    )
    assert len(requests) == 2
    assert requests[0].visible_glm_tokens < requests[1].visible_glm_tokens
    # The final 640 ms contains no newly aligned word, so its trailing GLM
    # tokens are intentionally excluded from the last text-event teacher.
    assert requests[-1].visible_glm_tokens < len(acoustic["source_glm"])
    assert all(
        request.prompt_ids[:2]
        == (c.TOKEN_TASK_S2S_TRANSLATION, c.TOKEN_SLOW_MODE)
        for request in requests
    )
    assert all(
        request.reference_labels[-1] == c.TOKEN_END_CONTENT for request in requests
    )
    selected_positions = [
        position for request in requests for position in request.student_positions
    ]
    active_streaming = [
        index
        for index, (mask, kind) in enumerate(zip(pack["loss_mask"], pack["loss_kinds"]))
        if mask and kind == LOSS_STREAMING_ASR
    ]
    assert set(selected_positions) < set(active_streaming)
    assert len(selected_positions) == len(set(selected_positions))


def test_causal_full_teacher_matches_the_complete_active_asr_target() -> None:
    pack, acoustic = _pack(LOSS_CAUSAL_FULL_ASR)
    request = requests_for_acoustic(
        pack,
        acoustic,
        fixed_speaker=list(range(100, 132)),
        encode_text=_encode,
        decode_text=_decode,
    )[0]
    active = [
        index
        for index, (mask, kind) in enumerate(zip(pack["loss_mask"], pack["loss_kinds"]))
        if mask and kind == LOSS_CAUSAL_FULL_ASR
    ]
    assert list(request.student_positions) == active
    assert list(request.reference_labels) == [pack["labels"][index] for index in active]
    assert request.target_ids[-2:] == (c.TOKEN_END_CONTENT, c.TOKEN_EOS)


def test_fixed_speaker_is_recovered_from_each_immutable_pack_prompt() -> None:
    expected = tuple(range(100, 132))
    for kind in (LOSS_STREAMING_ASR, LOSS_CAUSAL_FULL_ASR):
        pack, acoustic = _pack(kind)
        assert fixed_speaker_from_pack(pack, acoustic) == expected
