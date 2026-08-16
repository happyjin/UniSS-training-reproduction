from __future__ import annotations

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_CAUSAL_FULL_ASR,
    LOSS_OFFLINE_ASR_REPLAY,
    LOSS_PHASE3_REPLAY,
    LOSS_STREAMING_ASR,
    PACK_SCHEMA,
    build_stage_a_sample,
    pack_stage_a_samples,
    supervision_kind,
)
from training import constants_uniss as c


def _id_for_kind(kind: int) -> str:
    for index in range(10_000):
        value = f"sample-{kind}-{index}"
        if supervision_kind(value) == kind:
            return value
    raise AssertionError("could not find deterministic supervision bucket")


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
        "source_audio": "/tmp/not-read-during-pack.flac",
        "transcription": "Good morning.",
        "translation": "早上好。",
        "target_bicodec": list(range(24)),
        "bicodec_global": list(range(32)),
    }


def _encode(text: str) -> list[int]:
    return [1000 + (ord(value) % 1000) for value in text]


def test_supervision_buckets_build_all_four_sample_kinds() -> None:
    fixed = list(range(100, 132))
    samples = {
        kind: build_stage_a_sample(_record(kind), _encode, fixed)
        for kind in (
            LOSS_STREAMING_ASR,
            LOSS_CAUSAL_FULL_ASR,
            LOSS_OFFLINE_ASR_REPLAY,
            LOSS_PHASE3_REPLAY,
        )
    }
    assert samples[LOSS_STREAMING_ASR].task == "streaming_asr"
    assert samples[LOSS_CAUSAL_FULL_ASR].task == "asr"
    assert samples[LOSS_OFFLINE_ASR_REPLAY].acoustic is None
    assert samples[LOSS_PHASE3_REPLAY].task in {"quality", "performance"}
    for kind, sample in samples.items():
        assert any(value == kind for value in sample.loss_kinds)
        assert sample.labels[-1] == c.TOKEN_EOS


def test_streaming_pack_preserves_all_glm_and_byte_ctc_sidecars() -> None:
    sample = build_stage_a_sample(
        _record(LOSS_STREAMING_ASR),
        _encode,
        list(range(100, 132)),
    )
    assert sample.acoustic is not None
    assert sample.acoustic["source_glm"] == list(range(20))
    assert len(sample.acoustic["glm_positions"]) == 20
    assert bytes(sample.acoustic["ctc_ids"]).decode("utf-8") == "Good morning"
    packed = list(pack_stage_a_samples([sample, sample], seq_length=512))
    assert len(packed) == 1
    value = packed[0]
    assert value["schema_version"] == PACK_SCHEMA
    assert len(value["tokens"]) == 512
    assert len(value["sample_boundaries"]) == 2
    assert len(value["acoustics"]) == 2
    first, second = value["acoustics"]
    shift = value["sample_boundaries"][1][0]
    assert second["glm_positions"][0] == first["glm_positions"][0] + shift


def test_pack_splits_without_crossing_18k_boundary() -> None:
    sample = build_stage_a_sample(
        _record(LOSS_CAUSAL_FULL_ASR),
        _encode,
        list(range(100, 132)),
    )
    packed = list(pack_stage_a_samples([sample] * 20, seq_length=256))
    assert len(packed) > 1
    assert all(len(value["tokens"]) == 256 for value in packed)
    assert sum(len(value["source_ids"]) for value in packed) == 20
