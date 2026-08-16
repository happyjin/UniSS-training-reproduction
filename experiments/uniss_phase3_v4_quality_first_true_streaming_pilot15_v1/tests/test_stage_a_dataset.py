from __future__ import annotations

import json
import wave
from array import array
from pathlib import Path

import torch

from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.packing import (
    LOSS_STREAMING_ASR,
    build_stage_a_sample,
    pack_stage_a_samples,
    supervision_kind,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v1.stage_a_causal_whisper_asr.training.dataset import (
    IndexedStageAPackDataset,
    PaddedStageAValidationDataset,
    ThreeEpochStageASchedule,
    collate_stage_a,
)
from training.simul_uniss.jsonl_index import write_index


def _stream_id() -> str:
    return next(
        value
        for index in range(1000)
        if supervision_kind(value := f"stream-{index}") == LOSS_STREAMING_ASR
    )


def _record(audio: Path) -> dict[str, object]:
    return {
        "id": _stream_id(),
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "source_duration_ms": 160,
        "source_words": [{"text": "Hi", "start_ms": 0, "end_ms": 160}],
        "source_glm": [1, 2],
        "source_glm_end_ms": [80, 160],
        "source_audio": str(audio),
        "transcription": "Hi",
        "translation": "你好",
        "target_bicodec": [1, 2],
        "bicodec_global": list(range(32)),
    }


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes((torch.zeros(2560, dtype=torch.int16).numpy()).tobytes())


def test_indexed_dataset_loads_bounded_audio_and_epoch_shuffle(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    sample = build_stage_a_sample(_record(audio), lambda text: [100 + len(text)], list(range(32)))
    packed = list(pack_stage_a_samples([sample] * 3, seq_length=512))[0]
    path = tmp_path / "packs.jsonl"
    encoded = (json.dumps(packed, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    write_index(path, array("Q", [0]))
    dataset = IndexedStageAPackDataset(
        path,
        seq_length=512,
        max_acoustics_per_pack=2,
    )
    item = dataset.get_for_epoch(0, 1)
    assert item["selected_acoustics"] == 2
    assert item["disabled_acoustics"] == 1
    original_supervised = sum(packed["loss_mask"])
    assert int(item["loss_mask"].sum()) < original_supervised
    assert not bool(
        ((item["loss_mask"] > 0) & (item["loss_kinds"] == 0)).any()
    )
    batch = collate_stage_a([item])
    assert tuple(batch["waveform"].shape) == (2, 2560)
    assert batch["ctc_lengths"].tolist() == [2, 2]
    assert batch["disabled_acoustics"].tolist() == [1]
    assert batch["acoustic_sample_ids"] == [_stream_id(), _stream_id()]
    assert batch["source_audio_paths"] == [str(audio), str(audio)]
    assert batch["acoustic_source_duration_ms"].tolist() == [160, 160]
    schedule = ThreeEpochStageASchedule(
        dataset,
        coverage_epochs=3,
        data_parallel_group_size=1,
        global_batch_size=1,
        shuffle_seed=20260816,
    )
    assert len(schedule) == 3
    assert [schedule.source_index(index)[0] for index in range(3)] == [0, 1, 2]


def test_validation_repeats_to_complete_dp_microbatches(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio)
    sample = build_stage_a_sample(
        _record(audio), lambda text: [100 + len(text)], list(range(32))
    )
    packed = list(pack_stage_a_samples([sample], seq_length=512))[0]
    path = tmp_path / "valid.jsonl"
    encoded = (json.dumps(packed, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    write_index(path, array("Q", [0]))
    source = IndexedStageAPackDataset(
        path,
        seq_length=512,
        max_acoustics_per_pack=1,
    )
    valid = PaddedStageAValidationDataset(
        source,
        minimum_samples=4,
        data_parallel_group_size=8,
    )
    assert valid.unpadded_length == 1
    assert len(valid) == 8
    assert all(valid[index]["source_pack_index"] == 0 for index in range(8))
