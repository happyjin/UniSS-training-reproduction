"""The manifest/parquet join, and the integrity checks that guard it."""

from __future__ import annotations

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from experiments.uniss_omni_s2st_v1.data.dual_source import (
    JoinIntegrityError,
    iter_utterances,
)

TAR = "unist_bicodec_qwen3_v2/rank03/train-00001-00000.tar"


def _uri(offset, length):
    return (
        "wds://unist/audio/processed/shards/dataset=unist/"
        f"process_version={TAR}?offset={offset}&length={length}&format=flac"
    )


def _write_audio(root, waveform):
    buffer = io.BytesIO()
    sf.write(buffer, waveform, 16_000, format="FLAC")
    payload = buffer.getvalue()
    path = root / TAR
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 512 + payload)
    return 512, len(payload)


def _write_parquet(root, ids, codes, globals_):
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": ids,
                "transcription": ["t"] * len(ids),
                "translation": ["x"] * len(ids),
                "target_bicodec": codes,
                "bicodec_global": globals_,
            }
        ),
        root / "train-00001.parquet",
    )


def _manifest(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _row(sample_id, offset, length, row_index, *, bicodec_len, task="speech_to_speech_translation"):
    return {
        "record_id": f"unist:x:{sample_id}",
        "id": sample_id,
        "task": task,
        "audios": [_uri(offset, length)],
        "source_lang": "en",
        "target_lang": "zh",
        "source_transcript": "hello",
        "target_text": "你好",
        "provenance": {
            "source_parquet": "train-00001.parquet",
            "source_row_index": row_index,
            "target_bicodec_len": bicodec_len,
        },
    }


@pytest.fixture
def world(tmp_path):
    audio_root = tmp_path / "speech"
    audio_root.mkdir()
    offset, length = _write_audio(audio_root, np.zeros(1600, dtype="float32"))
    parquet_root = tmp_path / "parquet"
    _write_parquet(
        parquet_root,
        ["A", "B"],
        [[1, 2, 3], [4, 5]],
        [list(range(32)), list(range(32))],
    )
    return tmp_path, audio_root, parquet_root, offset, length


def test_joins_manifest_audio_to_parquet_codes(world):
    tmp_path, audio_root, parquet_root, offset, length = world
    manifest = tmp_path / "m.jsonl"
    _manifest(manifest, [_row("A", offset, length, 0, bicodec_len=3)])

    (utterance,) = list(
        iter_utterances(manifest, audio_root=audio_root, parquet_root=parquet_root)
    )
    assert utterance.sample_id == "A"
    assert utterance.target_bicodec.tolist() == [1, 2, 3]
    assert utterance.bicodec_global.shape == (32,)
    waveform, sample_rate = utterance.waveform(audio_root)
    assert sample_rate == 16_000 and waveform.shape == (1600,)


def test_skips_records_whose_tar_is_not_present(world):
    tmp_path, audio_root, parquet_root, offset, length = world
    absent = _row("A", offset, length, 0, bicodec_len=3)
    absent["audios"] = [absent["audios"][0].replace("00000.tar", "09999.tar")]
    manifest = tmp_path / "m.jsonl"
    _manifest(manifest, [absent])

    assert (
        list(iter_utterances(manifest, audio_root=audio_root, parquet_root=parquet_root))
        == []
    )


def test_skips_other_tasks(world):
    tmp_path, audio_root, parquet_root, offset, length = world
    manifest = tmp_path / "m.jsonl"
    _manifest(
        manifest,
        [_row("A", offset, length, 0, bicodec_len=3, task="automatic_speech_recognition")],
    )
    assert (
        list(iter_utterances(manifest, audio_root=audio_root, parquet_root=parquet_root))
        == []
    )


def test_a_row_index_pointing_at_the_wrong_utterance_is_an_error(world):
    """The join is positional; the id is what proves it landed correctly."""
    tmp_path, audio_root, parquet_root, offset, length = world
    manifest = tmp_path / "m.jsonl"
    _manifest(manifest, [_row("A", offset, length, 1, bicodec_len=3)])

    with pytest.raises(JoinIntegrityError, match="manifest says"):
        list(iter_utterances(manifest, audio_root=audio_root, parquet_root=parquet_root))


def test_a_length_disagreement_is_an_error(world):
    tmp_path, audio_root, parquet_root, offset, length = world
    manifest = tmp_path / "m.jsonl"
    _manifest(manifest, [_row("A", offset, length, 0, bicodec_len=99)])

    with pytest.raises(JoinIntegrityError, match="provenance says"):
        list(iter_utterances(manifest, audio_root=audio_root, parquet_root=parquet_root))


def test_limit_stops_early(world):
    tmp_path, audio_root, parquet_root, offset, length = world
    manifest = tmp_path / "m.jsonl"
    _manifest(
        manifest,
        [
            _row("A", offset, length, 0, bicodec_len=3),
            _row("B", offset, length, 1, bicodec_len=2),
        ],
    )
    rows = list(
        iter_utterances(
            manifest, audio_root=audio_root, parquet_root=parquet_root, limit=1
        )
    )
    assert [r.sample_id for r in rows] == ["A"]
