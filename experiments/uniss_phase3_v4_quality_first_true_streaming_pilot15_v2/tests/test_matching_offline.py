from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage00_matching_offline.build_manifest import (
    assemble,
)
from experiments.uniss_phase3_v4_quality_first_true_streaming_pilot15_v2.stage00_matching_offline.merge_offline_asr import (
    edit_distance,
)
from evaluation.text_metrics import normalize_for_bleu


def test_assemble_preserves_exact_source_identity() -> None:
    selected = [
        {
            "id": "sample",
            "task": "streaming_asr",
            "src_lang": "eng",
            "transcription": "hello world",
            "source_audio": "/tmp/audio.flac",
            "pack_index": 3,
            "acoustic_position": 1,
        }
    ]
    sources = {
        "sample": {
            "parquet_path": "/tmp/train.parquet",
            "row_index": 7,
            "formal_input_index": 8,
            "src_lang": "eng",
            "tgt_lang": "cmn",
            "transcription": "Hello, world.",
            "source_audio": "/tmp/audio.flac",
        }
    }
    rows = assemble(selected, sources)
    assert rows[0]["transcription"] == "hello world"
    assert rows[0]["source_transcription"] == "Hello, world."
    assert rows[0]["worker_rank"] == 0
    assert rows[0]["row_index"] == 7


def test_edit_distance_handles_insert_delete_replace() -> None:
    assert edit_distance(list("abc"), list("adc")) == 1
    assert edit_distance(list("abc"), list("ab")) == 1
    assert edit_distance(list("ab"), list("abc")) == 1


def test_chinese_error_units_are_characters() -> None:
    assert normalize_for_bleu("你好，世界！", "cmn").split() == ["你", "好", "世", "界"]
