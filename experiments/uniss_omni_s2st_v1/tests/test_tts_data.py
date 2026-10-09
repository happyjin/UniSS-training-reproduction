"""Row loading and the prompt/reply batch layout."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.uniss_omni_s2st_v1.training.tts_data import (
    build_thinker_batch,
    read_rows,
    tts_prompt,
)


def _parquet(tmp_path, ids, texts, langs, codes):
    path = tmp_path / "train-00001.parquet"
    pq.write_table(
        pa.table(
            {"id": ids, "translation": texts, "tgt_lang": langs, "target_bicodec": codes}
        ),
        path,
    )
    return path


def test_reads_text_and_codes(tmp_path):
    path = _parquet(tmp_path, ["A"], ["你好"], ["cmn"], [[1, 2, 3, 4, 5]])
    (row,) = read_rows([path])
    assert row["id"] == "A" and row["text"] == "你好" and row["lang"] == "cmn"
    assert row["codes"].tolist() == [1, 2, 3, 4, 5]
    assert row["codes"].dtype == np.int16


def test_drops_rows_no_stage_could_learn_from(tmp_path):
    path = _parquet(
        tmp_path,
        ["short", "empty", "long", "ok"],
        ["a", "   ", "c", "d"],
        ["cmn"] * 4,
        [[1, 2], [1, 2, 3, 4], list(range(50)), [1, 2, 3, 4, 5]],
    )
    assert [r["id"] for r in read_rows([path], min_codes=4, max_codes=10)] == ["ok"]


def test_limit_per_file_counts_kept_rows(tmp_path):
    path = _parquet(
        tmp_path, ["A", "B", "C"], ["x", "y", "z"], ["cmn"] * 3, [[1, 2, 3, 4]] * 3
    )
    assert len(read_rows([path], limit_per_file=2)) == 2


def test_prompt_puts_the_text_in_the_user_turn_with_omnis_system_prompt():
    messages = tts_prompt("你好", "cmn")
    assert messages[0]["role"] == "system"
    assert "virtual human developed by the Qwen Team" in messages[0]["content"][0]["text"]
    assert "Chinese" in messages[1]["content"][0]["text"]
    assert "你好" in messages[1]["content"][0]["text"]


def test_prompt_names_the_language_it_knows_and_passes_through_what_it_does_not():
    assert "English" in tts_prompt("hi", "eng")[1]["content"][0]["text"]
    assert "xyz" in tts_prompt("hi", "xyz")[1]["content"][0]["text"]


def test_prompts_are_left_padded_so_replies_start_together():
    batch = build_thinker_batch([[1, 2, 3], [4, 5]], [[7, 8], [9]], pad_id=0)
    assert batch.reply_start == 3
    assert batch.input_ids.tolist() == [[1, 2, 3, 7, 8], [0, 4, 5, 9, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1], [0, 1, 1, 1, 0]]


def test_padding_is_masked_on_both_sides():
    batch = build_thinker_batch([[1], [2, 3, 4]], [[5, 6], [7]], pad_id=0)
    # Left pad on row 0's prompt, right pad on row 1's reply.
    assert batch.attention_mask.tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert batch.reply_mask.tolist() == [[1, 1], [1, 0]]


def test_replies_are_returned_separately_for_embedding():
    batch = build_thinker_batch([[1, 2]], [[8, 9, 10]], pad_id=0)
    assert batch.reply_ids.tolist() == [[8, 9, 10]]


def test_mismatched_batch_sizes_are_rejected():
    with pytest.raises(ValueError, match="same batch size"):
        build_thinker_batch([[1]], [[1], [2]], pad_id=0)


def test_an_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="empty batch"):
        build_thinker_batch([], [], pad_id=0)
