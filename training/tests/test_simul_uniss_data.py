from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training import constants_uniss as c
from training.simul_uniss.dataset import packed_json_to_item
from training.simul_uniss.pack_sequences import make_shifted_sample, pack_samples
from training.simul_uniss.sample_builders import ACTION_WEIGHT, SEMANTIC_WEIGHT, build_interleaved_sample
from training.simul_uniss.schedule import build_pseudo_schedule


def fake_encoder(text: str) -> list[int]:
    return [1000 + index for index, _ in enumerate(text.encode("utf-8"))]


def fake_record() -> dict[str, object]:
    return {
        "id": "sample-1",
        "transcription": "hello world",
        "translation": "你好，世界。",
        "source_glm": list(range(20)),
        "source_bicodec": list(range(80)),
        "target_bicodec": list(range(60)),
        "bicodec_global": list(range(32)),
        "src_lang": "eng",
        "tgt_lang": "cmn",
        "dataset_name": "test",
        "split": "train",
    }


class ScheduleTests(unittest.TestCase):
    def test_schedule_covers_source_and_target(self) -> None:
        record = fake_record()
        schedule = build_pseudo_schedule(record, fake_encoder, chunk_ms=640, wait_k_chunks=1)
        events = schedule["events"]
        source = [token for event in events for token in event["source_glm"]]
        target = [
            token
            for event in events
            if event["action"] == "write"
            for token in event["target_semantic"]
        ]
        self.assertEqual(source, record["source_glm"])
        self.assertEqual(target, record["target_bicodec"])
        self.assertTrue(events[-1]["source_is_final"])
        self.assertEqual(events[-1]["action"], "write")

    def test_interleaved_sample_reuses_reserved_tokens(self) -> None:
        schedule = build_pseudo_schedule(fake_record(), fake_encoder, chunk_ms=640, wait_k_chunks=1)
        sample = build_interleaved_sample(schedule)
        self.assertEqual(sample.input_ids[0], c.TOKEN_TASK_STREAMING_S2ST)
        self.assertIn(c.TOKEN_WAIT_READ, sample.input_ids)
        self.assertIn(c.TOKEN_WRITE_GENERATE, sample.input_ids)
        self.assertEqual(sample.input_ids[-1], c.TOKEN_EOS)
        wait_index = sample.input_ids.index(c.TOKEN_WAIT_READ)
        self.assertEqual(sample.token_weights[wait_index], ACTION_WEIGHT)
        semantic_id = c.BICODEC_SEMANTIC_OFFSET
        semantic_index = sample.input_ids.index(semantic_id)
        self.assertEqual(sample.token_weights[semantic_index], SEMANTIC_WEIGHT)

    def test_weighted_pack_and_dataset(self) -> None:
        schedule = build_pseudo_schedule(fake_record(), fake_encoder, chunk_ms=640, wait_k_chunks=1)
        sample = build_interleaved_sample(schedule).to_json()
        shifted = make_shifted_sample(sample)
        seq_length = shifted.length + 8
        packed = next(pack_samples([shifted], seq_length))
        self.assertTrue(any(value == ACTION_WEIGHT for value in packed["loss_mask"]))
        item = packed_json_to_item(packed, seq_length)
        self.assertEqual(tuple(item["tokens"].shape), (seq_length,))
        self.assertEqual(item["loss_mask"].dtype.is_floating_point, True)
        self.assertEqual(int(item["cu_seqlens"][1]), shifted.length)


if __name__ == "__main__":
    unittest.main()
