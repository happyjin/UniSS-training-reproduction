from __future__ import annotations

import json
import tempfile
import unittest
from array import array
from pathlib import Path

import soundfile as sf
import torch

from training.simul_uniss.jsonl_index import write_index
from training.simul_uniss.subsecond_v2.stage_b_v2_data import (
    StageBV2SidecarDataset,
    collate_stage_b_v2,
)


class _Tokenizer:
    ctc_vocab_size = 32

    def encode_ctc(self, text: str) -> list[int]:
        return [1, 2]


class StageBV2DataTest(unittest.TestCase):
    def test_reads_mmap_sidecar_and_collates_hidden_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, [0.0] * 16_000, 16_000)
            source = root / "source.jsonl"
            source_row = {
                "id": "sample",
                "source_audio": str(audio),
                "transcription": "hello",
                "src_lang": "eng",
                "source_words": [{"text": "hello", "end_ms": 400}],
                "target_support": [{"support_end_ms": 400}],
            }
            source.write_text(json.dumps(source_row) + "\n")
            shard = root / "shard.pt"
            torch.save(
                {
                    "target_tokens": torch.tensor([3, 4, 5]),
                    "full_reference_tokens": torch.tensor([6, 7, 8]),
                    "stability": torch.tensor([1, 0, 1], dtype=torch.uint8),
                    "pre_vq_hidden": torch.randn(3, 8).bfloat16(),
                    "topk_ids": torch.tensor([[3, 2], [4, 1], [5, 0]]),
                    "topk_distances": torch.zeros(3, 2).half(),
                },
                shard,
            )
            sidecar = root / "sidecar.jsonl"
            row = {
                "source_manifest_offset": 0,
                "shard_path": str(shard),
                "target_start": 0,
                "target_end": 3,
                "reference_start": 0,
                "reference_end": 3,
            }
            sidecar.write_text(json.dumps(row) + "\n")
            write_index(sidecar, array("Q", [0]))
            dataset = StageBV2SidecarDataset(
                sidecar,
                source,
                _Tokenizer(),  # type: ignore[arg-type]
                prefix_training=False,
                max_audio_seconds=1,
            )
            value = dataset[0]
            self.assertEqual(value["target_ids"].tolist(), [3, 4, 5])
            self.assertTrue(bool(value["has_teacher_hidden"]))
            batch = collate_stage_b_v2([value, value])
            self.assertEqual(batch["teacher_hidden"].shape, (2, 3, 8))
            self.assertEqual(batch["topk_ids"].shape, (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
