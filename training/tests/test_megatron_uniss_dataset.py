import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training import constants_uniss as c
from training import megatron_uniss_dataset as d


class MegatronUniSSDatasetTest(unittest.TestCase):
    def _packed_item(self):
        return {
            "tokens": [10, 11, 20, 30, 31, c.TOKEN_PAD],
            "labels": [11, 20, 21, 31, 32, c.TOKEN_PAD],
            "loss_mask": [0, 1, 1, 0, 1, 0],
            "position_ids": [0, 1, 2, 0, 1, 0],
            "sample_boundaries": [[0, 3], [3, 5]],
            "tasks": ["quality", "performance"],
            "source_ids": ["a", "b"],
        }

    def test_cu_seqlens_from_boundaries(self):
        cu_seqlens, max_seqlen = d.boundaries_to_padded_cu_seqlens([[0, 3], [3, 5]], 6)
        self.assertEqual(cu_seqlens.tolist(), [0, 3, 5, 6, 6, 6, 6])
        self.assertEqual(max_seqlen.item(), 3)

    def test_max_seqlen_includes_padding_tail(self):
        cu_seqlens, max_seqlen = d.boundaries_to_padded_cu_seqlens([[0, 2]], 6)
        self.assertEqual(cu_seqlens.tolist(), [0, 2, 6, 6, 6, 6, 6])
        self.assertEqual(max_seqlen.item(), 4)

    def test_megatron_item_tensors(self):
        item = d.packed_json_to_megatron_item(self._packed_item(), seq_length=6)
        self.assertEqual(item["tokens"].dtype, torch.int64)
        self.assertEqual(item["labels"].dtype, torch.int64)
        self.assertEqual(item["loss_mask"].dtype, torch.float32)
        self.assertEqual(item["position_ids"].dtype, torch.int64)
        self.assertEqual(item["cu_seqlens"].dtype, torch.int32)
        self.assertEqual(item["max_seqlen"].dtype, torch.int32)
        self.assertEqual(item["loss_mask"].tolist(), [0, 1, 1, 0, 1, 0])

    def test_jsonl_dataset_and_default_collate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packed.jsonl"
            path.write_text(json.dumps(self._packed_item()) + "\n", encoding="utf-8")
            dataset = d.UniSSPackedJsonlDataset(path, seq_length=6)
            self.assertEqual(len(dataset), 1)
            batch = next(iter(DataLoader(dataset, batch_size=1)))
            self.assertEqual(batch["tokens"].shape, (1, 6))
            self.assertEqual(batch["cu_seqlens"].shape, (1, 7))
            self.assertEqual(batch["max_seqlen"].shape, (1,))

    def test_invalid_boundaries(self):
        with self.assertRaises(ValueError):
            d.boundaries_to_padded_cu_seqlens([[0, 3], [4, 5]], 6)
        with self.assertRaises(ValueError):
            d.boundaries_to_padded_cu_seqlens([[0, 7]], 6)


if __name__ == "__main__":
    unittest.main()
