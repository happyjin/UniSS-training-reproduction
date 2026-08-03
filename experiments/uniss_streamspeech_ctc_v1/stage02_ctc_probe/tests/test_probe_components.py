import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

from dataset import CTCProbeDataset, DistributedContiguousBatchSampler, collate_probe
from model import CTCProbeConfig, LanguageConditionalCTCProbe


class ProbeComponentsTest(unittest.TestCase):
    def test_model_has_four_conditioned_heads(self) -> None:
        model = LanguageConditionalCTCProbe(
            CTCProbeConfig(hidden_size=8, eng_vocab_size=11, cmn_vocab_size=13)
        )
        hidden = torch.randn(2, 5, 8)
        self.assertEqual(tuple(model(hidden, "asr_eng").shape), (2, 5, 12))
        self.assertEqual(tuple(model(hidden, "nar_s2tt_cmn").shape), (2, 5, 14))
        self.assertEqual(set(model.heads), {"asr_eng", "asr_cmn", "nar_s2tt_eng", "nar_s2tt_cmn"})

    def test_contiguous_sampler_is_distributed_without_overlap(self) -> None:
        left = list(DistributedContiguousBatchSampler(64, 8, 0, 2, shuffle=False))
        right = list(DistributedContiguousBatchSampler(64, 8, 1, 2, shuffle=False))
        self.assertFalse({tuple(x) for x in left}.intersection(tuple(x) for x in right))
        self.assertEqual(sum(map(len, left + right)), 64)

    def test_dataset_reads_only_referenced_hidden_slice(self) -> None:
        with tempfile.TemporaryDirectory(
            dir="/opt/dlami/nvme/jasonleeeli"
        ) as temporary:
            root = Path(temporary)
            shard = root / "shard.pt"
            torch.save({"pre_vq_hidden": torch.arange(48).reshape(6, 8).to(torch.bfloat16)}, shard)
            manifest = root / "train.jsonl"
            row = {
                "id": "x",
                "direction": "eng->cmn",
                "shard_path": str(shard),
                "hidden_start": 1,
                "hidden_end": 4,
                "hidden_frames": 3,
                "source_token_ids": [1, 2],
                "target_token_ids": [3, 4, 5],
            }
            encoded = (json.dumps(row) + "\n").encode()
            manifest.write_bytes(encoded)
            offsets = root / "train.jsonl.offsets.bin"
            offsets.write_bytes((0).to_bytes(8, "little"))
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {
                        "parts": {
                            "train": [{"manifest": str(manifest), "offsets": str(offsets), "records": 1}],
                            "valid": [],
                        }
                    }
                )
            )
            dataset = CTCProbeDataset(index, "train")
            item = dataset[0]
            self.assertEqual(tuple(item["hidden"].shape), (3, 8))
            batch = collate_probe([item])
            self.assertEqual(tuple(batch["hidden"].shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main()

