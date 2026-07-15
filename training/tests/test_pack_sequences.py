import json
import tempfile
import unittest
from pathlib import Path

from training import constants_uniss as c
from training import pack_sequences as p


class PackSequencesTest(unittest.TestCase):
    def test_shifted_sample_loss_mask_alignment(self):
        sample = {
            "id": "toy",
            "task": "asr",
            "prompt_ids": [10, 11],
            "target_ids": [20, 21, c.TOKEN_EOS],
        }
        shifted = p.make_shifted_sample(sample)
        self.assertEqual(shifted.tokens, [10, 11, 20, 21])
        self.assertEqual(shifted.labels, [11, 20, 21, c.TOKEN_EOS])
        self.assertEqual(shifted.loss_mask, [0, 1, 1, 1])
        self.assertEqual(shifted.position_ids, [0, 1, 2, 3])

    def test_pack_resets_positions_and_pads(self):
        s1 = p.make_shifted_sample(
            {"id": "a", "task": "quality", "prompt_ids": [1, 2], "target_ids": [3, 4]}
        )
        s2 = p.make_shifted_sample(
            {"id": "b", "task": "performance", "prompt_ids": [5, 6], "target_ids": [7]}
        )
        packed = list(p.pack_shifted_samples([s1, s2], seq_length=8))
        self.assertEqual(len(packed), 1)
        item = packed[0]
        self.assertEqual(item.sample_boundaries, [(0, 3), (3, 5)])
        self.assertEqual(item.position_ids[:5], [0, 1, 2, 0, 1])
        self.assertEqual(item.loss_mask[:5], [0, 1, 1, 0, 1])
        self.assertEqual(item.loss_mask[5:], [0, 0, 0])
        self.assertEqual(item.tokens[5:], [c.TOKEN_PAD, c.TOKEN_PAD, c.TOKEN_PAD])
        self.assertEqual(item.tasks, ["quality", "performance"])

    def test_dense_attention_mask_blocks_cross_sample_attention(self):
        mask = p.build_dense_attention_mask([(0, 3), (3, 5)], seq_length=6)
        self.assertEqual(mask[2][:3], [1, 1, 1])
        self.assertEqual(mask[4][3:5], [1, 1])
        self.assertEqual(mask[4][0:3], [0, 0, 0])
        self.assertEqual(mask[5], [0, 0, 0, 0, 0, 0])

    def test_overlong_policy(self):
        sample = p.make_shifted_sample(
            {"id": "long", "task": "x", "prompt_ids": [1], "target_ids": [2, 3, 4]}
        )
        with self.assertRaises(ValueError):
            list(p.pack_shifted_samples([sample], seq_length=2))
        self.assertEqual(list(p.pack_shifted_samples([sample], seq_length=2, drop_overlong=True)), [])

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "samples.jsonl"
            output_path = Path(tmp) / "packed.jsonl"
            input_path.write_text(
                json.dumps({"id": "a", "task": "x", "prompt_ids": [1, 2], "target_ids": [3]})
                + "\n",
                encoding="utf-8",
            )
            samples = [p.make_shifted_sample(sample) for sample in p.load_jsonl_samples(input_path)]
            count = p.write_packed_jsonl(p.pack_shifted_samples(samples, seq_length=4), output_path)
            self.assertEqual(count, 1)
            packed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(packed["sample_boundaries"], [[0, 2]])
            self.assertEqual(packed["loss_mask"], [0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
