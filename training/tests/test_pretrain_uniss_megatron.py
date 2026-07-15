import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training import constants_uniss as c
from training import pretrain_uniss_megatron as m


class PretrainUniSSMegatronTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "sft": True,
            "create_attention_mask_in_dataloader": False,
            "context_parallel_size": 1,
            "full_validation": False,
            "eval_iters": 0,
            "uniss_packed_valid": None,
            "uniss_packed_test": None,
            "vocab_size": c.VOCAB_SIZE,
            "uniss_strict_paper_config": False,
            "seq_length": 6,
            "global_batch_size": 1,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _packed_path(self, tmpdir: str, name: str) -> str:
        path = Path(tmpdir) / name
        item = {
            "tokens": [10, 11, 20, 30, 31, c.TOKEN_PAD],
            "labels": [11, 20, 21, 31, 32, c.TOKEN_PAD],
            "loss_mask": [0, 1, 1, 0, 1, 0],
            "position_ids": [0, 1, 2, 0, 1, 0],
            "sample_boundaries": [[0, 3], [3, 5]],
        }
        path.write_text(json.dumps(item) + "\n", encoding="utf-8")
        return str(path)

    def test_add_uniss_args(self):
        parser = argparse.ArgumentParser()
        m.add_uniss_args(parser)
        args = parser.parse_args(["--uniss-packed-train", "train.jsonl"])
        self.assertEqual(args.uniss_packed_train, "train.jsonl")
        self.assertIsNone(args.uniss_packed_valid)
        self.assertFalse(args.uniss_strict_paper_config)

    def test_argparse_boolean_optional_action_compat(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", action=argparse.BooleanOptionalAction, type=bool, default=False)
        self.assertTrue(parser.parse_args(["--flag"]).flag)
        self.assertFalse(parser.parse_args(["--no-flag"]).flag)

    def test_argparse_percent_help_compat(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--warmup", help="Use 5% warm-up.")
        self.assertIn("Use 5% warm-up.", parser.format_help())

    def test_validate_requires_sft(self):
        with self.assertRaisesRegex(ValueError, "--sft"):
            m.validate_uniss_args(self._args(sft=False))

    def test_validate_requires_valid_when_eval_is_enabled(self):
        with self.assertRaisesRegex(ValueError, "--uniss-packed-valid"):
            m.validate_uniss_args(self._args(eval_iters=100))

    def test_validate_strict_paper_config(self):
        with self.assertRaisesRegex(ValueError, "--seq-length 18000"):
            m.validate_uniss_args(self._args(uniss_strict_paper_config=True))

    def test_build_uniss_packed_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train = self._packed_path(tmpdir, "train.jsonl")
            valid = self._packed_path(tmpdir, "valid.jsonl")
            args = self._args(
                uniss_packed_train=train,
                uniss_packed_valid=valid,
                eval_iters=1,
            )
            m.validate_uniss_args(args)
            train_ds, valid_ds, test_ds = m.build_uniss_packed_datasets(args)
            self.assertEqual(len(train_ds), 1)
            self.assertEqual(len(valid_ds), 1)
            self.assertIsNone(test_ds)
            self.assertEqual(train_ds.split, m.Split.train)
            self.assertEqual(valid_ds.split, m.Split.valid)


if __name__ == "__main__":
    unittest.main()
