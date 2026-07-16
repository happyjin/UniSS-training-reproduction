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
            "add_bias_linear": False,
            "add_qkv_bias": True,
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

    def test_torch_checkpoint_no_dist_compat_drops_legacy_kwarg(self):
        import torch.distributed.checkpoint as checkpoint

        original_load = checkpoint.load
        calls = []

        def fake_load(state_dict, *, storage_reader=None):
            calls.append((state_dict, storage_reader))
            return "loaded"

        try:
            checkpoint.load = fake_load
            m.patch_torch_distributed_checkpoint_no_dist()
            result = checkpoint.load({"common": object()}, storage_reader="reader", no_dist=True)
        finally:
            checkpoint.load = original_load

        self.assertEqual(result, "loaded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], "reader")

    def test_validate_requires_sft(self):
        with self.assertRaisesRegex(ValueError, "--sft"):
            m.validate_uniss_args(self._args(sft=False))

    def test_validate_rejects_dense_attention_mask(self):
        with self.assertRaisesRegex(ValueError, "dense attention masks"):
            m.validate_uniss_args(self._args(create_attention_mask_in_dataloader=True))

    def test_validate_requires_valid_when_eval_is_enabled(self):
        with self.assertRaisesRegex(ValueError, "--uniss-packed-valid"):
            m.validate_uniss_args(self._args(eval_iters=100))

    def test_validate_strict_paper_config(self):
        with self.assertRaisesRegex(ValueError, "--seq-length 18000"):
            m.validate_uniss_args(self._args(uniss_strict_paper_config=True))

    def test_validate_strict_paper_config_rejects_qwen_bias_mismatch(self):
        strict_args = {
            "uniss_strict_paper_config": True,
            "seq_length": m.PAPER_SEQ_LENGTH,
            "global_batch_size": m.PAPER_GLOBAL_BATCH_SEQUENCES,
        }
        with self.assertRaisesRegex(ValueError, "--disable-bias-linear"):
            m.validate_uniss_args(self._args(**strict_args, add_bias_linear=True))
        with self.assertRaisesRegex(ValueError, "--add-qkv-bias"):
            m.validate_uniss_args(self._args(**strict_args, add_qkv_bias=False))

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
            train_ds, valid_ds, test_ds = m.build_uniss_packed_datasets(args, train_val_test_num_samples=[3, 2, 1])
            self.assertEqual(len(train_ds), 3)
            self.assertEqual(len(valid_ds), 2)
            self.assertIsNone(test_ds)
            self.assertEqual(train_ds.split, m.Split.train)
            self.assertEqual(valid_ds.split, m.Split.valid)
            self.assertEqual(train_ds[0]["tokens"].tolist(), train_ds[2]["tokens"].tolist())


if __name__ == "__main__":
    unittest.main()
